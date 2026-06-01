from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VALID_FACE_MODES = {"off", "auto", "codeformer_if_available"}


def _base_metadata(enabled: bool, reason: str) -> dict[str, Any]:
    return {
        "face_module_enabled": enabled,
        "face_detection_backend": "none",
        "face_restoration_backend": "none",
        "faces_detected": 0,
        "face_restoration_applied": False,
        "reason": reason,
        "warning": None,
    }


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def _detect_dependency() -> tuple[str | None, str | None]:
    if Path("external/CodeFormer").exists() or Path("CodeFormer").exists():
        return "codeformer", "Đã thấy thư mục CodeFormer nhưng chưa cấu hình adapter inference ổn định."
    if importlib.util.find_spec("gfpgan") is not None:
        return "gfpgan", "Đã thấy package GFPGAN nhưng wrapper chưa được bật cho inference xác định trong project."
    if importlib.util.find_spec("facexlib") is not None:
        return "facexlib", "Đã thấy facexlib nhưng thiếu backend restoration."
    return None, None


def apply_face_restoration(
    image_rgb: np.ndarray,
    mode: str = "auto",
    strength: float = 0.5,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Áp dụng Module 3 nếu dependency/adapter đã sẵn sàng, nếu chưa thì trả ảnh gốc kèm metadata."""
    if mode not in VALID_FACE_MODES:
        raise ValueError(f"mode không hợp lệ: {mode}. Hợp lệ: {sorted(VALID_FACE_MODES)}")
    image = np.clip(np.asarray(image_rgb), 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb phải có shape HxWx3, nhận được {image.shape}")

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        _save_rgb(output_path / "face_input.png", image)

    if mode == "off":
        metadata = _base_metadata(enabled=False, reason="disabled")
        metadata["face_strength"] = float(strength)
        if output_path is not None:
            (output_path / "face_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return image, metadata

    dependency, warning = _detect_dependency()
    if dependency is None:
        metadata = _base_metadata(enabled=True, reason="dependency_not_available")
        metadata["face_strength"] = float(strength)
        if output_path is not None:
            (output_path / "face_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return image, metadata

    metadata = _base_metadata(enabled=True, reason="adapter_not_configured")
    metadata["face_detection_backend"] = dependency
    metadata["face_restoration_backend"] = dependency
    metadata["warning"] = warning
    metadata["face_strength"] = float(strength)
    if output_path is not None:
        _save_rgb(output_path / "face_output.png", image)
        (output_path / "face_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return image, metadata
