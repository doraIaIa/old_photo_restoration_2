from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


OFFICIAL_LAMA_REPO = Path(r"F:\deeplearning\external_models\lama\lama")
OFFICIAL_LAMA_ENV = "lama"
OFFICIAL_LAMA_CHECKPOINT = Path(r"F:\deeplearning\external_models\lama\weights\big-lama\models\best.ckpt")
OFFICIAL_LAMA_MODEL_DIR = OFFICIAL_LAMA_CHECKPOINT.parents[1]
OFFICIAL_LAMA_BACKEND = "official_lama_pretrained"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _tail(text: str, limit: int = 2500) -> str:
    return text[-limit:] if text else ""


def _base_result(input_path: Path, mask_path: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "ok": False,
        "input": str(input_path),
        "mask": str(mask_path),
        "output": None,
        "backend": OFFICIAL_LAMA_BACKEND,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "reason": "",
        "device": "cpu",
        "checkpoint": str(OFFICIAL_LAMA_CHECKPOINT),
        "model_path": str(OFFICIAL_LAMA_MODEL_DIR),
        "repo": str(OFFICIAL_LAMA_REPO),
        "output_dir": str(output_dir),
    }


def _read_color(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _read_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Không ghi được file: {path}")


def _list_output_images(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        [path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _select_official_output(raw_output_dir: Path, input_stem: str) -> Path | None:
    preferred_names = [
        raw_output_dir / f"{input_stem}_mask.png",
        raw_output_dir / f"{input_stem}_mask.jpg",
        raw_output_dir / f"{input_stem}.png",
        raw_output_dir / f"{input_stem}.jpg",
    ]
    for candidate in preferred_names:
        if candidate.exists():
            return candidate
    images = _list_output_images(raw_output_dir)
    return images[0] if images else None


def _prepare_input(input_path: Path, mask_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    image = _read_color(input_path)
    mask = _read_mask(mask_path)
    if image is None:
        raise FileNotFoundError(f"Không đọc được input image: {input_path}")
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {mask_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Size mismatch: image H/W={image.shape[:2]}, mask H/W={mask.shape[:2]}")

    run_id = f"{input_path.stem}_{int(time.time() * 1000)}"
    prepared_dir = output_dir / "_official_lama_input" / run_id
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_image = prepared_dir / f"{input_path.stem}.png"
    prepared_mask = prepared_dir / f"{input_path.stem}_mask.png"
    _write_png(prepared_image, image)
    _write_png(prepared_mask, mask)
    return prepared_dir, prepared_image, prepared_mask


def run_official_lama_subprocess(
    input_image_path: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    """Chạy official/pretrained LaMa qua conda subprocess, không import LaMa vào env chính."""
    input_path = Path(input_image_path)
    mask = Path(mask_path)
    output_path = Path(output_dir)
    result = _base_result(input_path, mask, output_path)

    if not input_path.exists():
        result["reason"] = "input_missing"
        return result
    if not mask.exists():
        result["reason"] = "mask_missing"
        return result
    if not OFFICIAL_LAMA_REPO.exists():
        result["reason"] = "repo_missing"
        return result
    predict_script = OFFICIAL_LAMA_REPO / "bin" / "predict.py"
    if not predict_script.exists():
        result["reason"] = "predict_script_missing"
        return result
    if not OFFICIAL_LAMA_CHECKPOINT.exists():
        result["reason"] = "checkpoint_missing"
        return result

    output_path.mkdir(parents=True, exist_ok=True)
    try:
        prepared_dir, prepared_image, prepared_mask = _prepare_input(input_path, mask, output_path)
    except Exception as exc:
        result["reason"] = f"prepare_failed: {exc}"
        return result

    raw_output_dir = output_path / "official_lama_raw" / prepared_dir.name
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "conda",
        "run",
        "-n",
        OFFICIAL_LAMA_ENV,
        "python",
        str(predict_script),
        f"model.path={OFFICIAL_LAMA_MODEL_DIR}",
        f"indir={prepared_dir}",
        f"outdir={raw_output_dir}",
        "device=cpu",
    ]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(OFFICIAL_LAMA_REPO) if not existing_pythonpath else f"{OFFICIAL_LAMA_REPO}{os.pathsep}{existing_pythonpath}"

    result.update(
        {
            "prepared_input_dir": str(prepared_dir),
            "prepared_image": str(prepared_image),
            "prepared_mask": str(prepared_mask),
            "raw_output_dir": str(raw_output_dir),
            "command": command,
        }
    )
    try:
        completed = subprocess.run(
            command,
            cwd=OFFICIAL_LAMA_REPO,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        result["reason"] = "timeout"
        result["stdout_tail"] = _tail(exc.stdout if isinstance(exc.stdout, str) else "")
        result["stderr_tail"] = _tail(exc.stderr if isinstance(exc.stderr, str) else "")
        return result
    except Exception as exc:
        result["reason"] = f"subprocess_exception: {exc}"
        return result

    result["returncode"] = completed.returncode
    result["stdout_tail"] = _tail(completed.stdout)
    result["stderr_tail"] = _tail(completed.stderr)
    if completed.returncode != 0:
        result["reason"] = "subprocess_failed"
        return result

    selected_output = _select_official_output(raw_output_dir, input_path.stem)
    if selected_output is None:
        result["reason"] = "output_missing"
        return result
    final_output = output_path / "official_lama_restored.png"
    shutil.copy2(selected_output, final_output)
    result["ok"] = True
    result["reason"] = "applied"
    result["output"] = str(final_output)
    result["selected_raw_output"] = str(selected_output)
    return result
