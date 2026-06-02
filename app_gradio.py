from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from src.restoration.codeformer_adapter import CODEFORMER_REPO
from src.restoration.official_lama_adapter import OFFICIAL_LAMA_CHECKPOINT, OFFICIAL_LAMA_ENV, OFFICIAL_LAMA_REPO


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "blueprint21_final_assets" / "gradio_runs"
R011_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r011-repair-ft-s42" / "best_iou.ckpt"
R012_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r012-manual-repair-ft-s42" / "best_iou.ckpt"
FINE_TUNED_LAMA_MARKERS = [
    PROJECT_ROOT / "checkpoints" / "lama" / "fine_tuned_lama",
    PROJECT_ROOT / "configs" / "lama_finetuned.yaml",
]
MASK_MODE_CHOICES = [
    ("Stable — r011 Union Refined", "auto_r011_union_refined"),
    ("Stable — r011", "auto_r011"),
    ("Stable — r011 Refined", "auto_r011_refined"),
    ("Experimental — r011 Sensitive Low Threshold / high recall", "auto_r011_sensitive_low_threshold"),
    ("Experimental — r012", "auto_r012"),
    ("Experimental — r012 Refined", "auto_r012_refined"),
    ("Experimental — r012 Union Refined", "auto_r012_union_refined"),
    ("External / Oracle Mask", "external"),
]
BACKEND_CHOICES = [
    ("simple_lama — stable fallback, faster", "simple_lama"),
    ("official_lama — pretrained Big-LaMa, CPU local, slower", "official_lama"),
    ("opencv — classical fallback", "opencv"),
]
APP_CSS = """
body, .gradio-container {
  background: #111111;
  color: #f2f0ea;
}
.gradio-container {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button {
  border-radius: 6px !important;
}
"""

try:
    import gradio as gr
except ImportError:
    gr = None


def available_mask_modes() -> list[str]:
    modes = ["auto_r011", "auto_r011_refined", "auto_r011_union_refined", "auto_r011_sensitive_low_threshold", "external"]
    if R012_CHECKPOINT.exists():
        modes.extend(["auto_r012", "auto_r012_refined", "auto_r012_union_refined"])
    return modes


def available_backends() -> list[str]:
    backends = ["simple_lama", "official_lama", "opencv"]
    if any(path.exists() for path in FINE_TUNED_LAMA_MARKERS):
        backends.append("fine_tuned_lama")
    return backends


def get_system_readiness() -> dict:
    simple_lama_available = importlib.util.find_spec("simple_lama_inpainting") is not None
    official_lama_available = OFFICIAL_LAMA_REPO.exists() and OFFICIAL_LAMA_CHECKPOINT.exists()
    codeformer_available = CODEFORMER_REPO.exists()
    return {
        "segmentation": {
            "r011": "available" if R011_CHECKPOINT.exists() else "missing",
            "r012": "available" if R012_CHECKPOINT.exists() else "missing",
            "sensitive_mode": "available",
        },
        "inpainting": {
            "simple_lama": "available" if simple_lama_available else "unavailable",
            "official_lama": "cpu-only" if official_lama_available else "unavailable",
            "official_lama_env": OFFICIAL_LAMA_ENV,
            "opencv": "available",
        },
        "face_restoration": {
            "CodeFormer": "available" if codeformer_available else "unavailable",
        },
        "current_recommendation": {
            "demo3": "auto_r011_sensitive_low_threshold + official_lama + CodeFormer 0.7",
            "fallback": "auto_r011_union_refined + simple_lama",
        },
    }


def format_system_readiness() -> str:
    readiness = get_system_readiness()
    return (
        "Segmentation:\n"
        f"  r011: {readiness['segmentation']['r011']}\n"
        f"  r012: {readiness['segmentation']['r012']}\n"
        f"  sensitive mode: {readiness['segmentation']['sensitive_mode']}\n\n"
        "Inpainting:\n"
        f"  simple_lama: {readiness['inpainting']['simple_lama']}\n"
        f"  official_lama: {readiness['inpainting']['official_lama']}\n"
        f"  opencv: {readiness['inpainting']['opencv']}\n\n"
        "Face Restoration:\n"
        f"  CodeFormer: {readiness['face_restoration']['CodeFormer']}\n\n"
        "Current recommendation:\n"
        f"  demo3: {readiness['current_recommendation']['demo3']}\n"
        f"  fallback: {readiness['current_recommendation']['fallback']}"
    )


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    binary = (mask > 127).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), binary):
        raise RuntimeError(f"Không ghi được mask: {path}")


def _download_update(visible: bool, value: str | None = None):
    if gr is not None:
        return gr.update(visible=visible, value=value)
    return value if visible else None


def _empty_response(message: str, metadata_path: Path | None = None, warnings: list[str] | None = None) -> tuple:
    payload = {
        "success": False,
        "input_path": None,
        "mask_mode": None,
        "inpainting_backend_requested": None,
        "inpainting_backend_actual": None,
        "face_restoration_requested": False,
        "face_restoration_applied": False,
        "face_backend": "none",
        "codeformer_fidelity": None,
        "processing_time_sec": 0.0,
        "warnings": warnings or [],
        "errors": [message],
        "reason": message,
    }
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return None, None, None, _download_update(False), payload


def _load_metadata_text(metadata_path: Path, stdout: str, stderr: str) -> str:
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return json.dumps(payload, ensure_ascii=False, indent=2)
    payload = {"status": "missing_metadata", "stdout": stdout, "stderr": stderr}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _normalize_metadata(
    payload: dict,
    *,
    success: bool,
    input_path: Path,
    mode: str,
    backend: str,
    face_mode: str,
    codeformer_fidelity: float,
    processing_time_sec: float,
    warnings: list[str],
    errors: list[str],
) -> dict:
    output = dict(payload)
    output["success"] = bool(success)
    output["input_path"] = str(input_path)
    output["mask_mode"] = mode
    output["inpainting_backend_requested"] = backend
    output["inpainting_backend_actual"] = output.get("inpainting_backend_actual") or output.get("actual_backend")
    output["face_restoration_requested"] = face_mode != "off"
    output["face_restoration_applied"] = bool(output.get("face_restoration_applied", False))
    output["face_backend"] = output.get("face_backend", "none")
    output["codeformer_fidelity"] = output.get("codeformer_fidelity", codeformer_fidelity)
    output["processing_time_sec"] = float(processing_time_sec)
    output["warnings"] = warnings + [item for item in output.get("errors_or_warnings", []) if item]
    output["errors"] = errors
    if backend == "official_lama":
        output["runtime_warning"] = "official_lama_cpu_may_be_slow"
        output["official_lama_device"] = output.get("official_lama_device", "cpu")
        output["official_lama_reason"] = output.get("official_lama_reason", "unknown")
        output["fallback_applied"] = output.get("inpainting_backend_actual") != "official_lama"
    return output


def _validate_external_mask(image: np.ndarray, external_mask: np.ndarray | None) -> tuple[bool, dict, str | None]:
    if external_mask is None:
        return False, {}, "Mode external yêu cầu upload mask PNG."
    mask = np.asarray(external_mask)
    if mask.ndim == 3 and mask.shape[2] == 4:
        mask = mask[:, :, :3]
    if mask.ndim == 3:
        mask_gray = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif mask.ndim == 2:
        mask_gray = mask.astype(np.uint8)
    else:
        return False, {}, f"External mask shape không hợp lệ: {mask.shape}"
    size_match = mask_gray.shape[:2] == image.shape[:2]
    unique_values = sorted(int(value) for value in np.unique(mask_gray)[:20])
    binary = set(unique_values).issubset({0, 255})
    metadata = {
        "external_mask_used": True,
        "external_mask_binary": bool(binary),
        "external_mask_size_match": bool(size_match),
        "external_mask_unique_values_sample": unique_values,
    }
    if not size_match:
        return False, metadata, f"External mask size mismatch: image H/W={image.shape[:2]}, mask H/W={mask_gray.shape[:2]}"
    return True, metadata, None


def run_restoration(
    image: np.ndarray | None,
    mode: str,
    backend: str,
    face_mode: str,
    codeformer_fidelity: float,
    external_mask: np.ndarray | None,
    progress=gr.Progress() if gr is not None else None,
) -> tuple:
    start = time.perf_counter()
    run_id = uuid4().hex[:12]
    input_dir = OUTPUT_ROOT / "inputs"
    error_metadata_path = OUTPUT_ROOT / run_id / mode / "metadata_error.json"
    warnings: list[str] = []
    errors: list[str] = []
    if image is None:
        return _empty_response("Cần chọn ảnh đầu vào.", error_metadata_path)
    if progress is not None:
        progress(0.05, desc="loading image")
    if mode.startswith("auto_r012") and not R012_CHECKPOINT.exists():
        return _empty_response("Checkpoint r012 chưa tồn tại, mode r012 bị skip an toàn.", error_metadata_path)
    if backend == "fine_tuned_lama" and not any(path.exists() for path in FINE_TUNED_LAMA_MARKERS):
        return _empty_response("fine_tuned_lama chưa có checkpoint/config, backend bị skip an toàn.", error_metadata_path)
    external_metadata: dict = {}
    if mode.startswith("external"):
        ok_mask, external_metadata, mask_error = _validate_external_mask(image, external_mask)
        if not ok_mask:
            if gr is not None:
                gr.Warning(mask_error)
            return _empty_response(mask_error or "External mask không hợp lệ.", error_metadata_path)
    if backend == "official_lama":
        warnings.append("official_lama CPU local có thể chậm.")

    image_path = input_dir / f"{run_id}.png"
    _write_rgb(image_path, image)
    if progress is not None:
        progress(0.20, desc="generating mask")

    checkpoint = R012_CHECKPOINT if mode.startswith("auto_r012") else R011_CHECKPOINT
    command = [
        sys.executable,
        "scripts\\run_restoration_pipeline.py",
        "--image",
        str(image_path),
        "--mode",
        mode,
        "--output-dir",
        str(OUTPUT_ROOT),
        "--checkpoint",
        str(checkpoint),
        "--inpaint-backend",
        backend,
        "--face-mode",
        face_mode,
        "--codeformer-fidelity",
        f"{float(codeformer_fidelity):.3f}",
    ]
    if mode.startswith("external"):
        mask_path = input_dir / f"{run_id}_mask.png"
        _write_mask(mask_path, external_mask)
        command.extend(["--external-mask", str(mask_path)])
        external_metadata["external_mask_path"] = str(mask_path)
    if progress is not None:
        progress(0.45, desc="running inpainting")

    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if progress is not None:
        progress(0.80, desc="running CodeFormer")
    output_dir = OUTPUT_ROOT / run_id / mode
    metadata_path = output_dir / "metadata.json"
    metadata_payload = json.loads(_load_metadata_text(metadata_path, result.stdout, result.stderr))
    if result.returncode != 0:
        errors.append("pipeline_failed")
        error_payload = {
            "success": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "face_restoration_applied": False,
            "reason": "pipeline_failed",
            "errors": errors,
            "warnings": warnings,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata_payload = error_payload
        if gr is not None:
            gr.Warning("Pipeline failed; xem metadata để biết chi tiết.")
    if backend == "official_lama" and metadata_payload.get("inpainting_backend_actual") == "simple_lama":
        warnings.append("official_lama fail, đã fallback sang simple_lama.")
    metadata_payload.update(external_metadata)

    final_mask = str(output_dir / "final_mask.png") if (output_dir / "final_mask.png").exists() else None
    restored_final = str(output_dir / "restored_final.png") if (output_dir / "restored_final.png").exists() else None
    comparison = str(output_dir / "comparison_grid.png") if (output_dir / "comparison_grid.png").exists() else None
    success = result.returncode == 0 and restored_final is not None
    processing_time_sec = time.perf_counter() - start
    final_metadata = _normalize_metadata(
        metadata_payload,
        success=success,
        input_path=image_path,
        mode=mode,
        backend=backend,
        face_mode=face_mode,
        codeformer_fidelity=float(codeformer_fidelity),
        processing_time_sec=processing_time_sec,
        warnings=warnings,
        errors=errors,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(final_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress is not None:
        progress(1.0, desc="writing outputs")
    return restored_final, final_mask, comparison, _download_update(success, restored_final), final_metadata


def build_demo():
    if gr is None:
        return None
    with gr.Blocks(title="Old Photo Restoration Blueprint 2.1") as demo:
        gr.Markdown("# Old Photo Restoration Blueprint 2.1")
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Ảnh gốc", type="numpy")
                with gr.Accordion("Advanced settings", open=False):
                    external_mask = gr.Image(label="External / oracle mask PNG", type="numpy")
                readiness = gr.Textbox(label="System Readiness", value=format_system_readiness(), lines=16, interactive=False)
            with gr.Column():
                mode = gr.Dropdown(label="Mask mode", choices=MASK_MODE_CHOICES, value="auto_r011_union_refined")
                backend = gr.Dropdown(label="Inpainting backend", choices=BACKEND_CHOICES, value="simple_lama")
                face_mode = gr.Dropdown(label="Face restoration", choices=["off", "auto", "codeformer_if_available"], value="off")
                codeformer_fidelity = gr.Slider(
                    label="CodeFormer fidelity",
                    minimum=0.5,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                )
                gr.Markdown(
                    "Recommendation for demo3: experimental high-recall mask + official_lama + CodeFormer 0.7. "
                    "official_lama runs on local CPU and may be slow."
                )
        run_button = gr.Button("Restore")
        with gr.Row():
            restored_final = gr.Image(label="Restored final", type="filepath")
            final_mask = gr.Image(label="Final mask", type="filepath")
            comparison = gr.Image(label="Comparison grid", type="filepath")
        metadata = gr.JSON(label="Metadata")
        with gr.Row():
            restored_download = gr.File(label="Download restored_final", visible=False)
        run_button.click(
            run_restoration,
            inputs=[image, mode, backend, face_mode, codeformer_fidelity, external_mask],
            outputs=[
                restored_final,
                final_mask,
                comparison,
                restored_download,
                metadata,
            ],
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    if demo is None:
        print("gradio not installed")
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860, css=APP_CSS)
