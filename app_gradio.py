from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "blueprint21_final_assets" / "gradio_runs"
R011_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r011-repair-ft-s42" / "best_iou.ckpt"
R012_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r012-manual-repair-ft-s42" / "best_iou.ckpt"
FINE_TUNED_LAMA_MARKERS = [
    PROJECT_ROOT / "checkpoints" / "lama" / "fine_tuned_lama",
    PROJECT_ROOT / "configs" / "lama_finetuned.yaml",
]

try:
    import gradio as gr
except ImportError:
    gr = None


def available_mask_modes() -> list[str]:
    modes = ["auto_r011", "auto_r011_refined", "auto_r011_union_refined", "external"]
    if R012_CHECKPOINT.exists():
        modes.extend(["auto_r012", "auto_r012_refined", "auto_r012_union_refined"])
    return modes


def available_backends() -> list[str]:
    backends = ["simple_lama", "opencv"]
    if any(path.exists() for path in FINE_TUNED_LAMA_MARKERS):
        backends.append("fine_tuned_lama")
    return backends


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


def _empty_response(message: str, metadata_path: Path | None = None) -> tuple:
    payload = {
        "status": "error",
        "message": message,
        "face_restoration_applied": False,
        "reason": message,
    }
    metadata_text = json.dumps(payload, ensure_ascii=False, indent=2)
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(metadata_text, encoding="utf-8")
    return None, None, None, None, None, None, metadata_text, None, str(metadata_path) if metadata_path else None


def _load_metadata_text(metadata_path: Path, stdout: str, stderr: str) -> str:
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return json.dumps(payload, ensure_ascii=False, indent=2)
    payload = {"status": "missing_metadata", "stdout": stdout, "stderr": stderr}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_restoration(
    image: np.ndarray | None,
    mode: str,
    backend: str,
    face_mode: str,
    codeformer_fidelity: float,
    external_mask: np.ndarray | None,
) -> tuple:
    run_id = uuid4().hex[:12]
    input_dir = OUTPUT_ROOT / "inputs"
    error_metadata_path = OUTPUT_ROOT / run_id / mode / "metadata_error.json"
    if image is None:
        return _empty_response("Cần chọn ảnh đầu vào.", error_metadata_path)
    if mode.startswith("auto_r012") and not R012_CHECKPOINT.exists():
        return _empty_response("Checkpoint r012 chưa tồn tại, mode r012 bị skip an toàn.", error_metadata_path)
    if backend == "fine_tuned_lama" and not any(path.exists() for path in FINE_TUNED_LAMA_MARKERS):
        return _empty_response("fine_tuned_lama chưa có checkpoint/config, backend bị skip an toàn.", error_metadata_path)
    if mode.startswith("external") and external_mask is None:
        return _empty_response("Mode external yêu cầu mask đầu vào.", error_metadata_path)

    image_path = input_dir / f"{run_id}.png"
    _write_rgb(image_path, image)

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
        "--backend",
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

    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    output_dir = OUTPUT_ROOT / run_id / mode
    metadata_path = output_dir / "metadata.json"
    metadata_text = _load_metadata_text(metadata_path, result.stdout, result.stderr)
    if result.returncode != 0:
        error_payload = {
            "status": "pipeline_failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "face_restoration_applied": False,
            "reason": "pipeline_failed",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata_text = json.dumps(error_payload, ensure_ascii=False, indent=2)

    original = str(output_dir / "input.png") if (output_dir / "input.png").exists() else str(image_path)
    final_mask = str(output_dir / "final_mask.png") if (output_dir / "final_mask.png").exists() else None
    overlay = str(output_dir / "overlay_final.png") if (output_dir / "overlay_final.png").exists() else None
    before_face = str(output_dir / "restored_before_face.png") if (output_dir / "restored_before_face.png").exists() else None
    restored_final = str(output_dir / "restored_final.png") if (output_dir / "restored_final.png").exists() else None
    comparison = str(output_dir / "comparison_grid.png") if (output_dir / "comparison_grid.png").exists() else None
    metadata_download = str(metadata_path) if metadata_path.exists() else None
    return original, final_mask, overlay, before_face, restored_final, comparison, metadata_text, restored_final, metadata_download


def build_demo():
    if gr is None:
        return None
    with gr.Blocks(title="Old Photo Restoration Blueprint 2.1") as demo:
        gr.Markdown("# Old Photo Restoration Blueprint 2.1")
        with gr.Row():
            image = gr.Image(label="Ảnh gốc", type="numpy")
            external_mask = gr.Image(label="External mask tùy chọn", type="numpy")
        with gr.Row():
            mode = gr.Dropdown(label="Mask mode", choices=available_mask_modes(), value="auto_r011_union_refined")
            backend = gr.Dropdown(label="Inpainting backend", choices=available_backends(), value="simple_lama")
            face_mode = gr.Dropdown(label="Face restoration", choices=["off", "auto", "codeformer_if_available"], value="off")
            codeformer_fidelity = gr.Slider(
                label="CodeFormer fidelity",
                minimum=0.5,
                maximum=1.0,
                value=0.7,
                step=0.1,
            )
        run_button = gr.Button("Restore")
        with gr.Row():
            original = gr.Image(label="Ảnh gốc", type="filepath")
            final_mask = gr.Image(label="Final mask", type="filepath")
            overlay = gr.Image(label="Overlay mask", type="filepath")
        with gr.Row():
            before_face = gr.Image(label="Restored before face", type="filepath")
            restored_final = gr.Image(label="Restored final", type="filepath")
            comparison = gr.Image(label="Comparison grid", type="filepath")
        metadata = gr.Textbox(label="Metadata JSON", lines=18)
        with gr.Row():
            restored_download = gr.File(label="Download restored_final")
            metadata_download = gr.File(label="Download metadata")
        run_button.click(
            run_restoration,
            inputs=[image, mode, backend, face_mode, codeformer_fidelity, external_mask],
            outputs=[
                original,
                final_mask,
                overlay,
                before_face,
                restored_final,
                comparison,
                metadata,
                restored_download,
                metadata_download,
            ],
        )
    return demo


demo = build_demo()


if __name__ == "__main__":
    if demo is None:
        print("gradio not installed")
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860)
