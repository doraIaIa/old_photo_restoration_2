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
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r011-repair-ft-s42" / "best_iou.ckpt"

try:
    import gradio as gr
except ImportError:
    gr = None


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


def run_restoration(
    image: np.ndarray | None,
    mode: str,
    face_mode: str,
    external_mask: np.ndarray | None,
) -> tuple[str | None, str | None, str | None, str]:
    if image is None:
        return None, None, None, "Cần chọn ảnh đầu vào."

    run_id = uuid4().hex[:12]
    input_dir = OUTPUT_ROOT / "inputs"
    image_path = input_dir / f"{run_id}.png"
    _write_rgb(image_path, image)

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
        str(DEFAULT_CHECKPOINT),
        "--face-mode",
        face_mode,
    ]
    if mode.startswith("external"):
        if external_mask is None:
            return None, None, None, "Mode external yêu cầu mask đầu vào."
        mask_path = input_dir / f"{run_id}_mask.png"
        _write_mask(mask_path, external_mask)
        command.extend(["--external-mask", str(mask_path)])

    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    output_dir = OUTPUT_ROOT / run_id / mode
    metadata_path = output_dir / "metadata.json"
    metadata_text = result.stdout
    if result.stderr:
        metadata_text += "\n\nSTDERR:\n" + result.stderr
    if metadata_path.exists():
        metadata_text += "\n\nmetadata.json:\n" + json.dumps(json.loads(metadata_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    if result.returncode != 0:
        return None, None, None, metadata_text

    return (
        str(output_dir / "restored_final.png") if (output_dir / "restored_final.png").exists() else None,
        str(output_dir / "final_mask.png") if (output_dir / "final_mask.png").exists() else None,
        str(output_dir / "overlay_final.png") if (output_dir / "overlay_final.png").exists() else None,
        metadata_text,
    )


def build_demo():
    if gr is None:
        return None
    with gr.Blocks(title="Old Photo Restoration Blueprint 2.1") as demo:
        gr.Markdown("# Old Photo Restoration Blueprint 2.1")
        with gr.Row():
            image = gr.Image(label="Ảnh cũ đầu vào", type="numpy")
            external_mask = gr.Image(label="External mask tùy chọn", type="numpy")
        with gr.Row():
            mode = gr.Dropdown(
                label="Pipeline mode",
                choices=[
                    "auto_r011",
                    "auto_r011_union",
                    "auto_r011_refined",
                    "auto_r011_union_refined",
                    "auto_r011_union_refined_face_auto",
                    "external",
                    "external_face_auto",
                ],
                value="auto_r011_union_refined",
            )
            face_mode = gr.Dropdown(label="Face mode", choices=["off", "auto", "codeformer_if_available"], value="off")
        run_button = gr.Button("Run")
        with gr.Row():
            restored = gr.Image(label="Restored final", type="filepath")
            final_mask = gr.Image(label="Final mask", type="filepath")
            overlay = gr.Image(label="Overlay", type="filepath")
        metadata = gr.Textbox(label="Metadata/log", lines=18)
        run_button.click(run_restoration, inputs=[image, mode, face_mode, external_mask], outputs=[restored, final_mask, overlay, metadata])
    return demo


demo = build_demo()


if __name__ == "__main__":
    if demo is None:
        print("gradio not installed")
    else:
        demo.launch(server_name="127.0.0.1", server_port=7860)
