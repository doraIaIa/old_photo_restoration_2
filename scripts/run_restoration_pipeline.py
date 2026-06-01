from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.postprocess.mask_refinement import VALID_REFINE_MODES
from src.restoration.face_restoration import VALID_FACE_MODES, apply_face_restoration


MODE_CONFIGS = {
    "auto_r011": {"mask_source": "dl", "mask_refine": "none"},
    "auto_r011_union": {"mask_source": "union", "mask_refine": "none"},
    "auto_r011_refined": {"mask_source": "dl", "mask_refine": "repair_v3_conservative"},
    "auto_r011_union_refined": {"mask_source": "union", "mask_refine": "repair_v3_conservative"},
    "auto_r011_union_refined_face_auto": {"mask_source": "union", "mask_refine": "repair_v3_conservative"},
    "external": {"mask_source": "external", "mask_refine": "none"},
    "external_face_auto": {"mask_source": "external", "mask_refine": "none"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrypoint pipeline restoration cuối kỳ.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), default="auto_r011_union_refined")
    parser.add_argument("--checkpoint", default="checkpoints/segmenter/seg-unet-attn-r011-repair-ft-s42/best_iou.ckpt")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--fallback-threshold", type=float, default=0.40)
    parser.add_argument("--cv-profile", default="notebook_v7_candidate")
    parser.add_argument("--mask-refine", choices=sorted(VALID_REFINE_MODES), default=None)
    parser.add_argument("--external-mask", default="")
    parser.add_argument("--backend", default="auto", choices=["auto", "simple_lama", "opencv"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--face-mode", choices=sorted(VALID_FACE_MODES), default="auto")
    parser.add_argument("--face-strength", type=float, default=0.5)
    parser.add_argument("--skip-face-restoration", action="store_true")
    parser.add_argument("--save-all-masks", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def copy_required_outputs(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "input.png",
        "final_mask.png",
        "overlay_final.png",
        "comparison_grid.png",
        "metadata.json",
        "dl_prob_mask.png",
        "cv_mask.png",
        "union_mask.png",
        "external_mask.png",
        "final_mask_before_refine.png",
        "final_mask_refined.png",
    ]:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def run_demo(command: list[str]) -> None:
    print("run_demo:", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"run_demo.py lỗi với exit code {result.returncode}")


def main() -> int:
    args = parse_args()
    image_path = resolve_path(args.image)
    checkpoint_path = resolve_path(args.checkpoint)
    output_root = resolve_path(args.output_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy image: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    mode_config = MODE_CONFIGS[args.mode]
    mask_source = mode_config["mask_source"]
    mask_refine = args.mask_refine if args.mask_refine is not None else mode_config["mask_refine"]
    if mask_source == "external" and not args.external_mask:
        raise ValueError("Mode external yêu cầu --external-mask.")
    external_mask_path = resolve_path(args.external_mask) if args.external_mask else None
    if external_mask_path is not None and not external_mask_path.exists():
        raise FileNotFoundError(f"Không tìm thấy external mask: {external_mask_path}")

    scratch_root = output_root / "_run_demo_raw" / args.mode
    command = [
        sys.executable,
        "scripts\\run_demo.py",
        "--image",
        str(image_path),
        "--checkpoint",
        str(checkpoint_path),
        "--threshold",
        f"{args.threshold:.2f}",
        "--fallback-threshold",
        f"{args.fallback_threshold:.2f}",
        "--backend",
        args.backend,
        "--mask-source",
        mask_source,
        "--cv-profile",
        args.cv_profile,
        "--mask-dilate",
        "0",
        "--mask-refine",
        mask_refine,
        "--output-dir",
        str(scratch_root),
        "--device",
        args.device,
        "--save-prob-mask",
    ]
    if args.save_all_masks:
        command.append("--save-all-masks")
    if external_mask_path is not None:
        command.extend(["--external-mask", str(external_mask_path)])
    run_demo(command)

    run_demo_dir = scratch_root / image_path.stem / mask_source
    final_dir = output_root / image_path.stem / args.mode
    copy_required_outputs(run_demo_dir, final_dir)

    run_demo_restored = run_demo_dir / "restored_final.png"
    if not run_demo_restored.exists():
        raise FileNotFoundError(f"Không tìm thấy restored_final từ run_demo: {run_demo_restored}")
    restored_before_face = final_dir / "restored_before_face.png"
    shutil.copy2(run_demo_restored, restored_before_face)

    effective_face_mode = "off" if args.skip_face_restoration else args.face_mode
    face_restored_rgb, face_metadata = apply_face_restoration(
        read_rgb(restored_before_face),
        mode=effective_face_mode,
        strength=args.face_strength,
        output_dir=final_dir / "face_module",
    )
    restored_final = final_dir / "restored_final.png"
    write_rgb(restored_final, face_restored_rgb)

    metadata_path = final_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pipeline_mode"] = args.mode
    metadata["pipeline_output_dir"] = str(final_dir)
    metadata["face_mode"] = effective_face_mode
    metadata["face_strength"] = float(args.face_strength)
    metadata["restored_before_face_path"] = str(restored_before_face)
    metadata["restored_final_path"] = str(restored_final)
    metadata["face_module_enabled"] = face_metadata.get("face_module_enabled", False)
    metadata["face_detection_backend"] = face_metadata.get("face_detection_backend", "none")
    metadata["faces_detected"] = face_metadata.get("faces_detected", 0)
    metadata["face_restoration_applied"] = face_metadata.get("face_restoration_applied", False)
    metadata["face_restoration_backend"] = face_metadata.get("face_restoration_backend", "none")
    metadata["face_reason"] = face_metadata.get("reason", "")
    metadata["face_warning"] = face_metadata.get("warning")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"image: {image_path}")
    print(f"mode: {args.mode}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"actual_backend: {metadata.get('actual_backend', '')}")
    print(f"final_mask_ratio: {metadata.get('final_mask_ratio', '')}")
    print(f"face_mode: {effective_face_mode}")
    print(f"face_restoration_applied: {metadata.get('face_restoration_applied', False)}")
    print(f"face_reason: {metadata.get('face_reason', '')}")
    print(f"output_dir: {final_dir}")
    print(f"comparison_grid: {final_dir / 'comparison_grid.png'}")
    print(f"restored_before_face: {restored_before_face}")
    print(f"restored_final: {restored_final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
