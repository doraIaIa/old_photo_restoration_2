from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.postprocess.mask_refinement import VALID_REFINE_MODES


MODE_CONFIGS = {
    "auto_r011": {"mask_source": "dl", "mask_refine": "none"},
    "auto_r011_union": {"mask_source": "union", "mask_refine": "none"},
    "auto_r011_refined": {"mask_source": "dl", "mask_refine": "repair_v3_conservative"},
    "auto_r011_union_refined": {"mask_source": "union", "mask_refine": "repair_v3_conservative"},
    "external": {"mask_source": "external", "mask_refine": "none"},
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
        "restored_final.png",
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
    if args.mode == "external" and not args.external_mask:
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

    print("run_demo:", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"run_demo.py lỗi với exit code {result.returncode}")

    run_demo_dir = scratch_root / image_path.stem / mask_source
    final_dir = output_root / image_path.stem / args.mode
    copy_required_outputs(run_demo_dir, final_dir)
    metadata_path = final_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["pipeline_mode"] = args.mode
        metadata["pipeline_output_dir"] = str(final_dir)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"image: {image_path}")
    print(f"mode: {args.mode}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"actual_backend: {metadata.get('actual_backend', '')}")
    print(f"final_mask_ratio: {metadata.get('final_mask_ratio', '')}")
    print(f"output_dir: {final_dir}")
    print(f"comparison_grid: {final_dir / 'comparison_grid.png'}")
    print(f"restored_final: {final_dir / 'restored_final.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
