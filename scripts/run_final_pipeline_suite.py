from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


AUTO_MODES = [
    ("auto_r011", "off"),
    ("auto_r011_union", "off"),
    ("auto_r011_refined", "off"),
    ("auto_r011_union_refined", "off"),
    ("auto_r011_union_refined_face_auto", "auto"),
]
EXTERNAL_MODES = [
    ("external", "off"),
    ("external_face_auto", "auto"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy final restoration pipeline trên batch demo.")
    parser.add_argument("--demo-dir", required=True)
    parser.add_argument("--external-mask-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/segmenter/seg-unet-attn-r011-repair-ft-s42/best_iou.ckpt")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_command(command: list[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Lệnh `{label}` lỗi với exit code {result.returncode}.")


def find_demo_images(demo_dir: Path) -> list[Path]:
    return sorted(path for path in demo_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.is_file())


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "demo_id",
        "image_path",
        "mode",
        "checkpoint",
        "threshold",
        "fallback_threshold",
        "mask_source",
        "cv_profile",
        "mask_refine",
        "final_mask_ratio",
        "inpainting_backend_requested",
        "inpainting_backend_actual",
        "face_mode",
        "face_module_enabled",
        "faces_detected",
        "face_restoration_applied",
        "face_backend",
        "face_reason",
        "errors_or_warnings",
        "output_dir",
        "comparison_grid",
        "restored_before_face",
        "restored_final",
        "final_mask",
        "overlay_final",
        "metadata_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mode_row(image_path: Path, mode: str, checkpoint: Path, output_root: Path) -> dict[str, Any]:
    output_dir = output_root / image_path.stem / mode
    metadata_json = output_dir / "metadata.json"
    metadata = load_metadata(metadata_json)
    return {
        "demo_id": image_path.stem,
        "image_path": str(image_path),
        "mode": mode,
        "checkpoint": str(checkpoint),
        "threshold": metadata.get("threshold", ""),
        "fallback_threshold": metadata.get("fallback_threshold", ""),
        "mask_source": metadata.get("mask_source", ""),
        "cv_profile": metadata.get("cv_profile", ""),
        "mask_refine": metadata.get("mask_refine", ""),
        "final_mask_ratio": metadata.get("final_mask_ratio", ""),
        "inpainting_backend_requested": metadata.get("inpainting_backend_requested", metadata.get("backend_requested", "")),
        "inpainting_backend_actual": metadata.get("inpainting_backend_actual", metadata.get("actual_backend", "")),
        "face_mode": metadata.get("face_mode", ""),
        "face_module_enabled": metadata.get("face_module_enabled", ""),
        "faces_detected": metadata.get("faces_detected", ""),
        "face_restoration_applied": metadata.get("face_restoration_applied", ""),
        "face_backend": metadata.get("face_backend", metadata.get("face_restoration_backend", "")),
        "face_reason": metadata.get("face_reason", ""),
        "errors_or_warnings": "; ".join(str(item) for item in metadata.get("errors_or_warnings", [])),
        "output_dir": str(output_dir),
        "comparison_grid": str(output_dir / "comparison_grid.png") if (output_dir / "comparison_grid.png").exists() else "",
        "restored_before_face": str(output_dir / "restored_before_face.png") if (output_dir / "restored_before_face.png").exists() else "",
        "restored_final": str(output_dir / "restored_final.png") if (output_dir / "restored_final.png").exists() else "",
        "final_mask": str(output_dir / "final_mask.png") if (output_dir / "final_mask.png").exists() else "",
        "overlay_final": str(output_dir / "overlay_final.png") if (output_dir / "overlay_final.png").exists() else "",
        "metadata_json": str(metadata_json) if metadata_json.exists() else "",
    }


def write_summary(path: Path, rows: list[dict[str, Any]], checkpoint: Path) -> None:
    lines = [
        "# Final Pipeline Demo Suite",
        "",
        "## Mục tiêu",
        "",
        "Chạy pipeline restoration hiện tại trên bộ demo bằng r011 repair-mask checkpoint, gồm automatic, external-mask và face-auto variants.",
        "",
        f"- Default checkpoint r011: `{checkpoint}`",
        "",
        "## Modes",
        "",
        "- `auto_r011`: baseline automatic, DL repair mask.",
        "- `auto_r011_union`: DL + CV notebook_v7_candidate.",
        "- `auto_r011_refined`: DL + Module 1.5 refinement.",
        "- `auto_r011_union_refined`: DL + CV + Module 1.5.",
        "- `auto_r011_union_refined_face_auto`: automatic enhanced + Module 3 face auto nếu dependency/adapter sẵn sàng.",
        "- `external`: manual/external mask upper bound, không phải automatic.",
        "- `external_face_auto`: external mask + Module 3 face auto nếu dependency/adapter sẵn sàng.",
        "",
        "## Demo outputs",
        "",
        "| demo_id | mode | final_mask_ratio | inpaint_backend | face_mode | face_reason | comparison_grid |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['demo_id']} | {row['mode']} | {row['final_mask_ratio']} | {row['inpainting_backend_actual']} | "
            f"{row['face_mode']} | {row['face_reason']} | `{row['comparison_grid']}` |"
        )
    lines.extend(
        [
            "",
            "## Khuyến nghị",
            "",
            "- `auto_r011` là baseline automatic hiện tại.",
            "- `auto_r011_union` là optional nếu CV giúp bắt thêm crack trong visual review.",
            "- `auto_r011_union_refined` là enhanced automatic nhưng phải kiểm tra false positive.",
            "- Các mode `_face_auto` chỉ thay đổi Module 3; nếu dependency/adapter chưa có, ảnh cuối giữ nguyên và metadata ghi rõ lý do.",
            "- `external` và `external_face_auto` chỉ là upper bound/diagnosis, không phải automatic.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    demo_dir = resolve_path(args.demo_dir)
    external_mask_dir = resolve_path(args.external_mask_dir)
    output_root = resolve_path(args.output_root)
    checkpoint = resolve_path(args.checkpoint)
    if not demo_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy demo-dir: {demo_dir}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint}")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for image_path in find_demo_images(demo_dir):
        run_modes = list(AUTO_MODES)
        external_mask = external_mask_dir / f"{image_path.stem}_mask.png"
        if external_mask.exists():
            run_modes.extend(EXTERNAL_MODES)
        else:
            print(f"skip external cho {image_path.name}: thiếu {external_mask}")

        for mode, face_mode in run_modes:
            command = [
                sys.executable,
                "scripts\\run_restoration_pipeline.py",
                "--image",
                str(image_path),
                "--mode",
                mode,
                "--output-dir",
                str(output_root),
                "--checkpoint",
                str(checkpoint),
                "--device",
                args.device,
                "--face-mode",
                face_mode,
            ]
            if mode.startswith("external"):
                command.extend(["--external-mask", str(external_mask)])
            run_command(command, f"{image_path.name} {mode}")
            rows.append(mode_row(image_path, mode, checkpoint, output_root))

    write_csv(output_root / "demo_index.csv", rows)
    write_summary(output_root / "SUMMARY.md", rows, checkpoint)
    print(f"demo_index: {output_root / 'demo_index.csv'}")
    print(f"summary: {output_root / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
