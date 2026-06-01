from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.postprocess.mask_refinement import compute_mask_metrics


REFINE_MODES = ["none", "dilate1", "dilate2", "close_dilate1", "repair_v1", "repair_v2", "repair_v3_conservative"]
EXPECTED_DEMO_FILES = ["demo1.jpg", "demo2.png", "demo3.png"]


@dataclass(frozen=True)
class BaseVariant:
    name: str
    mask_source: str
    threshold: float
    fallback_threshold: float


BASE_VARIANTS = [
    BaseVariant("r010_dl", "dl", 0.70, 0.50),
    BaseVariant("r010_union_cv", "union", 0.70, 0.50),
    BaseVariant("r010_union_cv_t050", "union", 0.50, 0.40),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy demo refinement variants cho Module 1.5.")
    parser.add_argument("--demo-dir", required=True)
    parser.add_argument("--manual-mask-dir", required=True)
    parser.add_argument("--r010-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_command(command: list[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Lệnh `{label}` lỗi với exit code {result.returncode}.")


def load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_gray(path: Path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    return mask


def find_demo_images(demo_dir: Path) -> tuple[list[Path], list[Path]]:
    existing: list[Path] = []
    skipped: list[Path] = []
    for filename in EXPECTED_DEMO_FILES:
        candidate = demo_dir / filename
        if candidate.exists():
            existing.append(candidate)
        else:
            skipped.append(candidate)
    return existing, skipped


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "demo_id",
        "image_path",
        "base_name",
        "mask_source",
        "threshold",
        "fallback_threshold",
        "cv_profile",
        "mask_refine",
        "final_mask_ratio",
        "iou_vs_manual",
        "f1_vs_manual",
        "precision_vs_manual",
        "recall_vs_manual",
        "missing_ratio_vs_manual",
        "extra_ratio_vs_manual",
        "actual_backend",
        "comparison_grid",
        "restored_final",
        "final_mask",
        "metadata_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    demo_dir = resolve_path(args.demo_dir)
    manual_mask_dir = resolve_path(args.manual_mask_dir)
    checkpoint = resolve_path(args.r010_checkpoint)
    output_root = resolve_path(args.output_root)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Không tìm thấy r010 checkpoint: {checkpoint}")
    output_root.mkdir(parents=True, exist_ok=True)

    demo_images, skipped_images = find_demo_images(demo_dir)
    for skipped in skipped_images:
        print(f"skip demo image thiếu: {skipped}")

    rows: list[dict[str, Any]] = []
    for image_path in demo_images:
        manual_mask_path = manual_mask_dir / f"{image_path.stem}_mask.png"
        manual_mask = load_gray(manual_mask_path) if manual_mask_path.exists() else None

        for base in BASE_VARIANTS:
            for refine_mode in REFINE_MODES:
                variant_name = f"{base.name}__{refine_mode}"
                variant_root = output_root / variant_name
                command = [
                    sys.executable,
                    "scripts\\run_demo.py",
                    "--image",
                    str(image_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--threshold",
                    f"{base.threshold:.2f}",
                    "--fallback-threshold",
                    f"{base.fallback_threshold:.2f}",
                    "--backend",
                    "auto",
                    "--mask-source",
                    base.mask_source,
                    "--cv-profile",
                    "notebook_v7_candidate",
                    "--mask-dilate",
                    "0",
                    "--mask-refine",
                    refine_mode,
                    "--output-dir",
                    str(variant_root),
                    "--device",
                    args.device,
                    "--save-prob-mask",
                    "--save-all-masks",
                ]
                run_command(command, f"refinement {variant_name} {image_path.name}")
                sample_dir = variant_root / image_path.stem / base.mask_source
                metadata = load_json_or_empty(sample_dir / "metadata.json")
                final_mask_path = sample_dir / "final_mask.png"
                metric_values = {}
                if manual_mask is not None and final_mask_path.exists():
                    metric_values = compute_mask_metrics(load_gray(final_mask_path), manual_mask)

                rows.append(
                    {
                        "demo_id": image_path.stem,
                        "image_path": str(image_path),
                        "base_name": base.name,
                        "mask_source": base.mask_source,
                        "threshold": f"{base.threshold:.2f}",
                        "fallback_threshold": f"{base.fallback_threshold:.2f}",
                        "cv_profile": metadata.get("cv_profile", "notebook_v7_candidate"),
                        "mask_refine": refine_mode,
                        "final_mask_ratio": format_optional(metadata.get("final_mask_ratio")),
                        "iou_vs_manual": format_optional(metric_values.get("iou")),
                        "f1_vs_manual": format_optional(metric_values.get("f1")),
                        "precision_vs_manual": format_optional(metric_values.get("precision")),
                        "recall_vs_manual": format_optional(metric_values.get("recall")),
                        "missing_ratio_vs_manual": format_optional(metric_values.get("missing_ratio")),
                        "extra_ratio_vs_manual": format_optional(metric_values.get("extra_ratio")),
                        "actual_backend": metadata.get("actual_backend", ""),
                        "comparison_grid": str(sample_dir / "comparison_grid.png") if (sample_dir / "comparison_grid.png").exists() else "",
                        "restored_final": str(sample_dir / "restored_final.png") if (sample_dir / "restored_final.png").exists() else "",
                        "final_mask": str(final_mask_path) if final_mask_path.exists() else "",
                        "metadata_json": str(sample_dir / "metadata.json") if (sample_dir / "metadata.json").exists() else "",
                    }
                )

    index_path = output_root / "demo_refinement_index.csv"
    write_csv(index_path, rows)
    print(f"demo_refinement_index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
