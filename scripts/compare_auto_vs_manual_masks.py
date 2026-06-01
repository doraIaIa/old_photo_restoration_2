from __future__ import annotations

import argparse
import csv
import json
import sys
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

from src.postprocess.mask_refinement import compute_mask_metrics, save_mask_diff_visuals


AUTO_VARIANTS = [
    ("r009_dl_t090", "dl"),
    ("r010_dl_t070", "dl"),
    ("r010_union_cv_t070_dilate0", "union"),
    ("r010_union_cv_t070_dilate1", "union"),
    ("r010_union_cv_t070_dilate2", "union"),
    ("r010_union_cv_t050_dilate1", "union"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So sánh auto final masks với manual repair mask.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--manual-mask", required=True)
    parser.add_argument("--auto-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_rgb(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_gray(path: Path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    return mask


def load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "final_mask",
        "iou",
        "f1",
        "precision",
        "recall",
        "pred_ratio",
        "gt_ratio",
        "intersection_ratio",
        "missing_ratio",
        "extra_ratio",
        "metadata_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    image_path = resolve_path(args.image)
    manual_mask_path = resolve_path(args.manual_mask)
    auto_root = resolve_path(args.auto_root)
    output_dir = resolve_path(args.output_dir)
    visuals_dir = output_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgb(image_path)
    manual_mask = load_gray(manual_mask_path)
    demo_id = image_path.stem
    rows: list[dict[str, Any]] = []

    for variant, mask_source in AUTO_VARIANTS:
        sample_dir = auto_root / variant / demo_id / mask_source
        final_mask_path = sample_dir / "final_mask.png"
        metadata_path = sample_dir / "metadata.json"
        if not final_mask_path.exists():
            print(f"skip {variant}: thiếu {final_mask_path}")
            continue

        pred_mask = load_gray(final_mask_path)
        metrics = compute_mask_metrics(pred_mask, manual_mask)
        metadata = load_json_or_empty(metadata_path)
        row = {
            "variant": variant,
            "final_mask": str(final_mask_path),
            "metadata_json": str(metadata_path) if metadata_path.exists() else "",
            **{key: f"{value:.8f}" for key, value in metrics.items()},
        }
        rows.append(row)
        save_mask_diff_visuals(image, pred_mask, manual_mask, visuals_dir, variant)
        print(
            f"{variant}: iou={metrics['iou']:.6f} f1={metrics['f1']:.6f} "
            f"precision={metrics['precision']:.6f} recall={metrics['recall']:.6f} "
            f"missing={metrics['missing_ratio']:.6f} extra={metrics['extra_ratio']:.6f} "
            f"final_ratio={metadata.get('final_mask_ratio', '')}"
        )

    best_by_iou = max(rows, key=lambda item: float(item["iou"])) if rows else None
    write_csv(output_dir / "auto_vs_manual_metrics.csv", rows)
    summary = {
        "image": str(image_path),
        "manual_mask": str(manual_mask_path),
        "auto_root": str(auto_root),
        "num_compared": len(rows),
        "best_by_iou": best_by_iou,
        "metrics_csv": str(output_dir / "auto_vs_manual_metrics.csv"),
        "visuals_dir": str(visuals_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics_csv: {output_dir / 'auto_vs_manual_metrics.csv'}")
    print(f"summary_json: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
