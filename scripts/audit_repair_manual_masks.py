from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
CANDIDATE_IDS = ["057", "007", "012", "055", "059", "056", "009", "027", "047", "005", "017", "060", "053", "029", "010"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manual repair masks cho workflow r012.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks_repair_manual")
    parser.add_argument("--output-dir", default="outputs/final_pipeline_assets/repair_manual_audit")
    parser.add_argument("--fix", action="store_true", help="Threshold non-binary mask về 0/255 và ghi lại tại chỗ.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def find_image(images_dir: Path, sample_id: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{sample_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def mask_id(path: Path) -> str:
    name = path.stem
    return name[:-5] if name.endswith("_mask") else name


def audit_mask(mask_path: Path, images_dir: Path, fix: bool) -> dict[str, Any]:
    sample_id = mask_id(mask_path)
    image_path = find_image(images_dir, sample_id)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    row: dict[str, Any] = {
        "id": sample_id,
        "mask_path": str(mask_path),
        "image_path": str(image_path) if image_path else "",
        "missing": image_path is None,
        "unreadable": mask is None,
        "size_mismatch": False,
        "non_binary": False,
        "fixed": False,
        "positive_ratio": None,
        "too_sparse": False,
        "too_dense": False,
        "likely_empty": False,
        "valid": False,
    }
    if mask is None:
        return row

    if image_path is not None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            row["missing"] = True
        else:
            image_height, image_width = image.shape[:2]
            mask_height, mask_width = mask.shape[:2]
            row["image_size"] = {"width": int(image_width), "height": int(image_height)}
            row["mask_size"] = {"width": int(mask_width), "height": int(mask_height)}
            row["size_mismatch"] = (image_width, image_height) != (mask_width, mask_height)

    unique_values = sorted(int(value) for value in np.unique(mask))
    row["unique_values"] = unique_values[:32]
    row["unique_value_count"] = len(unique_values)
    binary = set(unique_values).issubset({0, 255})
    row["non_binary"] = not binary
    if not binary:
        mask = (mask > 127).astype(np.uint8) * 255
        if fix:
            if not cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"Không ghi được mask đã fix: {mask_path}")
            row["fixed"] = True

    positive_ratio = float(np.count_nonzero(mask > 127) / mask.size)
    row["positive_ratio"] = positive_ratio
    row["too_sparse"] = 0.0 < positive_ratio < 0.001
    row["too_dense"] = positive_ratio > 0.25
    row["likely_empty"] = positive_ratio == 0.0
    row["valid"] = (
        image_path is not None
        and not row["unreadable"]
        and not row["size_mismatch"]
        and not row["non_binary"]
        and not row["likely_empty"]
    )
    if fix and row["fixed"]:
        row["valid"] = image_path is not None and not row["unreadable"] and not row["size_mismatch"] and not row["likely_empty"]
    return row


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Repair Manual Mask Audit",
        "",
        "## Summary",
        "",
        f"- Data root: `{summary['data_root']}`",
        f"- Manual mask dir: `{summary['mask_dir']}`",
        f"- Manual masks found: {summary['manual_masks_found']}",
        f"- Valid manual masks: {summary['valid_manual_masks']}",
        f"- Missing candidate masks: {len(summary['missing_candidate_ids'])}",
        f"- Positive ratio min/mean/max: {summary['positive_ratio_min']} / {summary['positive_ratio_mean']} / {summary['positive_ratio_max']}",
        "",
        "## Flags",
        "",
        f"- non_binary: `{summary['non_binary_ids']}`",
        f"- too_sparse: `{summary['too_sparse_ids']}`",
        f"- too_dense: `{summary['too_dense_ids']}`",
        f"- likely_empty: `{summary['likely_empty_ids']}`",
        f"- size_mismatch: `{summary['size_mismatch_ids']}`",
        f"- missing image pair: `{summary['missing_pair_ids']}`",
        f"- missing candidate ids: `{summary['missing_candidate_ids']}`",
        "",
        "## Rows",
        "",
        "| id | valid | positive_ratio | non_binary | size_mismatch | too_sparse | too_dense | likely_empty | missing |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: item["id"]):
        lines.append(
            f"| {row['id']} | {row['valid']} | {row['positive_ratio']} | {row['non_binary']} | "
            f"{row['size_mismatch']} | {row['too_sparse']} | {row['too_dense']} | {row['likely_empty']} | {row['missing']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    images_dir = data_root / args.image_dir
    masks_dir = data_root / args.mask_dir
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Không tìm thấy data-root: {data_root}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy image-dir: {images_dir}")
    masks_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(masks_dir.glob("*_mask.png"))
    rows = [audit_mask(path, images_dir, args.fix) for path in mask_paths]
    positive_ratios = [float(row["positive_ratio"]) for row in rows if row.get("positive_ratio") is not None]
    present_ids = {row["id"] for row in rows}
    summary = {
        "data_root": str(data_root),
        "image_dir": str(images_dir),
        "mask_dir": str(masks_dir),
        "manual_masks_found": len(mask_paths),
        "valid_manual_masks": sum(1 for row in rows if row["valid"]),
        "candidate_ids": CANDIDATE_IDS,
        "missing_candidate_ids": [sample_id for sample_id in CANDIDATE_IDS if sample_id not in present_ids],
        "non_binary_ids": [row["id"] for row in rows if row["non_binary"]],
        "too_sparse_ids": [row["id"] for row in rows if row["too_sparse"]],
        "too_dense_ids": [row["id"] for row in rows if row["too_dense"]],
        "likely_empty_ids": [row["id"] for row in rows if row["likely_empty"]],
        "size_mismatch_ids": [row["id"] for row in rows if row["size_mismatch"]],
        "missing_pair_ids": [row["id"] for row in rows if row["missing"]],
        "unreadable_ids": [row["id"] for row in rows if row["unreadable"]],
        "positive_ratio_min": min(positive_ratios) if positive_ratios else None,
        "positive_ratio_mean": mean(positive_ratios) if positive_ratios else None,
        "positive_ratio_max": max(positive_ratios) if positive_ratios else None,
        "fix_applied": bool(args.fix),
        "rows": rows,
    }

    json_path = output_dir / "repair_manual_audit.json"
    md_path = output_dir / "repair_manual_audit.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, summary, rows)

    print(f"manual_masks_found: {summary['manual_masks_found']}")
    print(f"valid_manual_masks: {summary['valid_manual_masks']}")
    print(f"missing_candidate_ids: {summary['missing_candidate_ids']}")
    print(f"positive_ratio_min: {summary['positive_ratio_min']}")
    print(f"positive_ratio_mean: {summary['positive_ratio_mean']}")
    print(f"positive_ratio_max: {summary['positive_ratio_max']}")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
