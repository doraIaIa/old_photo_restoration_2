#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EXPECTED_IDS = [f"{index:03d}" for index in range(1, 61)]
VALID_MASK_VALUES = {0, 255}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dataset ảnh cũ thật và mask nứt/xước trước khi fine-tune segmentation."
    )
    parser.add_argument("--data-root", required=True, help="Thư mục chứa images/ và masks/.")
    parser.add_argument("--image-dir", default="images", help="Tên thư mục ảnh trong data-root.")
    parser.add_argument("--mask-dir", default="masks", help="Tên thư mục mask trong data-root.")
    parser.add_argument("--output-dir", required=True, help="Thư mục ghi audit.csv, summary, preview và split.")
    parser.add_argument("--make-overlays", action="store_true", help="Tạo overlay ảnh gốc + mask đỏ trong suốt.")
    parser.add_argument("--make-splits", action="store_true", help="Tạo train/val/test split từ các pair hợp lệ.")
    parser.add_argument("--seed", type=int, default=42, help="Seed dùng khi shuffle split.")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Không serialize được kiểu dữ liệu: {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_image_rgb(path: Path) -> np.ndarray | None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_mask_grayscale(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def read_mask_unchanged(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def find_image_path(images_dir: Path, sample_id: str) -> Path | None:
    candidates = [images_dir / f"{sample_id}.jpg", images_dir / f"{sample_id}.png"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def mask_color_audit(mask_raw: np.ndarray | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mask_channels": 0,
        "mask_is_rgb": False,
        "likely_red_mask": False,
        "red_pixel_count": 0,
        "red_pixel_ratio": 0.0,
    }
    if mask_raw is None:
        return result

    if mask_raw.ndim == 2:
        result["mask_channels"] = 1
        return result

    channels = int(mask_raw.shape[2])
    result["mask_channels"] = channels
    result["mask_is_rgb"] = channels >= 3

    bgr = mask_raw[:, :, :3]
    blue = bgr[:, :, 0].astype(np.int16)
    green = bgr[:, :, 1].astype(np.int16)
    red = bgr[:, :, 2].astype(np.int16)
    red_pixels = (red > 127) & (red > green + 40) & (red > blue + 40)
    red_pixel_count = int(red_pixels.sum())
    total_pixels = int(red_pixels.size)
    result["red_pixel_count"] = red_pixel_count
    result["red_pixel_ratio"] = round(red_pixel_count / total_pixels, 8) if total_pixels else 0.0
    result["likely_red_mask"] = red_pixel_count > 0
    return result


def binary_from_threshold(mask_gray: np.ndarray) -> np.ndarray:
    return np.where(mask_gray > 127, 255, 0).astype(np.uint8)


def count_connected_components(binary_mask: np.ndarray) -> int:
    positive = (binary_mask > 0).astype(np.uint8)
    if int(positive.sum()) == 0:
        return 0
    num_labels, _ = cv2.connectedComponents(positive, connectivity=8)
    return int(num_labels - 1)


def save_binary_preview(path: Path, binary_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), binary_mask)


def save_overlay(path: Path, image_rgb: np.ndarray, binary_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_for_preview = binary_mask
    if image_rgb.shape[:2] != binary_mask.shape[:2]:
        mask_for_preview = cv2.resize(
            binary_mask,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = image_rgb.astype(np.float32).copy()
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 0] = 255
    alpha = (mask_for_preview.astype(np.float32) / 255.0)[:, :, None] * 0.45
    overlay = overlay * (1.0 - alpha) + red_layer * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def list_files(directory: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not directory.exists():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return sorted(path for path in files if path.is_file())


def compute_split_counts(num_valid: int) -> tuple[int, int, int]:
    if num_valid == 60:
        return 42, 9, 9
    train_count = int(num_valid * 0.70)
    val_count = int(num_valid * 0.15)
    test_count = num_valid - train_count - val_count
    return train_count, val_count, test_count


def write_splits(splits_dir: Path, valid_ids: list[str], seed: int) -> dict[str, Any]:
    shuffled_ids = list(valid_ids)
    random.Random(seed).shuffle(shuffled_ids)

    train_count, val_count, test_count = compute_split_counts(len(shuffled_ids))
    split_map = {
        "train": shuffled_ids[:train_count],
        "val": shuffled_ids[train_count : train_count + val_count],
        "test": shuffled_ids[train_count + val_count : train_count + val_count + test_count],
    }

    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in split_map.items():
        split_path = splits_dir / f"{split_name}.txt"
        with split_path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample_id in ids:
                handle.write(f"{sample_id}\n")

    return {
        "seed": seed,
        "counts": {split_name: len(ids) for split_name, ids in split_map.items()},
        "paths": {split_name: str(splits_dir / f"{split_name}.txt") for split_name in split_map},
    }


def audit_sample(
    sample_id: str,
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    make_overlays: bool,
) -> dict[str, Any]:
    image_path = find_image_path(images_dir, sample_id)
    mask_path = masks_dir / f"{sample_id}_mask.png"
    missing_pair = image_path is None or not mask_path.exists()

    row: dict[str, Any] = {
        "id": sample_id,
        "image_path": str(image_path) if image_path else "",
        "mask_path": str(mask_path),
        "image_exists": image_path is not None,
        "mask_exists": mask_path.exists(),
        "image_width": "",
        "image_height": "",
        "mask_width": "",
        "mask_height": "",
        "size_mismatch": False,
        "unreadable": False,
        "missing_pair": missing_pair,
        "mask_channels": 0,
        "mask_is_rgb": False,
        "likely_red_mask": False,
        "red_pixel_count": 0,
        "red_pixel_ratio": 0.0,
        "mask_unique_values": "",
        "unique_value_count": 0,
        "non_binary": False,
        "empty_mask": False,
        "too_sparse": False,
        "too_dense": False,
        "likely_inverted": False,
        "positive_pixels": 0,
        "total_pixels": 0,
        "positive_ratio": 0.0,
        "connected_components": 0,
        "valid_for_split": False,
        "blocking_reasons": "",
    }

    if missing_pair:
        return row

    assert image_path is not None
    image_rgb = read_image_rgb(image_path)
    mask_gray = read_mask_grayscale(mask_path)
    mask_raw = read_mask_unchanged(mask_path)
    if image_rgb is None or mask_gray is None or mask_raw is None:
        row["unreadable"] = True
        return row

    row.update(mask_color_audit(mask_raw))
    row["image_height"], row["image_width"] = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    row["mask_height"], row["mask_width"] = int(mask_gray.shape[0]), int(mask_gray.shape[1])
    row["size_mismatch"] = image_rgb.shape[:2] != mask_gray.shape[:2]

    unique_values = np.unique(mask_gray)
    unique_values_list = [int(value) for value in unique_values.tolist()]
    row["mask_unique_values"] = " ".join(str(value) for value in unique_values_list)
    row["unique_value_count"] = len(unique_values_list)
    row["non_binary"] = not set(unique_values_list).issubset(VALID_MASK_VALUES)

    binary_mask = binary_from_threshold(mask_gray) if row["non_binary"] else mask_gray.astype(np.uint8)
    total_pixels = int(binary_mask.size)
    positive_pixels = int((binary_mask == 255).sum())
    positive_ratio = positive_pixels / total_pixels if total_pixels else 0.0

    row["total_pixels"] = total_pixels
    row["positive_pixels"] = positive_pixels
    row["positive_ratio"] = round(positive_ratio, 8)
    row["connected_components"] = count_connected_components(binary_mask)
    row["empty_mask"] = positive_pixels == 0
    row["too_sparse"] = positive_ratio < 0.001
    row["too_dense"] = positive_ratio > 0.25
    row["likely_inverted"] = positive_ratio > 0.50

    preview_path = output_dir / "binary_masks_preview" / f"{sample_id}_mask_binary.png"
    save_binary_preview(preview_path, binary_mask)

    if make_overlays:
        overlay_path = output_dir / "overlays" / f"{sample_id}_overlay.png"
        save_overlay(overlay_path, image_rgb, binary_mask)

    blocking_reasons = []
    for flag in [
        "missing_pair",
        "unreadable",
        "size_mismatch",
        "non_binary",
        "mask_is_rgb",
        "likely_red_mask",
        "empty_mask",
        "likely_inverted",
    ]:
        if bool(row[flag]):
            blocking_reasons.append(flag)

    row["blocking_reasons"] = ";".join(blocking_reasons)
    row["valid_for_split"] = not blocking_reasons
    return row


def build_summary(
    data_root: Path,
    output_dir: Path,
    image_count: int,
    mask_count: int,
    rows: list[dict[str, Any]],
    split_info: dict[str, Any] | None,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["valid_for_split"]]
    invalid_rows = [row for row in rows if not row["valid_for_split"]]
    readable_rows = [row for row in rows if not row["missing_pair"] and not row["unreadable"]]
    positive_ratios = [float(row["positive_ratio"]) for row in readable_rows]

    def ids_for_flag(flag: str) -> list[str]:
        return [str(row["id"]) for row in rows if bool(row[flag])]

    return {
        "created_at": now_iso(),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "expected_id_count": len(EXPECTED_IDS),
        "image_count": image_count,
        "mask_count": mask_count,
        "valid_pair_count": len(valid_rows),
        "invalid_or_missing_count": len(invalid_rows),
        "positive_ratio_min": min(positive_ratios) if positive_ratios else 0.0,
        "positive_ratio_mean": float(np.mean(positive_ratios)) if positive_ratios else 0.0,
        "positive_ratio_max": max(positive_ratios) if positive_ratios else 0.0,
        "flags": {
            "missing_pair": ids_for_flag("missing_pair"),
            "unreadable": ids_for_flag("unreadable"),
            "size_mismatch": ids_for_flag("size_mismatch"),
            "non_binary": ids_for_flag("non_binary"),
            "mask_is_rgb": ids_for_flag("mask_is_rgb"),
            "likely_red_mask": ids_for_flag("likely_red_mask"),
            "empty_mask": ids_for_flag("empty_mask"),
            "too_sparse": ids_for_flag("too_sparse"),
            "too_dense": ids_for_flag("too_dense"),
            "likely_inverted": ids_for_flag("likely_inverted"),
        },
        "split_info": split_info,
    }


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_dir = data_root / args.image_dir
    masks_dir = data_root / args.mask_dir

    if not data_root.exists():
        raise FileNotFoundError(f"Không tìm thấy data root: {data_root}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục {args.image_dir}/: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục {args.mask_dir}/: {masks_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_count = len(list_files(images_dir, ("*.jpg", "*.jpeg", "*.png")))
    mask_count = len(list_files(masks_dir, ("*_mask.png",)))

    rows = [
        audit_sample(
            sample_id=sample_id,
            images_dir=images_dir,
            masks_dir=masks_dir,
            output_dir=output_dir,
            make_overlays=args.make_overlays,
        )
        for sample_id in EXPECTED_IDS
    ]

    fieldnames = [
        "id",
        "image_path",
        "mask_path",
        "image_exists",
        "mask_exists",
        "image_width",
        "image_height",
        "mask_width",
        "mask_height",
        "size_mismatch",
        "unreadable",
        "missing_pair",
        "mask_channels",
        "mask_is_rgb",
        "likely_red_mask",
        "red_pixel_count",
        "red_pixel_ratio",
        "mask_unique_values",
        "unique_value_count",
        "non_binary",
        "empty_mask",
        "too_sparse",
        "too_dense",
        "likely_inverted",
        "positive_pixels",
        "total_pixels",
        "positive_ratio",
        "connected_components",
        "valid_for_split",
        "blocking_reasons",
    ]
    write_csv(output_dir / "audit.csv", rows, fieldnames)

    split_info = None
    if args.make_splits:
        valid_ids = [str(row["id"]) for row in rows if row["valid_for_split"]]
        split_info = write_splits(output_dir / "splits", valid_ids, args.seed)

    summary = build_summary(data_root, output_dir, image_count, mask_count, rows, split_info)
    write_json(output_dir / "audit_summary.json", summary)

    print(f"data_root: {data_root}")
    print(f"output_dir: {output_dir}")
    print(f"image_count: {summary['image_count']}")
    print(f"mask_count: {summary['mask_count']}")
    print(f"valid_pair_count: {summary['valid_pair_count']}")
    print(f"invalid_or_missing_count: {summary['invalid_or_missing_count']}")
    print(f"positive_ratio_min: {summary['positive_ratio_min']:.8f}")
    print(f"positive_ratio_mean: {summary['positive_ratio_mean']:.8f}")
    print(f"positive_ratio_max: {summary['positive_ratio_max']:.8f}")
    print(f"audit_csv: {output_dir / 'audit.csv'}")
    print(f"audit_summary_json: {output_dir / 'audit_summary.json'}")
    if args.make_overlays:
        print(f"overlays: {output_dir / 'overlays'}")
    if split_info:
        print(f"splits: {output_dir / 'splits'}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
