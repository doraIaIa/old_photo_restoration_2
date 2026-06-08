#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SERIOUS_FLAGS = {"missing_pair", "size_mismatch", "non_binary", "unreadable_image", "unreadable_mask"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dataset fine-tune r013 cho bài toán image -> binary repair mask."
    )
    parser.add_argument("--dataset-root", required=True, help="Thư mục dataset chứa images/ và masks/.")
    parser.add_argument("--mask-dir-name", default="masks", help="Tên thư mục mask cần validate.")
    parser.add_argument("--seed", type=int, default=42, help="Seed để shuffle split tái lập được.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Tỷ lệ train.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỷ lệ val.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Tỷ lệ test.")
    parser.add_argument("--make-overlays", action="store_true", help="Tạo overlay preview trong overlays/.")
    parser.add_argument("--write-splits", action="store_true", help="Ghi splits/train.txt, val.txt, test.txt.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_index(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}
        return {row.get("image_id", ""): row for row in reader if row.get("image_id")}


def image_id_from_image(path: Path) -> str:
    return path.stem


def image_id_from_mask(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_mask") else stem


def list_supported_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def list_unusual_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() not in extensions
    )


def build_mask_lookup(mask_files: list[Path]) -> tuple[dict[str, Path], set[str]]:
    lookup: dict[str, Path] = {}
    duplicate_ids: set[str] = set()
    for mask_path in mask_files:
        image_id = image_id_from_mask(mask_path)
        if image_id in lookup:
            duplicate_ids.add(image_id)
            preferred_name = f"{image_id}_mask.png"
            if mask_path.name == preferred_name:
                lookup[image_id] = mask_path
        else:
            lookup[image_id] = mask_path
    return lookup, duplicate_ids


def unique_values_sample(values: np.ndarray, limit: int = 32) -> str:
    values_list = [int(value) for value in values[:limit]]
    suffix = "..." if len(values) > limit else ""
    return "|".join(str(value) for value in values_list) + suffix


def flags_to_text(flags: list[str]) -> str:
    return "|".join(sorted(set(flags)))


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def read_mask_raw(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def mask_to_gray(mask_raw: np.ndarray) -> np.ndarray:
    if mask_raw.ndim == 2:
        return mask_raw
    if mask_raw.shape[2] == 4:
        return cv2.cvtColor(mask_raw[:, :, :3], cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)


def save_overlay(path: Path, image_bgr: np.ndarray, mask_gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positive = mask_gray > 0
    overlay = image_bgr.copy().astype(np.float32)
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    alpha = positive.astype(np.float32)[:, :, None] * 0.45
    overlay = overlay * (1.0 - alpha) + red * alpha
    if not cv2.imwrite(str(path), np.clip(overlay, 0, 255).astype(np.uint8)):
        raise RuntimeError(f"Không ghi được overlay: {path}")


def ratio_level(positive_ratio: float) -> str:
    if positive_ratio == 0:
        return "empty"
    if positive_ratio < 0.0005:
        return "too_small"
    if positive_ratio > 0.35:
        return "too_large"
    return "normal"


def compute_split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if total <= 0:
        return 0, 0, 0
    if ratio_sum <= 0:
        raise ValueError("Tổng train/val/test ratio phải lớn hơn 0.")
    normalized_train = train_ratio / ratio_sum
    normalized_val = val_ratio / ratio_sum
    if total == 120 and abs(normalized_train - 0.70) < 1e-9 and abs(normalized_val - 0.15) < 1e-9:
        return 84, 18, 18
    train_count = int(round(total * normalized_train))
    val_count = int(round(total * normalized_val))
    train_count = min(train_count, total)
    val_count = min(val_count, total - train_count)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def write_splits(
    splits_dir: Path,
    valid_ids: list[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[str]]:
    shuffled_ids = list(valid_ids)
    random.Random(seed).shuffle(shuffled_ids)
    train_count, val_count, test_count = compute_split_counts(
        len(shuffled_ids), train_ratio, val_ratio, test_ratio
    )
    split_map = {
        "train": shuffled_ids[:train_count],
        "val": shuffled_ids[train_count : train_count + val_count],
        "test": shuffled_ids[train_count + val_count : train_count + val_count + test_count],
    }
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in split_map.items():
        with (splits_dir / f"{split_name}.txt").open("w", encoding="utf-8", newline="\n") as handle:
            for image_id in ids:
                handle.write(f"{image_id}\n")
    return split_map


def validate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = resolve_path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset-root: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset-root không phải thư mục: {dataset_root}")

    images_dir = dataset_root / "images"
    mask_dir_name = args.mask_dir_name
    masks_dir = dataset_root / mask_dir_name
    fixed_mode = mask_dir_name != "masks"
    splits_dir = dataset_root / ("splits_fixed" if fixed_mode else "splits")
    quality_dir = dataset_root / ("quality_checks_fixed" if fixed_mode else "quality_checks")
    overlays_dir = dataset_root / ("overlays_fixed" if fixed_mode else "overlays")

    for required_dir in (splits_dir, quality_dir, overlays_dir):
        required_dir.mkdir(parents=True, exist_ok=True)
    if not images_dir.exists():
        raise FileNotFoundError(f"Thiếu thư mục images/: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Thiếu thư mục mask {mask_dir_name}/: {masks_dir}")

    image_files = list_supported_files(images_dir, IMAGE_EXTENSIONS)
    mask_files = list_supported_files(masks_dir, MASK_EXTENSIONS)
    unusual_image_files = list_unusual_files(images_dir, IMAGE_EXTENSIONS)
    unusual_mask_files = list_unusual_files(masks_dir, MASK_EXTENSIONS)

    image_lookup: dict[str, Path] = {}
    duplicate_image_ids: set[str] = set()
    for image_path in image_files:
        image_id = image_id_from_image(image_path)
        if image_id in image_lookup:
            duplicate_image_ids.add(image_id)
        else:
            image_lookup[image_id] = image_path

    mask_lookup, duplicate_mask_ids = build_mask_lookup(mask_files)
    all_ids = sorted(set(image_lookup) | set(mask_lookup))
    existing_index = read_existing_index(dataset_root / "dataset_index.csv")

    validation_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    non_binary_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    dataset_index_rows: list[dict[str, Any]] = []
    valid_ids: list[str] = []

    split_by_id: dict[str, str] = {}
    serious_by_id: dict[str, bool] = {}
    flags_by_id: dict[str, str] = {}
    positive_by_id: dict[str, float] = {}

    for image_id in all_ids:
        image_path = image_lookup.get(image_id)
        mask_path = mask_lookup.get(image_id)
        flags: list[str] = []
        notes: list[str] = []
        width: int | str = ""
        height: int | str = ""
        mask_width: int | str = ""
        mask_height: int | str = ""
        is_binary: bool | str = ""
        unique_sample = ""
        positive_ratio: float | str = ""

        if image_id in duplicate_image_ids:
            flags.append("duplicate_image_id")
        if image_id in duplicate_mask_ids:
            flags.append("duplicate_mask_id")
        if image_path is None:
            flags.append("missing_pair")
            notes.append("Mask không có ảnh tương ứng.")
            missing_rows.append(
                {
                    "image_id": image_id,
                    "image_path": "",
                    "mask_path": str(mask_path) if mask_path else "",
                    "missing_type": "missing_image",
                    "notes": "Mask orphan, không tìm thấy ảnh cùng image_id.",
                }
            )
        if mask_path is None:
            flags.append("missing_pair")
            notes.append("Ảnh thiếu mask tương ứng.")
            missing_rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path) if image_path else "",
                    "mask_path": "",
                    "missing_type": "missing_mask",
                    "notes": "Không tìm thấy mask theo quy ước <image_id>_mask.* hoặc cùng stem.",
                }
            )

        image = read_image(image_path) if image_path else None
        mask_raw = read_mask_raw(mask_path) if mask_path else None
        mask_gray: np.ndarray | None = None

        if image_path and image is None:
            flags.append("unreadable_image")
            notes.append("Không đọc được ảnh bằng OpenCV.")
        if mask_path and mask_raw is None:
            flags.append("unreadable_mask")
            notes.append("Không đọc được mask bằng OpenCV.")

        if image is not None:
            height, width = int(image.shape[0]), int(image.shape[1])
        if mask_raw is not None:
            mask_height, mask_width = int(mask_raw.shape[0]), int(mask_raw.shape[1])
            if mask_raw.ndim == 3 and mask_raw.shape[2] == 4:
                flags.append("alpha_channel")
                notes.append("Mask có alpha channel; chỉ dùng RGB để phân tích grayscale.")
            mask_gray = mask_to_gray(mask_raw)
            unique_values = np.unique(mask_gray)
            unique_set = {int(value) for value in unique_values}
            unique_sample = unique_values_sample(unique_values)
            is_binary = unique_set <= {0, 255}
            if unique_set <= {0, 1} and unique_set != {0}:
                flags.append("binary_0_1")
                notes.append("Mask dùng giá trị 0/1 thay vì 0/255.")
            elif not is_binary:
                flags.append("non_binary")
                notes.append("Mask có giá trị xám ngoài tập {0,255}.")
                non_binary_rows.append(
                    {
                        "image_id": image_id,
                        "mask_path": str(mask_path),
                        "unique_values_sample": unique_sample,
                        "notes": "Cần kiểm tra lại mask, không tự threshold trong bước validate.",
                    }
                )
            positive_pixels = int(np.count_nonzero(mask_gray == 255))
            if unique_set <= {0, 1} and 1 in unique_set:
                positive_pixels = int(np.count_nonzero(mask_gray == 1))
            positive_ratio_value = positive_pixels / float(mask_gray.size) if mask_gray.size else 0.0
            positive_ratio = round(positive_ratio_value, 8)
            level = ratio_level(positive_ratio_value)
            if level == "empty":
                flags.append("mask_empty")
            elif level == "too_small":
                flags.append("mask_too_small")
            elif level == "too_large":
                flags.append("mask_too_large")
            positive_rows.append(
                {
                    "image_id": image_id,
                    "mask_path": str(mask_path),
                    "positive_ratio": f"{positive_ratio_value:.8f}",
                    "level": level,
                    "notes": "" if level == "normal" else f"positive_ratio thuộc mức {level}.",
                }
            )

        if image is not None and mask_raw is not None and (width, height) != (mask_width, mask_height):
            flags.append("size_mismatch")
            notes.append("Kích thước ảnh và mask không khớp.")
            mismatch_rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "image_width": width,
                    "image_height": height,
                    "mask_width": mask_width,
                    "mask_height": mask_height,
                    "notes": "Cần sửa size ở bản fixed riêng nếu muốn dùng để train.",
                }
            )

        serious = bool(set(flags) & SERIOUS_FLAGS)
        status = "fail" if serious else ("warning" if flags else "pass")
        if not serious and image_path and mask_path and image is not None and mask_gray is not None:
            valid_ids.append(image_id)
            if args.make_overlays:
                save_overlay(overlays_dir / f"{image_id}_overlay.png", image, mask_gray)

        flags_text = flags_to_text(flags)
        flags_by_id[image_id] = flags_text
        positive_by_id[image_id] = float(positive_ratio) if isinstance(positive_ratio, float) else 0.0
        serious_by_id[image_id] = serious
        validation_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path) if image_path else "",
                "mask_path": str(mask_path) if mask_path else "",
                "status": status,
                "width": width,
                "height": height,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "is_binary": is_binary,
                "unique_values_sample": unique_sample,
                "positive_ratio": positive_ratio,
                "flags": flags_text,
                "notes": " ".join(notes),
            }
        )

    if args.write_splits and valid_ids:
        split_map = write_splits(
            splits_dir,
            valid_ids,
            args.seed,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        for split_name, ids in split_map.items():
            for image_id in ids:
                split_by_id[image_id] = split_name

    for image_id in sorted(image_lookup):
        previous = existing_index.get(image_id, {})
        image_path = image_lookup[image_id]
        mask_path = mask_lookup.get(image_id)
        dataset_index_rows.append(
            {
                "image_id": image_id,
                "filename": image_path.name,
                "mask_filename": mask_path.name if mask_path else "",
                "split": split_by_id.get(image_id, previous.get("split", "")),
                "has_face": previous.get("has_face", "unknown") or "unknown",
                "damage_level": previous.get("damage_level", "unknown") or "unknown",
                "damage_type": previous.get("damage_type", "unknown") or "unknown",
                "hard_negative_type": previous.get("hard_negative_type", "unknown") or "unknown",
                "positive_ratio": f"{positive_by_id.get(image_id, 0.0):.8f}",
                "flags": flags_by_id.get(image_id, ""),
                "notes": previous.get("notes", ""),
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "mask_dir_name": mask_dir_name,
        "total_images": len(image_files),
        "total_masks": len(mask_files),
        "total_pairs": sum(1 for image_id in all_ids if image_id in image_lookup and image_id in mask_lookup),
        "valid_pairs": len(valid_ids),
        "missing_images": sum(1 for row in missing_rows if row["missing_type"] == "missing_image"),
        "missing_masks": sum(1 for row in missing_rows if row["missing_type"] == "missing_mask"),
        "size_mismatch_count": sum(1 for row in validation_rows if "size_mismatch" in row["flags"].split("|")),
        "non_binary_count": sum(1 for row in validation_rows if "non_binary" in row["flags"].split("|")),
        "empty_mask_count": sum(1 for row in validation_rows if "mask_empty" in row["flags"].split("|")),
        "too_small_count": sum(1 for row in validation_rows if "mask_too_small" in row["flags"].split("|")),
        "too_large_count": sum(1 for row in validation_rows if "mask_too_large" in row["flags"].split("|")),
        "duplicate_image_id_count": len(duplicate_image_ids),
        "duplicate_mask_id_count": len(duplicate_mask_ids),
        "unusual_image_files": [path.name for path in unusual_image_files],
        "unusual_mask_files": [path.name for path in unusual_mask_files],
        "split_counts": {
            "train": sum(1 for value in split_by_id.values() if value == "train"),
            "val": sum(1 for value in split_by_id.values() if value == "val"),
            "test": sum(1 for value in split_by_id.values() if value == "test"),
        },
        "recommended_next_action": "",
    }
    if summary["valid_pairs"] == summary["total_pairs"] and not any(
        summary[key]
        for key in (
            "missing_images",
            "missing_masks",
            "size_mismatch_count",
            "non_binary_count",
            "empty_mask_count",
        )
    ):
        summary["recommended_next_action"] = "Dataset sẵn sàng cho bước fine-tune segmentation r013."
    else:
        summary["recommended_next_action"] = (
            "Cần xem các report trong quality_checks/ và sửa lỗi nghiêm trọng trước khi fine-tune."
        )

    write_csv(
        quality_dir / "mask_validation_report.csv",
        validation_rows,
        [
            "image_id",
            "image_path",
            "mask_path",
            "status",
            "width",
            "height",
            "mask_width",
            "mask_height",
            "is_binary",
            "unique_values_sample",
            "positive_ratio",
            "flags",
            "notes",
        ],
    )
    write_csv(
        quality_dir / "missing_pairs_report.csv",
        missing_rows,
        ["image_id", "image_path", "mask_path", "missing_type", "notes"],
    )
    write_csv(
        quality_dir / "size_mismatch_report.csv",
        mismatch_rows,
        ["image_id", "image_path", "mask_path", "image_width", "image_height", "mask_width", "mask_height", "notes"],
    )
    write_csv(
        quality_dir / "non_binary_mask_report.csv",
        non_binary_rows,
        ["image_id", "mask_path", "unique_values_sample", "notes"],
    )
    write_csv(
        quality_dir / "positive_ratio_report.csv",
        positive_rows,
        ["image_id", "mask_path", "positive_ratio", "level", "notes"],
    )
    summary_name = "validation_dataset_summary.json" if fixed_mode else "dataset_summary.json"
    with (quality_dir / summary_name).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    index_name = "dataset_index_validation.csv" if fixed_mode else "dataset_index.csv"
    write_csv(
        dataset_root / index_name,
        dataset_index_rows,
        [
            "image_id",
            "filename",
            "mask_filename",
            "split",
            "has_face",
            "damage_level",
            "damage_type",
            "hard_negative_type",
            "positive_ratio",
            "flags",
            "notes",
        ],
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = validate_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
