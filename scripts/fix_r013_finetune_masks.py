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
ASPECT_RATIO_TOLERANCE = 0.005

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo masks_fixed/ sạch cho dataset fine-tune r013, không đè dữ liệu gốc."
    )
    parser.add_argument("--dataset-root", required=True, help="Thư mục dataset chứa images/ và masks/.")
    parser.add_argument("--threshold", type=int, default=127, help="Pixel >= threshold sẽ thành 255, ngược lại 0.")
    parser.add_argument("--resize-safe", action="store_true", help="Cho phép resize mask bằng nearest nếu aspect ratio lệch <= 0.5%%.")
    parser.add_argument("--make-overlays", action="store_true", help="Tạo overlay review trong overlays_fixed/.")
    parser.add_argument("--write-splits", action="store_true", help="Ghi train/val/test vào splits_fixed/.")
    parser.add_argument("--seed", type=int, default=42, help="Seed để shuffle split tái lập được.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Tỷ lệ train.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỷ lệ val.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Tỷ lệ test.")
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


def list_supported_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def image_id_from_image(path: Path) -> str:
    return path.stem


def image_id_from_mask(path: Path) -> str:
    stem = path.stem
    return stem[:-5] if stem.endswith("_mask") else stem


def build_lookup(files: list[Path], id_func: Any) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in files:
        image_id = id_func(path)
        if image_id not in lookup or path.name == f"{image_id}_mask.png":
            lookup[image_id] = path
    return lookup


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def read_mask(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


def mask_to_gray(mask_raw: np.ndarray) -> np.ndarray:
    if mask_raw.ndim == 2:
        return mask_raw
    if mask_raw.shape[2] == 4:
        return cv2.cvtColor(mask_raw[:, :, :3], cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)


def unique_values_sample(mask_gray: np.ndarray, limit: int = 32) -> str:
    values = np.unique(mask_gray)
    values_list = [int(value) for value in values[:limit]]
    suffix = "..." if len(values) > limit else ""
    return "|".join(str(value) for value in values_list) + suffix


def is_binary(mask_gray: np.ndarray) -> bool:
    return {int(value) for value in np.unique(mask_gray)} <= {0, 255}


def binarize(mask_gray: np.ndarray, threshold: int) -> np.ndarray:
    return np.where(mask_gray >= threshold, 255, 0).astype(np.uint8)


def aspect_ratio(width: int, height: int) -> float:
    return width / float(height) if height else 0.0


def aspect_ratio_delta(image_width: int, image_height: int, mask_width: int, mask_height: int) -> float:
    image_ratio = aspect_ratio(image_width, image_height)
    mask_ratio = aspect_ratio(mask_width, mask_height)
    if image_ratio == 0:
        return 1.0
    return abs(image_ratio - mask_ratio) / image_ratio


def ratio_level(positive_ratio: float) -> str:
    if positive_ratio == 0:
        return "empty"
    if positive_ratio < 0.0005:
        return "too_small"
    if positive_ratio > 0.35:
        return "too_large"
    return "normal"


def save_overlay(path: Path, image_bgr: np.ndarray, fixed_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positive = fixed_mask > 0
    overlay = image_bgr.copy().astype(np.float32)
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    alpha = positive.astype(np.float32)[:, :, None] * 0.45
    overlay = overlay * (1.0 - alpha) + red * alpha
    if not cv2.imwrite(str(path), np.clip(overlay, 0, 255).astype(np.uint8)):
        raise RuntimeError(f"Không ghi được overlay: {path}")


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


def validate_original_pair(image: np.ndarray | None, mask_raw: np.ndarray | None) -> bool:
    if image is None or mask_raw is None:
        return False
    mask_gray = mask_to_gray(mask_raw)
    same_size = image.shape[:2] == mask_gray.shape[:2]
    return same_size and is_binary(mask_gray) and np.count_nonzero(mask_gray == 255) > 0


def run_fix(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.threshold <= 255:
        raise ValueError("--threshold phải nằm trong [0, 255].")

    dataset_root = resolve_path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset-root: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset-root không phải thư mục: {dataset_root}")

    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    masks_fixed_dir = dataset_root / "masks_fixed"
    quality_dir = dataset_root / "quality_checks_fixed"
    overlays_dir = dataset_root / "overlays_fixed"
    splits_dir = dataset_root / "splits_fixed"

    if not images_dir.exists():
        raise FileNotFoundError(f"Thiếu thư mục images/: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Thiếu thư mục masks/: {masks_dir}")
    for directory in (masks_fixed_dir, quality_dir, overlays_dir, splits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_lookup = build_lookup(list_supported_files(images_dir, IMAGE_EXTENSIONS), image_id_from_image)
    mask_lookup = build_lookup(list_supported_files(masks_dir, MASK_EXTENSIONS), image_id_from_mask)

    validation_rows: list[dict[str, Any]] = []
    size_rows: list[dict[str, Any]] = []
    non_binary_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    valid_ids: list[str] = []
    split_by_id: dict[str, str] = {}

    valid_before = 0
    binarized_count = 0
    resized_count = 0
    unsafe_size_mismatch_count = 0

    for image_id in sorted(image_lookup):
        image_path = image_lookup[image_id]
        original_mask_path = mask_lookup.get(image_id)
        fixed_mask_path = masks_fixed_dir / f"{image_id}_mask.png"
        image = read_image(image_path)
        mask_raw = read_mask(original_mask_path) if original_mask_path else None

        flags: list[str] = []
        notes: list[str] = []
        width: int | str = ""
        height: int | str = ""
        mask_width: int | str = ""
        mask_height: int | str = ""
        positive_ratio = 0.0
        fixed_exists = False
        action = "none"
        fixed_mask: np.ndarray | None = None

        if validate_original_pair(image, mask_raw):
            valid_before += 1

        if image is None:
            flags.append("unreadable_image")
            notes.append("Không đọc được ảnh gốc.")
        else:
            height, width = int(image.shape[0]), int(image.shape[1])

        if original_mask_path is None:
            flags.append("missing_original_mask")
            notes.append("Không tìm thấy mask gốc tương ứng.")
        elif mask_raw is None:
            flags.append("unreadable_original_mask")
            notes.append("Không đọc được mask gốc.")
        else:
            mask_gray = mask_to_gray(mask_raw)
            original_mask_height, original_mask_width = int(mask_gray.shape[0]), int(mask_gray.shape[1])
            mask_width, mask_height = original_mask_width, original_mask_height
            original_unique_sample = unique_values_sample(mask_gray)
            original_binary = is_binary(mask_gray)
            if not original_binary:
                non_binary_rows.append(
                    {
                        "image_id": image_id,
                        "original_mask_path": str(original_mask_path),
                        "unique_values_sample": original_unique_sample,
                        "threshold": args.threshold,
                        "action": "binarized",
                        "notes": "Đã threshold mask gốc sang binary 0/255 trong masks_fixed/.",
                    }
                )

            can_process = image is not None
            if can_process and (width, height) != (original_mask_width, original_mask_height):
                delta = aspect_ratio_delta(int(width), int(height), original_mask_width, original_mask_height)
                if args.resize_safe and delta <= ASPECT_RATIO_TOLERANCE:
                    resized = cv2.resize(mask_gray, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
                    fixed_mask = binarize(resized, args.threshold)
                    flags.append("resized_mask")
                    action = "resized_mask"
                    resized_count += 1
                    size_note = f"Resize nearest-neighbor vì aspect ratio lệch {delta:.6f} <= 0.005."
                else:
                    flags.append("unsafe_size_mismatch")
                    action = "unsafe_size_mismatch"
                    unsafe_size_mismatch_count += 1
                    size_note = f"Không resize vì aspect ratio lệch {delta:.6f} > 0.005 hoặc chưa bật --resize-safe."
                size_rows.append(
                    {
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "original_mask_path": str(original_mask_path),
                        "original_image_width": width,
                        "original_image_height": height,
                        "original_mask_width": original_mask_width,
                        "original_mask_height": original_mask_height,
                        "action": action,
                        "notes": size_note,
                    }
                )
            elif can_process:
                fixed_mask = binarize(mask_gray, args.threshold)
                action = "binarized"

            if fixed_mask is not None:
                binarized_count += 1
                if not cv2.imwrite(str(fixed_mask_path), fixed_mask):
                    raise RuntimeError(f"Không ghi được fixed mask: {fixed_mask_path}")
                fixed_exists = True
                mask_width, mask_height = int(fixed_mask.shape[1]), int(fixed_mask.shape[0])
                positive_ratio = float(np.count_nonzero(fixed_mask == 255) / fixed_mask.size)
                level = ratio_level(positive_ratio)
                if level == "empty":
                    flags.append("mask_empty")
                elif level == "too_small":
                    flags.append("mask_too_small")
                elif level == "too_large":
                    flags.append("mask_too_large")
                if not is_binary(fixed_mask):
                    flags.append("non_binary_remaining")
                if image is not None and image.shape[:2] != fixed_mask.shape[:2]:
                    flags.append("size_mismatch_remaining")

        blocking_flags = {
            "missing_original_mask",
            "unreadable_image",
            "unreadable_original_mask",
            "unsafe_size_mismatch",
            "non_binary_remaining",
            "size_mismatch_remaining",
            "mask_empty",
        }
        status = "pass" if fixed_exists and not (set(flags) & blocking_flags) else "fail"
        if status == "pass":
            valid_ids.append(image_id)
            if args.make_overlays and image is not None and fixed_mask is not None:
                save_overlay(overlays_dir / f"{image_id}_overlay.png", image, fixed_mask)

        validation_rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "original_mask_path": str(original_mask_path) if original_mask_path else "",
                "fixed_mask_path": str(fixed_mask_path) if fixed_exists else "",
                "status": status,
                "width": width,
                "height": height,
                "mask_width": mask_width,
                "mask_height": mask_height,
                "is_binary": bool(fixed_exists and fixed_mask is not None and is_binary(fixed_mask)),
                "positive_ratio": f"{positive_ratio:.8f}",
                "flags": "|".join(sorted(set(flags))),
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

    for row in validation_rows:
        image_id = row["image_id"]
        image_path = image_lookup[image_id]
        original_mask_path = mask_lookup.get(image_id)
        fixed_mask_path = masks_fixed_dir / f"{image_id}_mask.png"
        index_rows.append(
            {
                "image_id": image_id,
                "filename": image_path.name,
                "original_mask_filename": original_mask_path.name if original_mask_path else "",
                "fixed_mask_filename": fixed_mask_path.name if fixed_mask_path.exists() else "",
                "split": split_by_id.get(image_id, ""),
                "positive_ratio": row["positive_ratio"],
                "flags": row["flags"],
                "notes": row["notes"],
            }
        )

    fixed_mask_count = len(list_supported_files(masks_fixed_dir, MASK_EXTENSIONS))
    non_binary_remaining = sum(1 for row in validation_rows if "non_binary_remaining" in row["flags"].split("|"))
    size_mismatch_remaining = sum(1 for row in validation_rows if "size_mismatch_remaining" in row["flags"].split("|"))
    empty_count = sum(1 for row in validation_rows if "mask_empty" in row["flags"].split("|"))
    too_small_count = sum(1 for row in validation_rows if "mask_too_small" in row["flags"].split("|"))
    too_large_count = sum(1 for row in validation_rows if "mask_too_large" in row["flags"].split("|"))

    summary = {
        "dataset_root": str(dataset_root),
        "threshold": args.threshold,
        "resize_safe": bool(args.resize_safe),
        "total_images": len(image_lookup),
        "total_original_masks": len(mask_lookup),
        "total_fixed_masks": fixed_mask_count,
        "valid_pairs_before_fix": valid_before,
        "valid_pairs_after_fix": len(valid_ids),
        "binarized_count": binarized_count,
        "resized_count": resized_count,
        "unsafe_size_mismatch_count": unsafe_size_mismatch_count,
        "non_binary_remaining_count": non_binary_remaining,
        "size_mismatch_remaining_count": size_mismatch_remaining,
        "empty_mask_count": empty_count,
        "too_small_count": too_small_count,
        "too_large_count": too_large_count,
        "split_counts": {
            "train": sum(1 for split in split_by_id.values() if split == "train"),
            "val": sum(1 for split in split_by_id.values() if split == "val"),
            "test": sum(1 for split in split_by_id.values() if split == "test"),
        },
        "recommended_next_action": "",
    }
    if len(valid_ids) == len(image_lookup) and unsafe_size_mismatch_count == 0 and non_binary_remaining == 0:
        summary["recommended_next_action"] = "Dataset fixed sẵn sàng cho bước fine-tune segmentation r013."
    else:
        summary["recommended_next_action"] = (
            "Cần xử lý các mẫu unsafe_size_mismatch hoặc lỗi còn lại trước khi fine-tune toàn bộ dataset."
        )

    write_csv(
        quality_dir / "fixed_mask_validation_report.csv",
        validation_rows,
        [
            "image_id",
            "image_path",
            "original_mask_path",
            "fixed_mask_path",
            "status",
            "width",
            "height",
            "mask_width",
            "mask_height",
            "is_binary",
            "positive_ratio",
            "flags",
            "notes",
        ],
    )
    write_csv(
        quality_dir / "fixed_size_mismatch_report.csv",
        size_rows,
        [
            "image_id",
            "image_path",
            "original_mask_path",
            "original_image_width",
            "original_image_height",
            "original_mask_width",
            "original_mask_height",
            "action",
            "notes",
        ],
    )
    write_csv(
        quality_dir / "fixed_non_binary_report.csv",
        non_binary_rows,
        ["image_id", "original_mask_path", "unique_values_sample", "threshold", "action", "notes"],
    )
    with (quality_dir / "fixed_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_csv(
        dataset_root / "dataset_index_fixed.csv",
        index_rows,
        [
            "image_id",
            "filename",
            "original_mask_filename",
            "fixed_mask_filename",
            "split",
            "positive_ratio",
            "flags",
            "notes",
        ],
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = run_fix(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
