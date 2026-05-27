#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".mat"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chuẩn hóa CrackForest thành crack bank RGBA.")
    parser.add_argument(
        "--source-root",
        default=r"F:\deeplearning\_external_datasets\CrackForest-dataset",
    )
    parser.add_argument(
        "--output-root",
        default="data/crack_bank/processed/rgba",
    )
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--max-area-ratio", type=float, default=0.20)
    parser.add_argument("--crop-margin", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_files(root: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )


def top_level_dirs(paths: list[Path], root: Path) -> list[str]:
    top_dirs = set()
    for path in paths:
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if rel_parts:
            top_dirs.add(rel_parts[0])
    return sorted(top_dirs)


def summarize_source(source_root: Path) -> tuple[list[Path], list[Path]]:
    image_files = list_files(source_root, IMAGE_EXTENSIONS)
    candidate_mask_files = list_files(source_root, MASK_EXTENSIONS)
    print(f"source_root: {source_root}")
    print(f"image files: {len(image_files)}")
    print(f"candidate mask files: {len(candidate_mask_files)}")
    print(f"top-level directories: {', '.join(top_level_dirs(image_files + candidate_mask_files, source_root))}")
    return image_files, candidate_mask_files


def is_likely_mask_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    keywords = ("mask", "groundtruth", "ground_truth", "label", "labels", "annotation", "ann", "gt")
    return any(keyword in part for part in parts for keyword in keywords)


def load_mask_from_mat(path: Path) -> np.ndarray | None:
    try:
        mat = loadmat(path)
    except Exception:
        return None
    if "groundTruth" not in mat:
        return None
    ground_truth = mat["groundTruth"]
    if not isinstance(ground_truth, np.ndarray) or ground_truth.size == 0:
        return None
    try:
        segmentation = ground_truth[0, 0]["Segmentation"]
    except Exception:
        return None
    if not isinstance(segmentation, np.ndarray) or segmentation.ndim != 2:
        return None
    segmentation = np.asarray(segmentation)
    unique_values = np.unique(segmentation)
    if unique_values.size < 2:
        return None
    background_value = unique_values.min()
    binary = (segmentation != background_value).astype(np.uint8) * 255
    if int((binary > 0).sum()) == 0:
        return None
    return binary


def load_mask_from_image(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.ndim != 2:
        return None
    unique_values = np.unique(mask)
    if unique_values.size > 16:
        return None
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if int((binary > 0).sum()) == 0:
        return None
    return binary


def load_reliable_mask(path: Path) -> np.ndarray | None:
    if path.suffix.lower() == ".mat":
        return load_mask_from_mat(path)
    if path.suffix.lower() in IMAGE_EXTENSIONS and is_likely_mask_path(path):
        return load_mask_from_image(path)
    return None


def pair_images_and_masks(image_files: list[Path], candidate_mask_files: list[Path]) -> list[tuple[Path, Path]]:
    mask_map: dict[str, list[Path]] = {}
    for mask_path in candidate_mask_files:
        mask_map.setdefault(mask_path.stem, []).append(mask_path)

    pairs: list[tuple[Path, Path]] = []
    for image_path in image_files:
        candidates = mask_map.get(image_path.stem, [])
        reliable_mask = None
        for candidate in sorted(candidates, key=lambda p: (p.suffix.lower() != ".mat", str(p))):
            if load_reliable_mask(candidate) is not None:
                reliable_mask = candidate
                break
        if reliable_mask is not None:
            pairs.append((image_path, reliable_mask))
    return pairs


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root đã tồn tại: {output_root}. Dùng --overwrite nếu muốn sinh lại crack bank."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh source: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_components(mask: np.ndarray, min_area: int, max_area_ratio: float) -> list[tuple[int, int, int, int, int]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    image_area = float(mask.shape[0] * mask.shape[1])
    components: list[tuple[int, int, int, int, int]] = []
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if (area / image_area) > max_area_ratio:
            continue
        components.append((label_idx, x, y, w, h))
    return components


def crop_with_margin(image: np.ndarray, alpha: np.ndarray, x: int, y: int, w: int, h: int, margin: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = alpha.shape
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(width, x + w + margin)
    y1 = min(height, y + h + margin)
    crop_rgb = image[y0:y1, x0:x1]
    crop_alpha = alpha[y0:y1, x0:x1]
    return crop_rgb, crop_alpha


def smooth_alpha(alpha: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def save_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(path), bgra)


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def relative_to_output(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "crack_id",
        "rgba_path",
        "source_image",
        "source_mask",
        "width",
        "height",
        "alpha_pixels",
        "alpha_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Không tìm thấy source-root: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"source-root không phải thư mục: {source_root}")
    if args.min_area <= 0:
        raise ValueError("--min-area phải > 0.")
    if not 0.0 < args.max_area_ratio <= 1.0:
        raise ValueError("--max-area-ratio phải nằm trong khoảng (0, 1].")
    if args.crop_margin < 0:
        raise ValueError("--crop-margin phải >= 0.")

    image_files, candidate_mask_files = summarize_source(source_root)
    pairs = pair_images_and_masks(image_files, candidate_mask_files)
    print(f"reliable image-mask pairs: {len(pairs)}")
    if not pairs:
        raise RuntimeError("No reliable crack annotation/mask found. Do not generate RGBA from raw road images.")

    prepare_output_root(output_root, args.overwrite)

    manifest_rows: list[dict[str, Any]] = []
    alpha_ratios: list[float] = []
    crack_counter = 0

    for image_path, mask_path in pairs:
        rgb = load_rgb(image_path)
        mask = load_reliable_mask(mask_path)
        if mask is None:
            continue
        if rgb.shape[:2] != mask.shape[:2]:
            continue

        components = extract_components(mask, min_area=args.min_area, max_area_ratio=args.max_area_ratio)
        for label_idx, x, y, w, h in components:
            component_alpha = np.where(
                cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)[1] == label_idx,
                255,
                0,
            ).astype(np.uint8)
            crop_rgb, crop_alpha = crop_with_margin(
                rgb,
                component_alpha,
                x,
                y,
                w,
                h,
                margin=args.crop_margin,
            )
            crop_alpha = smooth_alpha(crop_alpha)
            alpha_pixels = int((crop_alpha > 0).sum())
            if alpha_pixels <= 0:
                continue
            crop_area = int(crop_alpha.shape[0] * crop_alpha.shape[1])
            alpha_ratio = alpha_pixels / float(crop_area)
            if alpha_ratio >= 0.90 or alpha_ratio <= 0.001:
                continue

            rgba = np.dstack([crop_rgb, crop_alpha]).astype(np.uint8)
            crack_counter += 1
            crack_id = f"crack_rgba_{crack_counter:06d}"
            rgba_path = output_root / f"{crack_id}.png"
            save_rgba(rgba_path, rgba)

            manifest_rows.append(
                {
                    "crack_id": crack_id,
                    "rgba_path": relative_to_output(rgba_path, output_root),
                    "source_image": relative_to_project(image_path),
                    "source_mask": relative_to_project(mask_path),
                    "width": rgba.shape[1],
                    "height": rgba.shape[0],
                    "alpha_pixels": alpha_pixels,
                    "alpha_ratio": round(alpha_ratio, 6),
                }
            )
            alpha_ratios.append(alpha_ratio)

    if crack_counter == 0:
        raise RuntimeError("No reliable crack annotation/mask found. Do not generate RGBA from raw road images.")

    manifest_path = output_root / "manifest.csv"
    stats_path = output_root / "stats.json"
    write_manifest(manifest_path, manifest_rows)

    stats_payload = {
        "num_source_images": len(image_files),
        "num_source_masks": len(candidate_mask_files),
        "num_pairs": len(pairs),
        "num_rgba_cracks": crack_counter,
        "min_alpha_ratio": float(min(alpha_ratios)),
        "mean_alpha_ratio": float(np.mean(alpha_ratios)),
        "max_alpha_ratio": float(max(alpha_ratios)),
        "created_at": now_iso(),
        "status": "generated",
        "notes": "RGBA crack bank extracted from reliable CrackForest annotations",
    }
    write_json(stats_path, stats_payload)

    print(f"output_root: {output_root}")
    print(f"num_rgba_cracks: {crack_counter}")
    print(f"manifest: {manifest_path}")
    print(f"stats: {stats_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
