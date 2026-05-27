#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.degradation import (  # noqa: E402
    generate_degraded_pair,
    load_crack_rgba,
    load_rgb,
    save_mask,
    save_rgb,
)


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class BuildPaths:
    dataset_root: Path
    train_images: Path
    train_masks: Path
    train_gt: Path
    val_images: Path
    val_masks: Path
    val_gt: Path
    previews: Path
    manifest: Path
    stats: Path
    config_snapshot: Path
    dataset_metadata: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sinh synthetic dataset từ DIV2K và CrackBank.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--version")
    parser.add_argument("--degradation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-previews", type=int, default=12)
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file phải parse ra mapping, nhận được: {type(data)!r}")
    return data


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    if args.num_samples is not None:
        cfg["build"]["num_samples"] = args.num_samples
    if args.image_size is not None:
        cfg["build"]["image_size"] = args.image_size
    if args.val_ratio is not None:
        cfg["build"]["val_ratio"] = args.val_ratio
    if args.seed is not None:
        cfg["build"]["seed"] = args.seed
    if args.version is not None:
        cfg["build"]["version"] = args.version
    if args.degradation is not None:
        cfg["build"]["degradation"] = args.degradation
    return cfg


def require_nested(cfg: dict[str, Any], *keys: str) -> Any:
    current: Any = cfg
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Thiếu field config bắt buộc: {'.'.join(traversed)}")
        current = current[key]
    return current


def resolve_required_dir(project_root: Path, value: str, field_name: str) -> Path:
    path = project_root / value
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục `{field_name}`: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Đường dẫn `{field_name}` không phải thư mục: {path}")
    return path


def list_image_files(path: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS
    )


def ensure_nonempty_images(paths: list[Path], label: str) -> None:
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh hợp lệ trong {label}.")


def make_dataset_id(degradation: str, image_size: int, num_samples: int, version: str) -> str:
    return f"ds-{degradation}-{image_size}-n{num_samples:04d}-{version}"


def make_build_paths(dataset_root: Path) -> BuildPaths:
    return BuildPaths(
        dataset_root=dataset_root,
        train_images=dataset_root / "train" / "images",
        train_masks=dataset_root / "train" / "masks",
        train_gt=dataset_root / "train" / "gt",
        val_images=dataset_root / "val" / "images",
        val_masks=dataset_root / "val" / "masks",
        val_gt=dataset_root / "val" / "gt",
        previews=dataset_root / "previews",
        manifest=dataset_root / "manifest.csv",
        stats=dataset_root / "stats.json",
        config_snapshot=dataset_root / "config_snapshot.yaml",
        dataset_metadata=dataset_root / "dataset_metadata.json",
    )


def prepare_output_dir(dataset_root: Path, overwrite: bool) -> None:
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset đã tồn tại, dừng an toàn: {dataset_root}. Dùng --overwrite nếu muốn sinh lại."
            )
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(paths: BuildPaths) -> None:
    for path in [
        paths.train_images,
        paths.train_masks,
        paths.train_gt,
        paths.val_images,
        paths.val_masks,
        paths.val_gt,
        paths.previews,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def prepare_clean_image(img: np.ndarray, image_size: int, rng: random.Random) -> np.ndarray:
    if not isinstance(img, np.ndarray):
        raise TypeError("Ảnh clean phải là numpy array.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Ảnh clean phải có shape (H, W, 3), nhận được {img.shape}")

    img = np.clip(img, 0, 255).astype(np.uint8)
    height, width = img.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Ảnh clean có kích thước không hợp lệ: {img.shape}")

    if height >= image_size and width >= image_size:
        crop_size = min(height, width)
        top = 0 if height == crop_size else rng.randint(0, height - crop_size)
        left = 0 if width == crop_size else rng.randint(0, width - crop_size)
        img = img[top : top + crop_size, left : left + crop_size]

    if img.shape[0] != image_size or img.shape[1] != image_size:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(img.astype(np.uint8))


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def relative_to_dataset(path: Path, dataset_root: Path) -> str:
    return path.relative_to(dataset_root).as_posix()


def make_overlay(clean: np.ndarray, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_rgb = np.repeat(mask[:, :, None], 3, axis=2)
    overlay = degraded.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
    overlay = overlay * (1.0 - alpha) + red * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return np.concatenate([clean, degraded, mask_rgb, overlay], axis=1)


def save_preview(paths: BuildPaths, preview_index: int, clean: np.ndarray, degraded: np.ndarray, mask: np.ndarray) -> None:
    preview_path = paths.previews / f"preview_{preview_index:06d}.png"
    save_rgb(preview_path, make_overlay(clean, degraded, mask))


def write_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "split",
        "degraded_path",
        "mask_path",
        "gt_path",
        "clean_source",
        "crack_source",
        "seed",
        "image_size",
        "mask_pixels",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def update_dataset_registry(
    project_root: Path,
    dataset_id: str,
    num_samples: int,
    image_size: int,
    seed: int,
    version: str,
    created_at: str,
) -> Path:
    registry_path = project_root / "results" / "registry" / "dataset_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "dataset_id",
        "num_samples",
        "image_size",
        "seed",
        "version",
        "created_at",
        "status",
        "notes",
    ]
    write_header = not registry_path.exists() or registry_path.stat().st_size == 0
    with registry_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                dataset_id,
                num_samples,
                image_size,
                seed,
                version,
                created_at,
                "generated",
                "synthetic degradation dataset",
            ]
        )
    return registry_path


def validate_generated_counts(train_generated: int, val_generated: int, num_train: int, num_val: int) -> None:
    if train_generated != num_train or val_generated != num_val:
        raise RuntimeError(
            "Không sinh đủ số sample yêu cầu. "
            f"train={train_generated}/{num_train}, val={val_generated}/{num_val}"
        )


def resolve_crack_dir(cfg: dict[str, Any]) -> tuple[str, Path]:
    crack_source_mode = str(require_nested(cfg, "build", "crack_source"))
    if crack_source_mode == "processed_rgba":
        crack_dir = resolve_required_dir(
            PROJECT_ROOT,
            require_nested(cfg, "raw", "crack_bank_processed_rgba"),
            "raw.crack_bank_processed_rgba",
        )
    elif crack_source_mode == "raw":
        crack_dir = resolve_required_dir(
            PROJECT_ROOT,
            require_nested(cfg, "raw", "crack_bank_raw"),
            "raw.crack_bank_raw",
        )
    else:
        raise ValueError(f"build.crack_source không hợp lệ: {crack_source_mode}")
    return crack_source_mode, crack_dir


def generate_split(
    split_name: str,
    target_count: int,
    clean_images: list[Path],
    crack_images: list[Path],
    image_size: int,
    master_rng: random.Random,
    paths: BuildPaths,
    dataset_root: Path,
    project_root: Path,
    preview_start_index: int,
    num_previews: int,
    manifest_rows: list[dict[str, Any]],
    mask_pixels: list[int],
    mask_ratio_min: float,
    mask_ratio_max: float,
) -> tuple[int, int, int]:
    generated = 0
    failed = 0
    preview_count = 0
    max_attempts = max(1, target_count * 10)
    attempts = 0

    while generated < target_count and attempts < max_attempts:
        attempts += 1
        sample_seed = master_rng.randint(0, 2**31 - 1)
        sample_rng = random.Random(sample_seed)
        random.seed(sample_seed)
        np.random.seed(sample_seed)

        clean_path = clean_images[sample_rng.randrange(len(clean_images))]
        crack_path = crack_images[sample_rng.randrange(len(crack_images))]

        try:
            clean_img = prepare_clean_image(load_rgb(clean_path), image_size, sample_rng)
            crack_rgba = load_crack_rgba(crack_path)
            degraded_img, crack_mask = generate_degraded_pair(clean_img, crack_rgba)

            if degraded_img.shape != (image_size, image_size, 3):
                raise ValueError(f"Degraded image shape không hợp lệ: {degraded_img.shape}")
            if crack_mask.shape != (image_size, image_size):
                raise ValueError(f"Mask shape không hợp lệ: {crack_mask.shape}")

            current_mask_pixels = int((crack_mask > 0).sum())
            mask_ratio = current_mask_pixels / float(image_size * image_size)
            if mask_ratio < mask_ratio_min or mask_ratio > mask_ratio_max:
                failed += 1
                continue

            generated += 1
            sample_id = f"{split_name}_{generated:06d}"
            if split_name == "train":
                image_path = paths.train_images / f"{sample_id}.png"
                mask_path = paths.train_masks / f"{sample_id}.png"
                gt_path = paths.train_gt / f"{sample_id}.png"
            else:
                image_path = paths.val_images / f"{sample_id}.png"
                mask_path = paths.val_masks / f"{sample_id}.png"
                gt_path = paths.val_gt / f"{sample_id}.png"

            save_rgb(image_path, degraded_img)
            save_mask(mask_path, crack_mask)
            save_rgb(gt_path, clean_img)

            mask_pixels.append(current_mask_pixels)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split_name,
                    "degraded_path": relative_to_dataset(image_path, dataset_root),
                    "mask_path": relative_to_dataset(mask_path, dataset_root),
                    "gt_path": relative_to_dataset(gt_path, dataset_root),
                    "clean_source": relative_to_project(clean_path, project_root),
                    "crack_source": relative_to_project(crack_path, project_root),
                    "seed": sample_seed,
                    "image_size": image_size,
                    "mask_pixels": current_mask_pixels,
                }
            )

            if preview_count < num_previews:
                preview_count += 1
                save_preview(
                    paths,
                    preview_start_index + preview_count - 1,
                    clean_img,
                    degraded_img,
                    crack_mask,
                )
        except Exception as exc:
            failed += 1
            print(
                f"WARNING: Lỗi khi sinh sample {split_name} ở attempt {attempts}/{max_attempts}: {exc}",
                file=sys.stderr,
            )

    return generated, failed, preview_count


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    cfg = apply_cli_overrides(load_config(config_path), args)

    raw_train_dir = resolve_required_dir(PROJECT_ROOT, require_nested(cfg, "raw", "div2k_train"), "raw.div2k_train")
    raw_val_dir = resolve_required_dir(PROJECT_ROOT, require_nested(cfg, "raw", "div2k_val"), "raw.div2k_val")
    crack_source_mode, crack_dir = resolve_crack_dir(cfg)
    processed_root = PROJECT_ROOT / require_nested(cfg, "processed", "root")

    clean_train_images = list_image_files(raw_train_dir)
    clean_val_images = list_image_files(raw_val_dir)
    crack_images = [
        path for path in list_image_files(crack_dir)
        if path.name not in {"manifest.csv", "stats.json"}
    ]
    ensure_nonempty_images(clean_train_images, f"raw.div2k_train ({raw_train_dir})")
    ensure_nonempty_images(clean_val_images, f"raw.div2k_val ({raw_val_dir})")
    ensure_nonempty_images(crack_images, f"{crack_source_mode} crack source ({crack_dir})")

    degradation = str(require_nested(cfg, "build", "degradation"))
    image_size = int(require_nested(cfg, "build", "image_size"))
    num_samples = int(require_nested(cfg, "build", "num_samples"))
    val_ratio = float(require_nested(cfg, "build", "val_ratio"))
    seed = int(require_nested(cfg, "build", "seed"))
    version = str(require_nested(cfg, "build", "version"))
    mask_ratio_min = float(require_nested(cfg, "build", "mask_ratio_min"))
    mask_ratio_max = float(require_nested(cfg, "build", "mask_ratio_max"))

    if num_samples <= 0:
        raise ValueError("num_samples phải > 0.")
    if image_size <= 0:
        raise ValueError("image_size phải > 0.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio phải nằm trong khoảng (0, 1).")
    if not 0.0 <= mask_ratio_min < mask_ratio_max <= 1.0:
        raise ValueError("mask_ratio_min/max không hợp lệ.")

    dataset_id = make_dataset_id(degradation, image_size, num_samples, version)
    dataset_root = processed_root / dataset_id
    paths = make_build_paths(dataset_root)

    prepare_output_dir(dataset_root, overwrite=args.overwrite)
    ensure_output_dirs(paths)

    random.seed(seed)
    np.random.seed(seed)
    master_rng = random.Random(seed)

    num_train = int(num_samples * (1.0 - val_ratio))
    num_val = num_samples - num_train
    created_at = now_iso()
    git_commit = get_git_commit(PROJECT_ROOT)

    manifest_rows: list[dict[str, Any]] = []
    mask_pixels: list[int] = []

    train_generated, train_failed, train_preview_count = generate_split(
        split_name="train",
        target_count=num_train,
        clean_images=clean_train_images,
        crack_images=crack_images,
        image_size=image_size,
        master_rng=master_rng,
        paths=paths,
        dataset_root=dataset_root,
        project_root=PROJECT_ROOT,
        preview_start_index=1,
        num_previews=args.num_previews,
        manifest_rows=manifest_rows,
        mask_pixels=mask_pixels,
        mask_ratio_min=mask_ratio_min,
        mask_ratio_max=mask_ratio_max,
    )
    remaining_previews = max(0, args.num_previews - train_preview_count)
    val_generated, val_failed, _ = generate_split(
        split_name="val",
        target_count=num_val,
        clean_images=clean_val_images,
        crack_images=crack_images,
        image_size=image_size,
        master_rng=master_rng,
        paths=paths,
        dataset_root=dataset_root,
        project_root=PROJECT_ROOT,
        preview_start_index=train_preview_count + 1,
        num_previews=remaining_previews,
        manifest_rows=manifest_rows,
        mask_pixels=mask_pixels,
        mask_ratio_min=mask_ratio_min,
        mask_ratio_max=mask_ratio_max,
    )
    num_failed = train_failed + val_failed
    validate_generated_counts(train_generated, val_generated, num_train, num_val)

    write_manifest(paths.manifest, manifest_rows)
    with paths.config_snapshot.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)

    mask_pixels_array = np.asarray(mask_pixels, dtype=np.int64)
    total_pixels = image_size * image_size
    stats_payload = {
        "dataset_id": dataset_id,
        "num_samples": num_samples,
        "num_train": num_train,
        "num_val": num_val,
        "image_size": image_size,
        "seed": seed,
        "mean_mask_pixels": float(mask_pixels_array.mean()),
        "min_mask_pixels": int(mask_pixels_array.min()),
        "max_mask_pixels": int(mask_pixels_array.max()),
        "created_at": created_at,
        "degradation": degradation,
        "version": version,
        "num_failed": num_failed,
        "mask_pixel_ratio_mean": float(mask_pixels_array.mean() / total_pixels),
        "mask_pixel_ratio_min": float(mask_pixels_array.min() / total_pixels),
        "mask_pixel_ratio_max": float(mask_pixels_array.max() / total_pixels),
        "clean_train_count": len(clean_train_images),
        "clean_val_count": len(clean_val_images),
        "crack_count": len(crack_images),
        "crack_source_mode": crack_source_mode,
    }
    write_json(paths.stats, stats_payload)

    metadata_payload = {
        "dataset_id": dataset_id,
        "created_at": created_at,
        "image_size": image_size,
        "num_samples": num_samples,
        "num_train": num_train,
        "num_val": num_val,
        "seed": seed,
        "clean_sources": [
            relative_to_project(raw_train_dir, PROJECT_ROOT),
            relative_to_project(raw_val_dir, PROJECT_ROOT),
        ],
        "crack_sources": [relative_to_project(crack_dir, PROJECT_ROOT)],
        "generation_script": relative_to_project(Path(__file__), PROJECT_ROOT),
        "config_snapshot": relative_to_dataset(paths.config_snapshot, dataset_root),
        "status": "generated",
        "notes": f"synthetic degradation dataset using {crack_source_mode}",
        "git_commit": git_commit,
    }
    write_json(paths.dataset_metadata, metadata_payload)

    registry_path = update_dataset_registry(
        project_root=PROJECT_ROOT,
        dataset_id=dataset_id,
        num_samples=num_samples,
        image_size=image_size,
        seed=seed,
        version=version,
        created_at=created_at,
    )

    print(f"dataset_id: {dataset_id}")
    print(f"dataset_root: {dataset_root}")
    print(f"num_train generated: {train_generated}")
    print(f"num_val generated: {val_generated}")
    print(f"num_failed: {num_failed}")
    print(f"mean_mask_pixels: {stats_payload['mean_mask_pixels']}")
    print(f"min_mask_pixels: {stats_payload['min_mask_pixels']}")
    print(f"max_mask_pixels: {stats_payload['max_mask_pixels']}")
    print(f"manifest path: {paths.manifest}")
    print(f"stats path: {paths.stats}")
    print(f"previews path: {paths.previews}")
    print(f"registry updated path: {registry_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
