#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_MASK_VALUES = {0, 255}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit chất lượng dataset synthetic trước khi train.")
    parser.add_argument("--dataset-id")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--mask-ratio-warn", type=float, default=0.10)
    parser.add_argument("--mask-ratio-reject", type=float, default=0.20)
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config file: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file phải parse ra mapping, nhận được: {type(data)!r}")
    return data


def require_nested(cfg: dict[str, Any], *keys: str) -> Any:
    current: Any = cfg
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Thiếu field config bắt buộc: {'.'.join(traversed)}")
        current = current[key]
    return current


def resolve_dataset_id(cfg: dict[str, Any], args: argparse.Namespace) -> str:
    if args.dataset_id:
        return args.dataset_id
    active_dataset = require_nested(cfg, "processed", "active_dataset")
    if not active_dataset:
        raise ValueError("Cần truyền --dataset-id hoặc khai báo processed.active_dataset trong config.")
    return str(active_dataset)


def ensure_required_paths(dataset_root: Path) -> dict[str, Path]:
    required = {
        "train_images": dataset_root / "train" / "images",
        "train_masks": dataset_root / "train" / "masks",
        "train_gt": dataset_root / "train" / "gt",
        "val_images": dataset_root / "val" / "images",
        "val_masks": dataset_root / "val" / "masks",
        "val_gt": dataset_root / "val" / "gt",
        "manifest": dataset_root / "manifest.csv",
        "stats": dataset_root / "stats.json",
        "dataset_metadata": dataset_root / "dataset_metadata.json",
    }
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Thiếu path bắt buộc `{label}`: {path}")
    return required


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest rỗng: {manifest_path}")
    required_columns = {
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
    }
    missing = required_columns.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Manifest thiếu cột bắt buộc: {sorted(missing)}")
    return rows


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh RGB: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    return mask


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_binary_like(mask: np.ndarray) -> tuple[bool, float]:
    unique_values = np.unique(mask)
    if unique_values.size == 0:
        return False, 0.0
    if set(unique_values.tolist()).issubset(VALID_MASK_VALUES):
        return True, 1.0
    nearest = np.where(mask >= 128, 255, 0).astype(np.uint8)
    binary_fraction = float((nearest == mask).sum()) / float(mask.size)
    return binary_fraction >= 0.99, binary_fraction


def make_overlay(gt: np.ndarray, degraded: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_rgb = np.repeat(mask[:, :, None], 3, axis=2)
    overlay = degraded.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
    overlay = overlay * (1.0 - alpha) + red * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return np.concatenate([gt, degraded, mask_rgb, overlay], axis=1)


def save_overlay(path: Path, panel: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "degraded_path",
        "mask_path",
        "gt_path",
        "status",
        "missing_files",
        "shape_match",
        "mask_binary_like",
        "mask_binary_fraction",
        "mask_nonempty",
        "mask_pixels",
        "mask_ratio",
        "warn_ratio",
        "reject_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    cfg = load_config(config_path)

    dataset_id = resolve_dataset_id(cfg, args)
    processed_root = PROJECT_ROOT / require_nested(cfg, "processed", "root")
    dataset_root = processed_root / dataset_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset root: {dataset_root}")

    ensure_required_paths(dataset_root)
    manifest_rows = read_manifest(dataset_root / "manifest.csv")

    max_samples = len(manifest_rows) if args.max_samples <= 0 else min(args.max_samples, len(manifest_rows))
    selected_rows = manifest_rows[:max_samples]

    audit_root = dataset_root / "audit"
    overlays_root = audit_root / "overlays"
    report_path = audit_root / "audit_report.json"
    samples_path = audit_root / "audit_samples.csv"
    audit_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)

    sample_reports: list[dict[str, Any]] = []
    mask_ratios: list[float] = []
    num_missing_files = 0
    num_empty_masks = 0
    num_shape_mismatch = 0
    num_warn_ratio = 0
    num_reject_ratio = 0

    for row in selected_rows:
        degraded_path = dataset_root / row["degraded_path"]
        mask_path = dataset_root / row["mask_path"]
        gt_path = dataset_root / row["gt_path"]
        missing_files: list[str] = []
        for label, path in [("image", degraded_path), ("mask", mask_path), ("gt", gt_path)]:
            if not path.exists():
                missing_files.append(label)

        report_row: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "degraded_path": row["degraded_path"],
            "mask_path": row["mask_path"],
            "gt_path": row["gt_path"],
            "status": "ok",
            "missing_files": ";".join(missing_files),
            "shape_match": False,
            "mask_binary_like": False,
            "mask_binary_fraction": 0.0,
            "mask_nonempty": False,
            "mask_pixels": 0,
            "mask_ratio": 0.0,
            "warn_ratio": False,
            "reject_ratio": False,
        }

        if missing_files:
            num_missing_files += len(missing_files)
            report_row["status"] = "missing_files"
            sample_reports.append(report_row)
            continue

        try:
            degraded_img = load_rgb(degraded_path)
            gt_img = load_rgb(gt_path)
            mask = load_mask(mask_path)
        except Exception as exc:
            num_missing_files += 1
            report_row["status"] = f"read_error:{exc}"
            sample_reports.append(report_row)
            continue

        shape_match = (
            degraded_img.shape[:2] == gt_img.shape[:2]
            and degraded_img.shape[:2] == mask.shape[:2]
        )
        report_row["shape_match"] = shape_match
        if not shape_match:
            num_shape_mismatch += 1
            report_row["status"] = "shape_mismatch"

        is_binary_like, binary_fraction = mask_binary_like(mask)
        report_row["mask_binary_like"] = is_binary_like
        report_row["mask_binary_fraction"] = round(binary_fraction, 6)

        current_mask_pixels = int((mask > 0).sum())
        report_row["mask_pixels"] = current_mask_pixels
        report_row["mask_nonempty"] = current_mask_pixels > 0
        if current_mask_pixels == 0:
            num_empty_masks += 1
            if report_row["status"] == "ok":
                report_row["status"] = "empty_mask"

        total_pixels = int(mask.shape[0] * mask.shape[1]) if mask.ndim == 2 else 1
        mask_ratio = float(current_mask_pixels) / float(total_pixels)
        report_row["mask_ratio"] = round(mask_ratio, 6)
        mask_ratios.append(mask_ratio)

        warn_ratio = mask_ratio > args.mask_ratio_warn
        reject_ratio = mask_ratio > args.mask_ratio_reject
        report_row["warn_ratio"] = warn_ratio
        report_row["reject_ratio"] = reject_ratio
        if warn_ratio:
            num_warn_ratio += 1
        if reject_ratio:
            num_reject_ratio += 1
            if report_row["status"] == "ok":
                report_row["status"] = "reject_ratio"

        if shape_match:
            overlay_panel = make_overlay(gt_img, degraded_img, mask)
            overlay_path = overlays_root / f"{row['sample_id']}.png"
            save_overlay(overlay_path, overlay_panel)

        sample_reports.append(report_row)

    if mask_ratios:
        mean_mask_ratio = float(np.mean(mask_ratios))
        min_mask_ratio = float(np.min(mask_ratios))
        max_mask_ratio = float(np.max(mask_ratios))
    else:
        mean_mask_ratio = 0.0
        min_mask_ratio = 0.0
        max_mask_ratio = 0.0

    likely_invalid = mean_mask_ratio > args.mask_ratio_reject or num_reject_ratio > max(1, int(0.25 * max_samples))
    report_payload = {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "created_at": now_iso(),
        "num_samples_checked": len(selected_rows),
        "num_missing_files": num_missing_files,
        "num_empty_masks": num_empty_masks,
        "num_shape_mismatch": num_shape_mismatch,
        "mean_mask_ratio": mean_mask_ratio,
        "min_mask_ratio": min_mask_ratio,
        "max_mask_ratio": max_mask_ratio,
        "num_warn_ratio": num_warn_ratio,
        "num_reject_ratio": num_reject_ratio,
        "mask_ratio_warn_threshold": args.mask_ratio_warn,
        "mask_ratio_reject_threshold": args.mask_ratio_reject,
        "likely_invalid_for_crack_segmentation": likely_invalid,
    }

    write_json(report_path, report_payload)
    write_csv(samples_path, sample_reports)

    print(f"dataset_id: {dataset_id}")
    print(f"dataset_root: {dataset_root}")
    print(f"num_samples_checked: {report_payload['num_samples_checked']}")
    print(f"num_missing_files: {num_missing_files}")
    print(f"num_empty_masks: {num_empty_masks}")
    print(f"num_shape_mismatch: {num_shape_mismatch}")
    print(f"mean_mask_ratio: {mean_mask_ratio:.6f}")
    print(f"min_mask_ratio: {min_mask_ratio:.6f}")
    print(f"max_mask_ratio: {max_mask_ratio:.6f}")
    print(f"num_warn_ratio: {num_warn_ratio}")
    print(f"num_reject_ratio: {num_reject_ratio}")
    print(f"audit_report.json: {report_path}")
    print(f"audit_samples.csv: {samples_path}")
    print(f"overlays: {overlays_root}")
    if likely_invalid:
        print("WARNING: DATASET LIKELY INVALID FOR CRACK SEGMENTATION")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
