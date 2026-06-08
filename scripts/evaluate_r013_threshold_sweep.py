#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from scripts.train_r013_finetune import R013SegmentationDataset, load_model, resolve_device, resolve_path
from src.utils.metrics import binary_f1, binary_iou, binary_precision, binary_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sweep cho checkpoint r013.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--image-dir-name", default="images")
    parser.add_argument("--mask-dir-name", default="masks_fixed")
    parser.add_argument("--split-dir-name", default="splits_fixed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70")
    parser.add_argument("--output-name", default="", help="Tên CSV output, mặc định threshold_sweep_<split>.csv.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("--thresholds phải là danh sách số trong [0,1].")
    return values


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["threshold", "mean_iou", "mean_f1", "precision", "recall", "mean_mask_ratio"],
        )
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    device = resolve_device(args.device)
    dataset = R013SegmentationDataset(
        resolve_path(args.dataset_root),
        args.split,
        args.image_dir_name,
        args.mask_dir_name,
        args.split_dir_name,
        args.image_size,
        train=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model, _ = load_model(resolve_path(args.checkpoint), device)
    model.eval()
    sums = {threshold: {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "mask_ratio": 0.0, "count": 0} for threshold in thresholds}
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        for threshold in thresholds:
            sums[threshold]["iou"] += float(binary_iou(logits, masks, threshold=threshold).detach().cpu())
            sums[threshold]["f1"] += float(binary_f1(logits, masks, threshold=threshold).detach().cpu())
            sums[threshold]["precision"] += float(binary_precision(logits, masks, threshold=threshold).detach().cpu())
            sums[threshold]["recall"] += float(binary_recall(logits, masks, threshold=threshold).detach().cpu())
            sums[threshold]["mask_ratio"] += float((probs >= threshold).float().mean().detach().cpu())
            sums[threshold]["count"] += 1
    rows: list[dict[str, str]] = []
    for threshold in thresholds:
        count = max(sums[threshold]["count"], 1)
        rows.append(
            {
                "threshold": f"{threshold:.2f}",
                "mean_iou": f"{sums[threshold]['iou'] / count:.6f}",
                "mean_f1": f"{sums[threshold]['f1'] / count:.6f}",
                "precision": f"{sums[threshold]['precision'] / count:.6f}",
                "recall": f"{sums[threshold]['recall'] / count:.6f}",
                "mean_mask_ratio": f"{sums[threshold]['mask_ratio'] / count:.6f}",
            }
        )
    output_name = args.output_name or f"threshold_sweep_{args.split}.csv"
    output_path = resolve_path(args.output_dir) / output_name
    write_csv(output_path, rows)
    best_iou = max(rows, key=lambda row: float(row["mean_iou"]))
    best_f1 = max(rows, key=lambda row: float(row["mean_f1"]))
    print(f"device: {device}")
    print(f"checkpoint: {resolve_path(args.checkpoint)}")
    print(f"output_csv: {output_path}")
    print(f"best_by_iou: {best_iou}")
    print(f"best_by_f1: {best_f1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
