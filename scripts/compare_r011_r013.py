#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
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
    parser = argparse.ArgumentParser(description="So sánh checkpoint r011 và r013 trên dataset fixed.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--image-dir-name", default="images")
    parser.add_argument("--mask-dir-name", default="masks_fixed")
    parser.add_argument("--split-dir-name", default="splits_fixed")
    parser.add_argument("--r011-checkpoint", required=True)
    parser.add_argument("--r013-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--r011-threshold", type=float, default=0.5)
    parser.add_argument("--r013-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-grid-items", type=int, default=8)
    parser.add_argument("--csv-name", default="", help="Tên CSV output, mặc định r011_vs_r013_<split>.csv.")
    parser.add_argument("--grid-name", default="", help="Tên PNG grid output, mặc định comparison_grid_<split>.png.")
    parser.add_argument("--select-interesting", action="store_true", help="Chọn case grid theo chênh lệch metric thay vì lấy tuần tự.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "split", "threshold", "iou", "f1", "precision", "recall", "mask_ratio"])
        writer.writeheader()
        writer.writerows(rows)


def metrics_for(logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    return {
        "iou": float(binary_iou(logits, masks, threshold=threshold).detach().cpu()),
        "f1": float(binary_f1(logits, masks, threshold=threshold).detach().cpu()),
        "precision": float(binary_precision(logits, masks, threshold=threshold).detach().cpu()),
        "recall": float(binary_recall(logits, masks, threshold=threshold).detach().cpu()),
        "mask_ratio": float(preds.mean().detach().cpu()),
    }


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy().astype(np.float32)
    red = np.zeros_like(result)
    red[:, :, 0] = 255
    alpha = (mask > 0).astype(np.float32)[:, :, None] * 0.45
    return np.clip(result * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def f1_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bool = pred > 0
    target_bool = target > 0
    tp = float(np.logical_and(pred_bool, target_bool).sum())
    fp = float(np.logical_and(pred_bool, ~target_bool).sum())
    fn = float(np.logical_and(~pred_bool, target_bool).sum())
    return (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)


@torch.no_grad()
def make_grid(
    dataset: R013SegmentationDataset,
    r011: torch.nn.Module,
    r013: torch.nn.Module,
    device: torch.device,
    path: Path,
    r011_threshold: float,
    r013_threshold: float,
    max_items: int,
    select_interesting: bool,
) -> None:
    rows: list[np.ndarray] = []
    r011.eval()
    r013.eval()

    selected_indices = list(range(min(len(dataset), max_items)))
    cached: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    if select_interesting:
        scored: list[dict[str, float | int]] = []
        for index in range(len(dataset)):
            sample = dataset[index]
            image_tensor = sample["image"].unsqueeze(0).to(device)
            gt = (sample["mask"].numpy()[0] >= 0.5).astype(np.uint8) * 255
            image = (sample["image"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pred011 = (torch.sigmoid(r011(image_tensor))[0, 0].detach().cpu().numpy() >= r011_threshold).astype(np.uint8) * 255
            pred013 = (torch.sigmoid(r013(image_tensor))[0, 0].detach().cpu().numpy() >= r013_threshold).astype(np.uint8) * 255
            f1_011 = f1_numpy(pred011, gt)
            f1_013 = f1_numpy(pred013, gt)
            cached[index] = (image, gt, pred011, pred013)
            scored.append(
                {
                    "index": index,
                    "f1_011": f1_011,
                    "f1_013": f1_013,
                    "delta": f1_013 - f1_011,
                    "both_fail": max(f1_011, f1_013),
                    "gt_ratio": float((gt > 0).mean()),
                }
            )
        candidates: list[int] = []
        selectors = [
            sorted(scored, key=lambda item: float(item["delta"]), reverse=True),
            sorted(scored, key=lambda item: float(item["delta"])),
            sorted(scored, key=lambda item: float(item["both_fail"])),
            sorted(scored, key=lambda item: abs(float(item["gt_ratio"]) - 0.02)),
            sorted(scored, key=lambda item: float(item["gt_ratio"]), reverse=True),
        ]
        for group in selectors:
            for item in group[:3]:
                index = int(item["index"])
                if index not in candidates:
                    candidates.append(index)
                if len(candidates) >= max_items:
                    break
            if len(candidates) >= max_items:
                break
        selected_indices = candidates[:max_items]

    for index in selected_indices:
        sample = dataset[index]
        if index in cached:
            image, gt, pred011, pred013 = cached[index]
        else:
            image_tensor = sample["image"].unsqueeze(0).to(device)
            gt = (sample["mask"].numpy()[0] >= 0.5).astype(np.uint8) * 255
            image = (sample["image"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pred011 = (torch.sigmoid(r011(image_tensor))[0, 0].detach().cpu().numpy() >= r011_threshold).astype(np.uint8) * 255
            pred013 = (torch.sigmoid(r013(image_tensor))[0, 0].detach().cpu().numpy() >= r013_threshold).astype(np.uint8) * 255
        tiles = [
            image,
            cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(pred011, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(pred013, cv2.COLOR_GRAY2RGB),
            overlay(image, pred011),
            overlay(image, pred013),
        ]
        rows.append(np.concatenate(tiles, axis=1))
    if rows:
        grid = np.concatenate(rows, axis=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def main() -> int:
    args = parse_args()
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
    r011, _ = load_model(resolve_path(args.r011_checkpoint), device)
    r013, _ = load_model(resolve_path(args.r013_checkpoint), device)
    r011.eval()
    r013.eval()
    sums = {
        "r011": {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "mask_ratio": 0.0, "count": 0},
        "r013": {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "mask_ratio": 0.0, "count": 0},
    }
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits011 = r011(images)
        logits013 = r013(images)
        for name, logits, threshold in (("r011", logits011, args.r011_threshold), ("r013", logits013, args.r013_threshold)):
            current = metrics_for(logits, masks, threshold)
            for key, value in current.items():
                sums[name][key] += value
            sums[name]["count"] += 1
    rows: list[dict[str, str]] = []
    for name, threshold in (("r011", args.r011_threshold), ("r013", args.r013_threshold)):
        count = max(sums[name]["count"], 1)
        rows.append(
            {
                "model": name,
                "split": args.split,
                "threshold": f"{threshold:.2f}",
                "iou": f"{sums[name]['iou'] / count:.6f}",
                "f1": f"{sums[name]['f1'] / count:.6f}",
                "precision": f"{sums[name]['precision'] / count:.6f}",
                "recall": f"{sums[name]['recall'] / count:.6f}",
                "mask_ratio": f"{sums[name]['mask_ratio'] / count:.6f}",
            }
        )
    output_dir = resolve_path(args.output_dir)
    csv_path = output_dir / (args.csv_name or f"r011_vs_r013_{args.split}.csv")
    grid_path = output_dir / (args.grid_name or f"comparison_grid_{args.split}.png")
    write_csv(csv_path, rows)
    make_grid(
        dataset,
        r011,
        r013,
        device,
        grid_path,
        args.r011_threshold,
        args.r013_threshold,
        args.max_grid_items,
        args.select_interesting,
    )
    print(f"device: {device}")
    print(f"csv: {csv_path}")
    print(f"grid: {grid_path}")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
