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
import torch
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.losses.segmentation import bce_dice_loss, bce_tversky_loss
from src.models.segmenter import CrackSegmenter
from src.utils.metrics import binary_f1, binary_iou, binary_precision, binary_recall


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
VALID_LOSSES = {"bce_dice", "bce_tversky"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune segmentation r011 thành r013 trên dataset fixed.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--image-dir-name", default="images")
    parser.add_argument("--mask-dir-name", default="masks_fixed")
    parser.add_argument("--split-dir-name", default="splits_fixed")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/r013_finetune")
    parser.add_argument("--run-name", default="r013_gen120_fixed118_local")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=sorted(VALID_LOSSES), default="bce_dice")
    parser.add_argument("--overfit-subset", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--smoke-test", action="store_true", help="Chạy một batch forward/backward rồi dừng.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold metric trong train/eval.")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng nhưng được yêu cầu.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"--device không hợp lệ: {device_arg}")


def read_split_ids(split_path: Path) -> list[str]:
    if not split_path.exists():
        raise FileNotFoundError(f"Không tìm thấy split: {split_path}")
    ids = [line.strip().lstrip("\ufeff") for line in split_path.read_text(encoding="utf-8-sig").splitlines()]
    ids = [item for item in ids if item]
    if not ids:
        raise ValueError(f"Split rỗng: {split_path}")
    return ids


def find_image_path(images_dir: Path, image_id: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Không tìm thấy ảnh cho image_id={image_id} trong {images_dir}")


def load_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_mask_binary(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    unique_values = {int(value) for value in np.unique(mask)}
    if not unique_values <= {0, 255}:
        raise ValueError(f"Mask không binary 0/255: {path}; unique={sorted(unique_values)[:16]}")
    return (mask >= 128).astype(np.uint8) * 255


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).float()


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    mask = (mask >= 128).astype(np.float32)
    return torch.from_numpy(mask[None, :, :]).float()


def apply_train_aug(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.ascontiguousarray(np.flip(image, axis=1))
        mask = np.ascontiguousarray(np.flip(mask, axis=1))
    if random.random() < 0.25:
        alpha = 1.0 + random.uniform(-0.08, 0.08)
        beta = random.uniform(-0.06, 0.06) * 255.0
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return image, mask


class R013SegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split_name: str,
        image_dir_name: str,
        mask_dir_name: str,
        split_dir_name: str,
        image_size: int,
        train: bool,
    ) -> None:
        self.dataset_root = dataset_root
        self.images_dir = dataset_root / image_dir_name
        self.masks_dir = dataset_root / mask_dir_name
        self.split_path = dataset_root / split_dir_name / f"{split_name}.txt"
        self.image_size = image_size
        self.train = train
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Thiếu image dir: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Thiếu mask dir: {self.masks_dir}")
        self.sample_ids = read_split_ids(self.split_path)
        self.samples: list[dict[str, Any]] = []
        for image_id in self.sample_ids:
            image_path = find_image_path(self.images_dir, image_id)
            mask_path = self.masks_dir / f"{image_id}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Thiếu mask fixed cho {image_id}: {mask_path}")
            self.samples.append({"id": image_id, "image_path": image_path, "mask_path": mask_path})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = load_rgb(sample["image_path"])
        mask = load_mask_binary(sample["mask_path"])
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.train:
            image, mask = apply_train_aug(image, mask)
        return {
            "id": sample["id"],
            "image": image_to_tensor(image),
            "mask": mask_to_tensor(mask),
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
        }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint không phải dict: {path}")
    return payload


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[CrackSegmenter, dict[str, Any]]:
    payload = load_checkpoint(checkpoint_path, device)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint thiếu model_state_dict: {checkpoint_path}")
    model_config = payload.get("model_config") or {}
    model = CrackSegmenter(
        in_channels=int(model_config.get("in_channels", 3)),
        out_channels=int(model_config.get("out_channels", 1)),
        base_channels=int(model_config.get("base_channels", 8)),
    ).to(device)
    model.load_state_dict(state_dict)
    return model, payload


def compute_loss(logits: torch.Tensor, masks: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "bce_dice":
        return bce_dice_loss(logits, masks, bce_weight=0.5, dice_weight=0.5)
    if loss_name == "bce_tversky":
        return bce_tversky_loss(logits, masks, bce_weight=0.5, tversky_weight=0.5, alpha=0.3, beta=0.7)
    raise ValueError(f"Loss không hợp lệ: {loss_name}")


def batch_metrics(logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> dict[str, float]:
    return {
        "iou": float(binary_iou(logits, masks, threshold=threshold).detach().cpu()),
        "f1": float(binary_f1(logits, masks, threshold=threshold).detach().cpu()),
        "precision": float(binary_precision(logits, masks, threshold=threshold).detach().cpu()),
        "recall": float(binary_recall(logits, masks, threshold=threshold).detach().cpu()),
    }


def train_one_epoch(
    model: CrackSegmenter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
    threshold: float,
) -> dict[str, float]:
    model.train()
    sums = {"loss": 0.0, "iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = compute_loss(logits, masks, loss_name)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Train loss không hữu hạn: {float(loss.detach().cpu())}")
        loss.backward()
        optimizer.step()
        metrics = batch_metrics(logits.detach(), masks, threshold)
        sums["loss"] += float(loss.detach().cpu())
        for key in ("iou", "f1", "precision", "recall"):
            sums[key] += metrics[key]
        count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


@torch.no_grad()
def evaluate(
    model: CrackSegmenter,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "mask_ratio": 0.0}
    count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = compute_loss(logits, masks, loss_name)
        metrics = batch_metrics(logits, masks, threshold)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        sums["loss"] += float(loss.detach().cpu())
        for key in ("iou", "f1", "precision", "recall"):
            sums[key] += metrics[key]
        sums["mask_ratio"] += float(preds.mean().detach().cpu())
        count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


def save_checkpoint(path: Path, model: CrackSegmenter, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_config": model.get_config(),
        },
        path,
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def make_preview_grid(model: CrackSegmenter, dataset: Dataset, device: torch.device, output_path: Path, threshold: float, max_items: int = 8) -> None:
    model.eval()
    tiles: list[np.ndarray] = []
    with torch.no_grad():
        for index in range(min(len(dataset), max_items)):
            sample = dataset[index]
            image_tensor = sample["image"].unsqueeze(0).to(device)
            mask_tensor = sample["mask"].unsqueeze(0).to(device)
            logits = model(image_tensor)
            pred = (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= threshold).astype(np.uint8) * 255
            image = (sample["image"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            gt = (mask_tensor[0, 0].detach().cpu().numpy() >= 0.5).astype(np.uint8) * 255
            gt_rgb = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
            pred_rgb = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
            overlay = image.copy().astype(np.float32)
            red = np.zeros_like(overlay)
            red[:, :, 0] = 255
            alpha = (pred > 0).astype(np.float32)[:, :, None] * 0.45
            overlay = np.clip(overlay * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)
            tiles.append(np.concatenate([image, gt_rgb, pred_rgb, overlay], axis=1))
    if tiles:
        grid = np.concatenate(tiles, axis=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def build_loaders(args: argparse.Namespace) -> tuple[R013SegmentationDataset, R013SegmentationDataset, R013SegmentationDataset, DataLoader, DataLoader]:
    dataset_root = resolve_path(args.dataset_root)
    train_dataset = R013SegmentationDataset(dataset_root, "train", args.image_dir_name, args.mask_dir_name, args.split_dir_name, args.image_size, train=True)
    val_dataset = R013SegmentationDataset(dataset_root, "val", args.image_dir_name, args.mask_dir_name, args.split_dir_name, args.image_size, train=False)
    test_dataset = R013SegmentationDataset(dataset_root, "test", args.image_dir_name, args.mask_dir_name, args.split_dir_name, args.image_size, train=False)
    if args.overfit_subset and args.overfit_subset > 0:
        subset_count = min(args.overfit_subset, len(train_dataset))
        indices = list(range(subset_count))
        train_dataset_for_loader: Dataset = Subset(train_dataset, indices)
        val_dataset_for_loader: Dataset = Subset(train_dataset, indices)
    else:
        train_dataset_for_loader = train_dataset
        val_dataset_for_loader = val_dataset
    train_loader = DataLoader(train_dataset_for_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset_for_loader, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader


def run_smoke(args: argparse.Namespace, model: CrackSegmenter, loader: DataLoader, device: torch.device, run_dir: Path) -> dict[str, Any]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    logits = model(images)
    loss = compute_loss(logits, masks, args.loss)
    loss.backward()
    optimizer.step()
    payload = {
        "status": "pass",
        "device": str(device),
        "batch_image_shape": list(images.shape),
        "batch_mask_shape": list(masks.shape),
        "mask_min": float(masks.min().detach().cpu()),
        "mask_max": float(masks.max().detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    dump_json(run_dir / "smoke_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs phải > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size phải > 0.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = resolve_path(args.output_dir) / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run output đã tồn tại, dừng để tránh overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader = build_loaders(args)
    model, init_payload = load_model(resolve_path(args.init_checkpoint), device)
    if args.smoke_test:
        run_smoke(args, model, train_loader, device, run_dir)
        return 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metadata = {
        "run_name": args.run_name,
        "created_at": now_iso(),
        "dataset_root": str(resolve_path(args.dataset_root)),
        "image_dir_name": args.image_dir_name,
        "mask_dir_name": args.mask_dir_name,
        "split_dir_name": args.split_dir_name,
        "init_checkpoint": str(resolve_path(args.init_checkpoint)),
        "init_epoch": init_payload.get("epoch"),
        "init_metrics": init_payload.get("metrics", {}),
        "dataset_note": "Dataset generated/synthetic-assisted repaired old-photo, dùng masks_fixed.",
        "device": str(device),
        "train_count": len(train_loader.dataset),
        "val_count": len(val_loader.dataset),
        "test_count": len(test_dataset),
        "args": vars(args),
    }
    dump_json(run_dir / "run_metadata.json", metadata)

    rows: list[dict[str, Any]] = []
    best_iou = -1.0
    best_f1 = -1.0
    best_iou_metrics: dict[str, Any] | None = None
    best_f1_metrics: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    fieldnames = [
        "epoch", "train_loss", "train_iou", "train_f1", "train_precision", "train_recall",
        "val_loss", "val_iou", "val_f1", "val_precision", "val_recall", "val_mask_ratio",
        "improved_iou", "improved_f1", "epochs_without_improvement",
    ]
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.loss, args.threshold)
        val_metrics = evaluate(model, val_loader, device, args.loss, args.threshold)
        improved_iou = val_metrics["iou"] > best_iou + 5e-4
        improved_f1 = val_metrics["f1"] > best_f1 + 5e-4
        if improved_iou:
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_iou_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            save_checkpoint(run_dir / "best_val_iou.pth", model, optimizer, epoch, best_iou_metrics)
        else:
            epochs_without_improvement += 1
        if improved_f1:
            best_f1 = val_metrics["f1"]
            best_f1_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            save_checkpoint(run_dir / "best_val_f1.pth", model, optimizer, epoch, best_f1_metrics)
        current_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        save_checkpoint(run_dir / "last.pth", model, optimizer, epoch, current_metrics)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, current_metrics)
        row = {
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.6f}",
            "train_iou": f"{train_metrics['iou']:.6f}",
            "train_f1": f"{train_metrics['f1']:.6f}",
            "train_precision": f"{train_metrics['precision']:.6f}",
            "train_recall": f"{train_metrics['recall']:.6f}",
            "val_loss": f"{val_metrics['loss']:.6f}",
            "val_iou": f"{val_metrics['iou']:.6f}",
            "val_f1": f"{val_metrics['f1']:.6f}",
            "val_precision": f"{val_metrics['precision']:.6f}",
            "val_recall": f"{val_metrics['recall']:.6f}",
            "val_mask_ratio": f"{val_metrics['mask_ratio']:.6f}",
            "improved_iou": int(improved_iou),
            "improved_f1": int(improved_f1),
            "epochs_without_improvement": epochs_without_improvement,
        }
        rows.append(row)
        write_csv(run_dir / "train_log.csv", rows, fieldnames)
        print(
            f"epoch={epoch} train_loss={row['train_loss']} train_iou={row['train_iou']} train_f1={row['train_f1']} "
            f"val_loss={row['val_loss']} val_iou={row['val_iou']} val_f1={row['val_f1']} "
            f"precision={row['val_precision']} recall={row['val_recall']}"
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stopping: epoch={epoch} best_epoch={best_epoch} best_val_iou={best_iou:.6f}")
            break

    summary = {
        "run_name": args.run_name,
        "epochs_requested": args.epochs,
        "epochs_executed": len(rows),
        "best_epoch_by_iou": best_iou_metrics["epoch"] if best_iou_metrics else None,
        "best_val_iou": best_iou,
        "best_epoch_by_f1": best_f1_metrics["epoch"] if best_f1_metrics else None,
        "best_val_f1": best_f1,
        "best_iou_metrics": best_iou_metrics,
        "best_f1_metrics": best_f1_metrics,
        "final_row": rows[-1] if rows else {},
        "checkpoint_paths": {
            "best_val_iou": str(run_dir / "best_val_iou.pth"),
            "best_val_f1": str(run_dir / "best_val_f1.pth"),
            "last": str(run_dir / "last.pth"),
        },
    }
    dump_json(run_dir / "metrics_summary.json", summary)
    make_preview_grid(model, val_dataset, device, run_dir / "preview_val_grid.png", args.threshold)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
