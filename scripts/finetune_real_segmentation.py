from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset


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


VALID_LOSS_TYPES = {"bce_dice", "bce_tversky"}
IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune segmentation model trên real old-photo crack dataset.")
    parser.add_argument("--data-root", required=True, help="Thư mục ngoài repo chứa images/ và masks/.")
    parser.add_argument("--image-dir", default="images", help="Tên thư mục ảnh tương đối trong data-root.")
    parser.add_argument("--mask-dir", default="masks", help="Tên thư mục mask tương đối trong data-root.")
    parser.add_argument("--split-dir", required=True, help="Thư mục chứa train.txt, val.txt, test.txt.")
    parser.add_argument("--init-checkpoint", required=True, help="Checkpoint khởi tạo từ run synthetic tốt nhất.")
    parser.add_argument("--run-id", required=True, help="Run ID mới, ví dụ seg-unet-attn-r010-real-ft-s42.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--loss-type", choices=sorted(VALID_LOSS_TYPES), default="bce_tversky")
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5, help="Dùng cho Dice hoặc Tversky component.")
    parser.add_argument("--tversky-alpha", type=float, default=0.3)
    parser.add_argument("--tversky-beta", type=float, default=0.7)
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-run", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


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
            raise RuntimeError("CUDA không khả dụng nhưng được yêu cầu qua --device cuda.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"--device không hợp lệ: {device_arg}")


def prepare_run_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Run đã tồn tại, dừng an toàn: {path}. Dùng --overwrite-run nếu muốn chạy lại.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_iou",
        "val_f1",
        "val_precision",
        "val_recall",
        "improved",
        "epochs_without_improvement",
        "early_stop_triggered",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint phải là dict, nhận được: {type(checkpoint)!r}")
    return checkpoint


def find_image_path(images_dir: Path, sample_id: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{sample_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Không tìm thấy ảnh cho id={sample_id} trong {images_dir}")


def read_split_ids(split_file: Path) -> list[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Không tìm thấy split file: {split_file}")
    ids = [line.strip().lstrip("\ufeff") for line in split_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"Split file rỗng: {split_file}")
    return ids


def load_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh RGB: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    return mask


def to_image_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.clip(image, 0, 255).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).float()


def to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask[None, :, :]).float()


def apply_light_train_aug(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.ascontiguousarray(np.flip(image, axis=1))
        mask = np.ascontiguousarray(np.flip(mask, axis=1))

    if random.random() < 0.35:
        alpha = 1.0 + random.uniform(-0.10, 0.10)
        beta = random.uniform(-0.08, 0.08) * 255.0
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.12:
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=0.0)

    if random.random() < 0.12:
        noise = np.random.normal(loc=0.0, scale=4.0, size=image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, mask


class RealOldPhotoCrackDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split_file: str | Path,
        image_size: int,
        train: bool = False,
        image_dir: str = "images",
        mask_dir: str = "masks",
    ) -> None:
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images_dir = self.data_root / image_dir
        self.masks_dir = self.data_root / mask_dir

        if self.image_size <= 0:
            raise ValueError("image_size phải > 0.")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy images/: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy masks/: {self.masks_dir}")

        self.sample_ids = read_split_ids(self.split_file)
        self.samples: list[dict[str, Any]] = []
        for sample_id in self.sample_ids:
            image_path = find_image_path(self.images_dir, sample_id)
            mask_path = self.masks_dir / f"{sample_id}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Không tìm thấy mask cho id={sample_id}: {mask_path}")
            self.samples.append({"id": sample_id, "image_path": image_path, "mask_path": mask_path})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = load_rgb(sample["image_path"])
        mask = load_mask(sample["mask_path"])

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.train:
            image, mask = apply_light_train_aug(image, mask)

        return {
            "id": sample["id"],
            "image": to_image_tensor(image),
            "mask": to_mask_tensor(mask),
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
        }


def compute_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    loss_type: str,
    bce_weight: float,
    dice_weight: float,
    tversky_alpha: float,
    tversky_beta: float,
) -> torch.Tensor:
    if loss_type == "bce_dice":
        return bce_dice_loss(logits, masks, bce_weight=bce_weight, dice_weight=dice_weight)
    if loss_type == "bce_tversky":
        return bce_tversky_loss(
            logits,
            masks,
            bce_weight=bce_weight,
            tversky_weight=dice_weight,
            alpha=tversky_alpha,
            beta=tversky_beta,
        )
    raise ValueError(f"loss_type không hợp lệ: {loss_type}")


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[CrackSegmenter, dict[str, Any]]:
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint thiếu `model_state_dict`: {checkpoint_path}")

    model_config = checkpoint.get("model_config") or {}
    model = CrackSegmenter(
        in_channels=int(model_config.get("in_channels", 3)),
        out_channels=int(model_config.get("out_channels", 1)),
        base_channels=int(model_config.get("base_channels", 8)),
    ).to(device)
    model.load_state_dict(state_dict)
    return model, checkpoint


def save_checkpoint(
    path: Path,
    model: CrackSegmenter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
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


def train_one_epoch(
    model: CrackSegmenter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = compute_loss(
            logits=logits,
            masks=masks,
            loss_type=args.loss_type,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Train loss không hữu hạn: {float(loss.detach().cpu())}")
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batch_count += 1

    return total_loss / max(batch_count, 1)


@torch.no_grad()
def evaluate_one_epoch(
    model: CrackSegmenter,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    iou_sum = 0.0
    f1_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    batch_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = compute_loss(
            logits=logits,
            masks=masks,
            loss_type=args.loss_type,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Validation loss không hữu hạn: {float(loss.detach().cpu())}")

        total_loss += float(loss.detach().cpu())
        iou_sum += float(binary_iou(logits, masks).detach().cpu())
        f1_sum += float(binary_f1(logits, masks).detach().cpu())
        precision_sum += float(binary_precision(logits, masks).detach().cpu())
        recall_sum += float(binary_recall(logits, masks).detach().cpu())
        batch_count += 1

    divisor = max(batch_count, 1)
    return {
        "val_loss": total_loss / divisor,
        "val_iou": iou_sum / divisor,
        "val_f1": f1_sum / divisor,
        "val_precision": precision_sum / divisor,
        "val_recall": recall_sum / divisor,
    }


def make_config_snapshot(args: argparse.Namespace, model_config: dict[str, Any], train_size: int, val_size: int) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "created_at": now_iso(),
        "data_root": str(Path(args.data_root).resolve()),
        "split_dir": str(Path(args.split_dir).resolve()),
        "image_dir": args.image_dir,
        "mask_dir": args.mask_dir,
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "train_size": train_size,
        "val_size": val_size,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "image_size": args.image_size,
            "loss_type": args.loss_type,
            "bce_weight": args.bce_weight,
            "dice_weight": args.dice_weight,
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
            "device": args.device,
            "num_workers": args.num_workers,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "seed": args.seed,
            "augmentation": "horizontal_flip + light_brightness_contrast + light_blur_noise",
        },
        "model_config": model_config,
        "checkpoint_format": ["epoch", "model_state_dict", "optimizer_state_dict", "metrics", "model_config"],
    }


def make_notes(args: argparse.Namespace, train_size: int, val_size: int) -> str:
    return "\n".join(
        [
            f"# {args.run_id}",
            "",
            "- Fine-tune segmentation model r009 trên real old-photo crack dataset.",
            "- Dataset nguồn nằm ngoài repo; không copy ảnh/mask vào repo.",
            f"- Train size: `{train_size}`",
            f"- Val size: `{val_size}`",
            f"- Loss: `{args.loss_type}`",
            f"- Tversky alpha/beta: `{args.tversky_alpha}` / `{args.tversky_beta}`",
            f"- Image size: `{args.image_size}`",
            f"- Seed: `{args.seed}`",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs phải > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size phải > 0.")
    if args.image_size <= 0:
        raise ValueError("--image-size phải > 0.")
    if args.lr <= 0.0:
        raise ValueError("--lr phải > 0.")
    if args.patience < 0:
        raise ValueError("--patience phải >= 0.")
    if args.min_delta < 0.0:
        raise ValueError("--min-delta phải >= 0.")

    set_seed(args.seed)
    device = resolve_device(args.device)

    data_root = Path(args.data_root).resolve()
    split_dir = Path(args.split_dir).resolve()
    init_checkpoint = Path(args.init_checkpoint).resolve()
    checkpoints_root = PROJECT_ROOT / "checkpoints" / "segmenter" / args.run_id
    experiments_root = PROJECT_ROOT / "experiments" / "segmenter" / args.run_id
    prepare_run_dir(checkpoints_root, args.overwrite_run)
    prepare_run_dir(experiments_root, args.overwrite_run)

    train_dataset = RealOldPhotoCrackDataset(
        data_root,
        split_dir / "train.txt",
        image_size=args.image_size,
        train=True,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )
    val_dataset = RealOldPhotoCrackDataset(
        data_root,
        split_dir / "val.txt",
        image_size=args.image_size,
        train=False,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, init_payload = load_model_from_checkpoint(init_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    config_snapshot = make_config_snapshot(args, model.get_config(), len(train_dataset), len(val_dataset))
    dump_yaml(checkpoints_root / "config_snapshot.yaml", config_snapshot)
    dump_yaml(experiments_root / "config_snapshot.yaml", config_snapshot)
    (experiments_root / "notes.md").write_text(make_notes(args, len(train_dataset), len(val_dataset)), encoding="utf-8")

    run_metadata = {
        "run_id": args.run_id,
        "stage": "segmentation_real_finetune",
        "created_at": now_iso(),
        "git_commit": get_git_commit(),
        "status": "running",
        "device": str(device),
        "data_root": str(data_root),
        "split_dir": str(split_dir),
        "image_dir": args.image_dir,
        "mask_dir": args.mask_dir,
        "init_checkpoint": str(init_checkpoint),
        "init_epoch": init_payload.get("epoch"),
        "init_metrics": init_payload.get("metrics", {}),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "best_metric": "",
        "checkpoint_path": "",
    }
    dump_json(checkpoints_root / "run_metadata.json", run_metadata)
    dump_json(experiments_root / "run_metadata.json", run_metadata)

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda_device: {torch.cuda.get_device_name(device)}")
    print(f"train_size: {len(train_dataset)}")
    print(f"val_size: {len(val_dataset)}")
    print(f"checkpoint_root: {checkpoints_root}")
    print(f"experiment_root: {experiments_root}")

    history: list[dict[str, Any]] = []
    best_iou = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    epochs_without_improvement = 0
    early_stop_triggered = False

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args)
        val_metrics = evaluate_one_epoch(model, val_loader, device, args)

        improved = val_metrics["val_iou"] > best_iou + args.min_delta
        if improved:
            best_iou = val_metrics["val_iou"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_metrics = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            save_checkpoint(checkpoints_root / "best_iou.ckpt", model, optimizer, epoch, best_metrics)
        else:
            epochs_without_improvement += 1

        current_metrics = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        save_checkpoint(checkpoints_root / "last.ckpt", model, optimizer, epoch, current_metrics)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_iou": round(val_metrics["val_iou"], 6),
            "val_f1": round(val_metrics["val_f1"], 6),
            "val_precision": round(val_metrics["val_precision"], 6),
            "val_recall": round(val_metrics["val_recall"], 6),
            "improved": int(improved),
            "epochs_without_improvement": epochs_without_improvement,
            "early_stop_triggered": 0,
        }
        history.append(row)
        write_metrics_csv(experiments_root / "metrics.csv", history)

        print(
            f"epoch={epoch} "
            f"train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} "
            f"val_iou={row['val_iou']:.6f} "
            f"val_f1={row['val_f1']:.6f} "
            f"val_precision={row['val_precision']:.6f} "
            f"val_recall={row['val_recall']:.6f} "
            f"improved={bool(improved)}"
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            early_stop_triggered = True
            history[-1]["early_stop_triggered"] = 1
            write_metrics_csv(experiments_root / "metrics.csv", history)
            print(f"early_stopping: epoch={epoch} best_epoch={best_epoch} best_val_iou={best_iou:.6f}")
            break

    if best_metrics is None:
        raise RuntimeError("Không có best metrics sau training.")

    metrics_summary = {
        "run_id": args.run_id,
        "device": str(device),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "epochs_requested": args.epochs,
        "epochs_executed": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
        "early_stopping": {
            "monitor": "val_iou",
            "patience": args.patience,
            "min_delta": args.min_delta,
            "triggered": early_stop_triggered,
            "epochs_without_improvement": epochs_without_improvement,
        },
    }
    dump_json(checkpoints_root / "metrics.json", metrics_summary)

    run_metadata.update(
        {
            "status": "completed",
            "best_metric": f"val_iou={best_metrics['val_iou']:.6f}",
            "checkpoint_path": str(checkpoints_root / "best_iou.ckpt"),
            "best_epoch": best_epoch,
            "epochs_executed": history[-1]["epoch"],
            "early_stop_triggered": early_stop_triggered,
        }
    )
    dump_json(checkpoints_root / "run_metadata.json", run_metadata)
    dump_json(experiments_root / "run_metadata.json", run_metadata)

    print(f"best_epoch: {best_epoch}")
    print(f"best_val_iou: {best_metrics['val_iou']:.6f}")
    print(f"best_checkpoint: {checkpoints_root / 'best_iou.ckpt'}")
    print(f"last_checkpoint: {checkpoints_root / 'last.ckpt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
