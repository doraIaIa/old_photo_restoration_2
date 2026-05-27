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

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.dataset import CrackSegDataset
from src.data.transforms import VALID_AUG_PROFILES, get_segmentation_transforms
from src.losses.segmentation import bce_dice_loss
from src.models.segmenter import CrackSegmenter
from src.utils.metrics import binary_f1, binary_iou, binary_precision, binary_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrypoint huấn luyện segmentation.")
    parser.add_argument("--config", default="configs/data.yaml", help="Đường dẫn file cấu hình YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Chạy một batch forward/backward để kiểm tra pipeline.")
    parser.add_argument("--smoke-run", action="store_true", help="Chạy smoke training ngắn có kiểm soát.")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch cho smoke-run.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size cho DataLoader.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--run-id", default="seg-unet-attn-r001-s42", help="Run ID theo quy ước project.")
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda.")
    parser.add_argument("--num-workers", type=int, default=0, help="Số worker cho DataLoader.")
    parser.add_argument("--bce-weight", type=float, default=0.5, help="Trọng số BCE trong loss.")
    parser.add_argument("--dice-weight", type=float, default=0.5, help="Trọng số Dice trong loss.")
    parser.add_argument("--base-channels", type=int, default=8, help="Số channel gốc của U-Net skeleton.")
    parser.add_argument("--aug-profile", choices=sorted(VALID_AUG_PROFILES), default="baseline", help="Profile augmentation cho train split.")
    parser.add_argument("--patience", type=int, default=8, help="Số epoch chờ cải thiện `val_iou` trước khi early stop.")
    parser.add_argument("--min-delta", type=float, default=5e-4, help="Mức cải thiện tối thiểu của `val_iou` để reset patience.")
    parser.add_argument("--overwrite-run", action="store_true", help="Cho phép ghi đè đúng run_id hiện có.")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def dump_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
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


def upsert_registry_row(path: Path, header: list[str], match_field: str, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    updated = False
    for existing in rows:
        if existing.get(match_field) == str(row[match_field]):
            for key in header:
                existing[key] = str(row.get(key, ""))
            updated = True
            break

    if not updated:
        rows.append({key: str(row.get(key, "")) for key in header})

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def load_best_segmentation_run(metric_registry_path: Path, experiment_registry_path: Path) -> dict[str, Any] | None:
    if not metric_registry_path.exists() or metric_registry_path.stat().st_size == 0:
        return None

    metric_rows: list[dict[str, Any]] = []
    with metric_registry_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("run_id") or row.get("split") != "val":
                continue
            try:
                row["iou_float"] = float(row["iou"])
                row["f1_float"] = float(row["f1"])
                row["precision_float"] = float(row["precision"])
                row["recall_float"] = float(row["recall"])
            except (TypeError, ValueError):
                continue
            metric_rows.append(row)

    if not metric_rows:
        return None

    best_metric_row = max(metric_rows, key=lambda item: item["iou_float"])
    checkpoint_path = ""
    dataset_id = ""
    notes = ""
    if experiment_registry_path.exists() and experiment_registry_path.stat().st_size > 0:
        with experiment_registry_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("run_id") == best_metric_row["run_id"]:
                    checkpoint_path = row.get("checkpoint_path", "")
                    dataset_id = row.get("dataset_id", "")
                    notes = row.get("notes", "")
                    break

    return {
        "run_id": best_metric_row["run_id"],
        "dataset_id": dataset_id,
        "val_iou": best_metric_row["iou_float"],
        "val_f1": best_metric_row["f1_float"],
        "val_precision": best_metric_row["precision_float"],
        "val_recall": best_metric_row["recall_float"],
        "checkpoint_path": checkpoint_path,
        "notes": notes,
    }


def update_best_runs_md(path: Path, run_id: str, dataset_id: str, best_metrics: dict[str, float], checkpoint_path: str) -> None:
    content = "\n".join(
        [
            "# Best Runs",
            "",
            "## Segmentation",
            f"- Run tốt nhất hiện tại: `{run_id}`",
            f"- Lý do chọn: best smoke run hiện tại trên `{dataset_id}` theo `val_iou = {best_metrics['val_iou']:.6f}`; chưa phải final training run",
            f"- Checkpoint: `{checkpoint_path}`",
            f"- Metric evidence: `val_f1 = {best_metrics['val_f1']:.6f}`, `val_precision = {best_metrics['val_precision']:.6f}`, `val_recall = {best_metrics['val_recall']:.6f}`",
            "",
            "## Restoration",
            "- Run tốt nhất hiện tại: chưa có",
            "- Lý do chọn: chưa có completed training run",
            "- Checkpoint: chưa có",
            "- Metric evidence: chưa có",
            "",
            "## Evaluation",
            "- Run tốt nhất hiện tại: chưa có",
            "- Lý do chọn: chưa có completed training run",
            "- Checkpoint: không áp dụng",
            "- Metric evidence: chưa có",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def build_dataloaders(
    config: dict[str, Any],
    batch_size: int,
    num_workers: int,
    aug_profile: str,
) -> tuple[Path, DataLoader, DataLoader]:
    dataset_root = PROJECT_ROOT / config["processed"]["root"] / config["processed"]["active_dataset"]
    image_size = int(config["build"]["image_size"])

    train_transform = get_segmentation_transforms(split="train", image_size=image_size, aug_profile=aug_profile)
    val_transform = get_segmentation_transforms(split="val", image_size=image_size, aug_profile="baseline")

    train_dataset = CrackSegDataset(dataset_root=dataset_root, split="train", transform=train_transform)
    val_dataset = CrackSegDataset(dataset_root=dataset_root, split="val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset_root, train_loader, val_loader


def make_model(base_channels: int, device: torch.device) -> CrackSegmenter:
    if base_channels <= 0:
        raise ValueError("--base-channels phải > 0.")
    return CrackSegmenter(base_channels=base_channels).to(device)


def compute_loss(logits: torch.Tensor, masks: torch.Tensor, bce_weight: float, dice_weight: float) -> torch.Tensor:
    if bce_weight < 0.0 or dice_weight < 0.0:
        raise ValueError("bce_weight và dice_weight phải không âm.")
    if bce_weight == 0.0 and dice_weight == 0.0:
        raise ValueError("Không thể đặt cả bce_weight và dice_weight bằng 0.")
    return bce_dice_loss(logits, masks, bce_weight=bce_weight, dice_weight=dice_weight)


def make_notes_text(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "# Smoke Training Notes",
            "",
            "- Đây là smoke training ngắn để kiểm tra training loop, metric, checkpoint và registry.",
            "- Không phải final training run.",
            f"- `bce_weight = {args.bce_weight}`",
            f"- `dice_weight = {args.dice_weight}`",
            f"- `base_channels = {args.base_channels}`",
            f"- `aug_profile = {args.aug_profile}`",
            f"- `patience = {args.patience}`",
            f"- `min_delta = {args.min_delta}`",
            f"- `epochs = {args.epochs}`",
            f"- `batch_size = {args.batch_size}`",
            f"- `lr = {args.lr}`",
            f"- `seed = {args.seed}`",
            "",
        ]
    )


def run_notes_value(args: argparse.Namespace) -> str:
    return (
        "controlled smoke training, not final training; "
        f"bce_weight={args.bce_weight}, dice_weight={args.dice_weight}, "
        f"base_channels={args.base_channels}, aug_profile={args.aug_profile}, "
        f"patience={args.patience}, min_delta={args.min_delta}"
    )


def run_dry_run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    dataset_root, train_loader, _ = build_dataloaders(
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_profile=args.aug_profile,
    )
    device = resolve_device(args.device)

    batch = next(iter(train_loader))
    model = make_model(args.base_channels, device)
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    model.train()
    logits = model(images)
    loss = compute_loss(logits, masks, args.bce_weight, args.dice_weight)
    loss.backward()

    print(f"device: {device}")
    print(f"dataset_root: {dataset_root}")
    print(f"batch_shape: {tuple(images.shape)}")
    print(f"logits_shape: {tuple(logits.shape)}")
    print(f"bce_weight: {args.bce_weight}")
    print(f"dice_weight: {args.dice_weight}")
    print(f"base_channels: {args.base_channels}")
    print(f"aug_profile: {args.aug_profile}")
    print(f"patience: {args.patience}")
    print(f"min_delta: {args.min_delta}")
    print(f"dry_run_loss: {float(loss.detach().cpu()):.6f}")


def train_one_epoch(
    model: CrackSegmenter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    bce_weight: float,
    dice_weight: float,
) -> float:
    model.train()
    loss_sum = 0.0
    batch_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = compute_loss(logits, masks, bce_weight, dice_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss không hữu hạn trong train loop: {float(loss.detach().cpu())}")
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.detach().cpu())
        batch_count += 1

    return loss_sum / max(batch_count, 1)


@torch.no_grad()
def evaluate(
    model: CrackSegmenter,
    loader: DataLoader,
    device: torch.device,
    bce_weight: float,
    dice_weight: float,
) -> dict[str, float]:
    model.eval()
    val_loss_sum = 0.0
    iou_sum = 0.0
    f1_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    batch_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss = compute_loss(logits, masks, bce_weight, dice_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss không hữu hạn trong validation loop: {float(loss.detach().cpu())}")

        val_loss_sum += float(loss.detach().cpu())
        iou_sum += float(binary_iou(logits, masks).detach().cpu())
        f1_sum += float(binary_f1(logits, masks).detach().cpu())
        precision_sum += float(binary_precision(logits, masks).detach().cpu())
        recall_sum += float(binary_recall(logits, masks).detach().cpu())
        batch_count += 1

    divisor = max(batch_count, 1)
    return {
        "val_loss": val_loss_sum / divisor,
        "val_iou": iou_sum / divisor,
        "val_f1": f1_sum / divisor,
        "val_precision": precision_sum / divisor,
        "val_recall": recall_sum / divisor,
    }


def make_run_payload(
    run_id: str,
    dataset_id: str,
    config_path: Path,
    config_snapshot: dict[str, Any],
    seed: int,
    git_commit: str,
    status: str,
    best_metric: str,
    checkpoint_path: str,
    notes: str,
    early_stopping: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": "segmentation",
        "created_at": now_iso(),
        "dataset_id": dataset_id,
        "seed": seed,
        "config_path": str(config_path.as_posix()),
        "config_snapshot": config_snapshot,
        "git_commit": git_commit,
        "status": status,
        "best_metric": best_metric,
        "checkpoint_path": checkpoint_path,
        "notes": notes,
        "early_stopping": early_stopping,
    }


def save_checkpoint(
    path: Path,
    model: CrackSegmenter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    ensure_parent(path)
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


def run_smoke_training(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs phải > 0 khi dùng --smoke-run.")
    if args.patience < 0:
        raise ValueError("--patience phải >= 0.")
    if args.min_delta < 0.0:
        raise ValueError("--min-delta phải >= 0.")

    set_seed(args.seed)
    dataset_root, train_loader, val_loader = build_dataloaders(
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug_profile=args.aug_profile,
    )
    dataset_id = dataset_root.name
    device = resolve_device(args.device)
    git_commit = get_git_commit(PROJECT_ROOT)

    checkpoints_root = PROJECT_ROOT / config["checkpoints"]["root"] / "segmenter" / args.run_id
    experiments_root = PROJECT_ROOT / config["experiments"]["root"] / "segmenter" / args.run_id
    prepare_run_dir(checkpoints_root, overwrite=args.overwrite_run)
    prepare_run_dir(experiments_root, overwrite=args.overwrite_run)

    config_snapshot = json.loads(json.dumps(config))
    config_snapshot["training"] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "run_id": args.run_id,
        "device": args.device,
        "num_workers": args.num_workers,
        "bce_weight": args.bce_weight,
        "dice_weight": args.dice_weight,
        "base_channels": args.base_channels,
        "aug_profile": args.aug_profile,
        "patience": args.patience,
        "min_delta": args.min_delta,
    }
    dump_yaml(checkpoints_root / "config_snapshot.yaml", config_snapshot)
    dump_yaml(experiments_root / "config_snapshot.yaml", config_snapshot)

    notes_text = make_notes_text(args)
    notes_value = run_notes_value(args)
    early_stopping_state = {
        "enabled": True,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "triggered": False,
        "stopped_epoch": None,
        "epochs_without_improvement_at_stop": 0,
    }

    run_metadata = make_run_payload(
        run_id=args.run_id,
        dataset_id=dataset_id,
        config_path=(PROJECT_ROOT / args.config).resolve(),
        config_snapshot=config_snapshot,
        seed=args.seed,
        git_commit=git_commit,
        status="running",
        best_metric="",
        checkpoint_path="",
        notes=notes_value,
        early_stopping=early_stopping_state,
    )
    dump_json(checkpoints_root / "run_metadata.json", run_metadata)
    dump_json(experiments_root / "run_metadata.json", run_metadata)
    (experiments_root / "notes.md").write_text(notes_text, encoding="utf-8")

    model = make_model(args.base_channels, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    best_iou = -1.0
    epochs_without_improvement = 0
    early_stop_triggered = False
    stopped_epoch: int | None = None

    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: CUDA không có, smoke training sẽ chạy trên CPU.")
    else:
        print(f"cuda_device: {torch.cuda.get_device_name(device)}")
    print(f"dataset_root: {dataset_root}")
    print(f"run_id: {args.run_id}")
    print(f"bce_weight: {args.bce_weight}")
    print(f"dice_weight: {args.dice_weight}")
    print(f"base_channels: {args.base_channels}")
    print(f"aug_profile: {args.aug_profile}")
    print(f"patience: {args.patience}")
    print(f"min_delta: {args.min_delta}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.bce_weight, args.dice_weight)
        val_metrics = evaluate(model, val_loader, device, args.bce_weight, args.dice_weight)
        is_improved = val_metrics["val_iou"] > (best_iou + args.min_delta)
        if is_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_iou": round(val_metrics["val_iou"], 6),
            "val_f1": round(val_metrics["val_f1"], 6),
            "val_precision": round(val_metrics["val_precision"], 6),
            "val_recall": round(val_metrics["val_recall"], 6),
            "improved": int(is_improved),
            "epochs_without_improvement": epochs_without_improvement,
            "early_stop_triggered": 0,
        }
        history.append(epoch_row)

        print(
            f"epoch={epoch} "
            f"train_loss={epoch_row['train_loss']:.6f} "
            f"val_loss={epoch_row['val_loss']:.6f} "
            f"val_iou={epoch_row['val_iou']:.6f} "
            f"val_f1={epoch_row['val_f1']:.6f} "
            f"val_precision={epoch_row['val_precision']:.6f} "
            f"val_recall={epoch_row['val_recall']:.6f}"
        )

        if is_improved:
            best_iou = val_metrics["val_iou"]
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                **val_metrics,
            }
            save_checkpoint(checkpoints_root / "best_iou.ckpt", model, optimizer, epoch, best_metrics)

        save_checkpoint(
            checkpoints_root / "last.ckpt",
            model,
            optimizer,
            epoch,
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                **val_metrics,
            },
        )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            early_stop_triggered = True
            stopped_epoch = epoch
            history[-1]["early_stop_triggered"] = 1
            print(
                f"early_stopping: triggered at epoch={epoch} "
                f"best_epoch={best_epoch} best_val_iou={best_iou:.6f} "
                f"epochs_without_improvement={epochs_without_improvement}"
            )
            break

    if best_metrics is None:
        raise RuntimeError("Smoke training kết thúc nhưng không ghi nhận được best metrics.")

    early_stopping_state = {
        "enabled": True,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "triggered": early_stop_triggered,
        "stopped_epoch": stopped_epoch or history[-1]["epoch"],
        "epochs_without_improvement_at_stop": epochs_without_improvement,
    }

    metrics_summary = {
        "run_id": args.run_id,
        "dataset_id": dataset_id,
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_executed": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "early_stopping": early_stopping_state,
        "best_metrics": best_metrics,
        "history": history,
        "loss_config": {
            "bce_weight": args.bce_weight,
            "dice_weight": args.dice_weight,
        },
        "model_config": model.get_config(),
        "augmentation": {
            "train_aug_profile": args.aug_profile,
            "val_aug_profile": "baseline",
        },
    }
    dump_json(checkpoints_root / "metrics.json", metrics_summary)
    dump_metrics_csv(experiments_root / "metrics.csv", history)

    final_checkpoint_path = checkpoints_root / "best_iou.ckpt"
    final_run_metadata = make_run_payload(
        run_id=args.run_id,
        dataset_id=dataset_id,
        config_path=(PROJECT_ROOT / args.config).resolve(),
        config_snapshot=config_snapshot,
        seed=args.seed,
        git_commit=git_commit,
        status="smoke_completed",
        best_metric=f"val_iou={best_metrics['val_iou']:.6f}",
        checkpoint_path=str(final_checkpoint_path.as_posix()),
        notes=notes_value,
        early_stopping=early_stopping_state,
    )
    dump_json(checkpoints_root / "run_metadata.json", final_run_metadata)
    dump_json(experiments_root / "run_metadata.json", final_run_metadata)

    experiment_registry_path = PROJECT_ROOT / "results" / "registry" / "experiment_registry.csv"
    upsert_registry_row(
        experiment_registry_path,
        header=[
            "run_id",
            "stage",
            "dataset_id",
            "config_path",
            "seed",
            "status",
            "best_metric",
            "checkpoint_path",
            "notes",
        ],
        match_field="run_id",
        row={
            "run_id": args.run_id,
            "stage": "segmentation",
            "dataset_id": dataset_id,
            "config_path": str((PROJECT_ROOT / args.config).resolve().as_posix()),
            "seed": args.seed,
            "status": "smoke_completed",
            "best_metric": f"val_iou={best_metrics['val_iou']:.6f}",
            "checkpoint_path": str(final_checkpoint_path.as_posix()),
            "notes": notes_value + f"; early_stop_triggered={early_stop_triggered}",
        },
    )

    metric_registry_path = PROJECT_ROOT / "results" / "registry" / "metric_registry.csv"
    upsert_registry_row(
        metric_registry_path,
        header=[
            "run_id",
            "split",
            "iou",
            "f1",
            "precision",
            "recall",
            "lpips",
            "fid",
            "psnr",
            "ssim",
            "created_at",
        ],
        match_field="run_id",
        row={
            "run_id": args.run_id,
            "split": "val",
            "iou": f"{best_metrics['val_iou']:.6f}",
            "f1": f"{best_metrics['val_f1']:.6f}",
            "precision": f"{best_metrics['val_precision']:.6f}",
            "recall": f"{best_metrics['val_recall']:.6f}",
            "lpips": "",
            "fid": "",
            "psnr": "",
            "ssim": "",
            "created_at": now_iso(),
        },
    )

    best_run = load_best_segmentation_run(metric_registry_path, experiment_registry_path)
    if best_run is not None:
        update_best_runs_md(
            PROJECT_ROOT / "results" / "registry" / "best_runs.md",
            run_id=str(best_run["run_id"]),
            dataset_id=str(best_run["dataset_id"] or dataset_id),
            best_metrics={
                "val_iou": float(best_run["val_iou"]),
                "val_f1": float(best_run["val_f1"]),
                "val_precision": float(best_run["val_precision"]),
                "val_recall": float(best_run["val_recall"]),
            },
            checkpoint_path=str(best_run["checkpoint_path"] or final_checkpoint_path.as_posix()),
        )

    print(f"best_epoch: {best_epoch}")
    print(f"best_val_iou: {best_metrics['val_iou']:.6f}")
    print(f"early_stop_triggered: {early_stop_triggered}")
    print(f"stopped_epoch: {stopped_epoch or history[-1]['epoch']}")
    print(f"checkpoint_root: {checkpoints_root}")
    print(f"experiment_root: {experiments_root}")


def main() -> None:
    args = parse_args()
    if args.dry_run and args.smoke_run:
        raise SystemExit("Chỉ chọn một trong hai mode: --dry-run hoặc --smoke-run.")
    if not args.dry_run and not args.smoke_run:
        raise SystemExit("Cần chọn --dry-run hoặc --smoke-run.")

    config = load_config(PROJECT_ROOT / args.config)
    if args.dry_run:
        run_dry_run(args, config)
        return

    run_smoke_training(args, config)


if __name__ == "__main__":
    main()
