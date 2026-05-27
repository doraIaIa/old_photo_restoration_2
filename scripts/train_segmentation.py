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

from src.data.dataset import CrackSegDataset
from src.data.transforms import get_segmentation_transforms
from src.losses.segmentation import bce_dice_loss
from src.models.segmenter import CrackSegmenter
from src.utils.metrics import binary_f1, binary_iou, binary_precision, binary_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmentation training entrypoint.")
    parser.add_argument("--config", default="configs/data.yaml", help="Đường dẫn file cấu hình YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ chạy 1 batch forward/backward để kiểm tra pipeline.")
    parser.add_argument("--smoke-run", action="store_true", help="Chạy smoke training ngắn 3–5 epoch.")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch cho smoke-run.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size cho DataLoader.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate cho optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--run-id", default="seg-unet-attn-r001-s42", help="Run ID theo quy ước project.")
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda.")
    parser.add_argument("--num-workers", type=int, default=0, help="Số worker cho DataLoader.")
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
            reader = csv.DictReader(handle)
            rows = list(reader)

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
        reader = csv.DictReader(handle)
        for row in reader:
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
    if experiment_registry_path.exists() and experiment_registry_path.stat().st_size > 0:
        with experiment_registry_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("run_id") == best_metric_row["run_id"]:
                    checkpoint_path = row.get("checkpoint_path", "")
                    dataset_id = row.get("dataset_id", "")
                    break

    return {
        "run_id": best_metric_row["run_id"],
        "dataset_id": dataset_id,
        "val_iou": best_metric_row["iou_float"],
        "val_f1": best_metric_row["f1_float"],
        "val_precision": best_metric_row["precision_float"],
        "val_recall": best_metric_row["recall_float"],
        "checkpoint_path": checkpoint_path,
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


def build_dataloaders(config: dict[str, Any], batch_size: int, num_workers: int) -> tuple[Path, DataLoader, DataLoader]:
    dataset_root = PROJECT_ROOT / config["processed"]["root"] / config["processed"]["active_dataset"]
    image_size = int(config["build"]["image_size"])

    train_transform = get_segmentation_transforms(split="train", image_size=image_size)
    val_transform = get_segmentation_transforms(split="val", image_size=image_size)

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


def run_dry_run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    dataset_root, train_loader, _ = build_dataloaders(config, args.batch_size, args.num_workers)
    device = resolve_device(args.device)

    batch = next(iter(train_loader))
    model = CrackSegmenter().to(device)
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    model.train()
    logits = model(images)
    loss = bce_dice_loss(logits, masks)
    loss.backward()

    print(f"device: {device}")
    print(f"dataset_root: {dataset_root}")
    print(f"batch_shape: {tuple(images.shape)}")
    print(f"logits_shape: {tuple(logits.shape)}")
    print(f"dry_run_loss: {float(loss.detach().cpu()):.6f}")


def train_one_epoch(model: CrackSegmenter, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    batch_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss không hữu hạn trong train loop: {float(loss.detach().cpu())}")
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.detach().cpu())
        batch_count += 1

    return loss_sum / max(batch_count, 1)


@torch.no_grad()
def evaluate(model: CrackSegmenter, loader: DataLoader, device: torch.device) -> dict[str, float]:
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
        loss = bce_dice_loss(logits, masks)
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
    }


def save_checkpoint(path: Path, model: CrackSegmenter, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    ensure_parent(path)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def run_smoke_training(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs phải > 0 khi dùng --smoke-run.")

    set_seed(args.seed)
    dataset_root, train_loader, val_loader = build_dataloaders(config, args.batch_size, args.num_workers)
    dataset_id = dataset_root.name
    device = resolve_device(args.device)
    git_commit = get_git_commit(PROJECT_ROOT)

    checkpoints_root = PROJECT_ROOT / config["checkpoints"]["root"] / "segmenter" / args.run_id
    experiments_root = PROJECT_ROOT / config["experiments"]["root"] / "segmenter" / args.run_id
    prepare_run_dir(checkpoints_root, overwrite=args.overwrite_run)
    prepare_run_dir(experiments_root, overwrite=args.overwrite_run)

    config_snapshot = json.loads(json.dumps(config))
    dump_yaml(checkpoints_root / "config_snapshot.yaml", config_snapshot)
    dump_yaml(experiments_root / "config_snapshot.yaml", config_snapshot)

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
        notes="smoke training 3-5 epochs, not final training",
    )
    dump_json(checkpoints_root / "run_metadata.json", run_metadata)
    dump_json(experiments_root / "run_metadata.json", run_metadata)
    (experiments_root / "notes.md").write_text(
        "# Smoke Training Notes\n\n- Đây là smoke training ngắn để kiểm tra training loop, metric, checkpoint và registry.\n- Không phải final training run.\n",
        encoding="utf-8",
    )

    model = CrackSegmenter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    best_iou = -1.0

    print(f"device: {device}")
    if device.type == "cpu":
        print("WARNING: CUDA không có, smoke training sẽ chạy trên CPU.")
    else:
        print(f"cuda_device: {torch.cuda.get_device_name(device)}")
    print(f"dataset_root: {dataset_root}")
    print(f"run_id: {args.run_id}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_iou": round(val_metrics["val_iou"], 6),
            "val_f1": round(val_metrics["val_f1"], 6),
            "val_precision": round(val_metrics["val_precision"], 6),
            "val_recall": round(val_metrics["val_recall"], 6),
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

        if val_metrics["val_iou"] > best_iou:
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

    if best_metrics is None:
        raise RuntimeError("Smoke training kết thúc nhưng không ghi nhận được best metrics.")

    metrics_summary = {
        "run_id": args.run_id,
        "dataset_id": dataset_id,
        "device": str(device),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
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
        notes="smoke training 3-5 epochs, not final training",
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
            "notes": "smoke training 3-5 epochs, not final training",
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
