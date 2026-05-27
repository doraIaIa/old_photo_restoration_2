from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CrackSegDataset
from src.data.transforms import get_segmentation_transforms
from src.models.segmenter import CrackSegmenter
from src.utils.metrics import binary_f1, binary_iou, binary_precision, binary_recall


THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá metric segmentation theo nhiều threshold.")
    parser.add_argument("--run-id", required=True, help="Run ID chứa checkpoint `best_iou.ckpt`.")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID dùng để evaluate.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Split cần evaluate.")
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda.")
    parser.add_argument("--config", default="configs/data.yaml", help="Đường dẫn config YAML.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size cho evaluation.")
    parser.add_argument("--num-workers", type=int, default=0, help="Số worker cho DataLoader.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def load_model(checkpoint_path: Path, device: torch.device) -> CrackSegmenter:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    model.eval()
    return model


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "iou", "f1", "precision", "recall"])
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    device = resolve_device(args.device)

    dataset_root = PROJECT_ROOT / config["processed"]["root"] / args.dataset_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset root: {dataset_root}")

    checkpoint_path = PROJECT_ROOT / config["checkpoints"]["root"] / "segmenter" / args.run_id / "best_iou.ckpt"
    model = load_model(checkpoint_path, device)

    image_size = int(config["build"]["image_size"])
    dataset = CrackSegDataset(
        dataset_root=dataset_root,
        split=args.split,
        transform=get_segmentation_transforms(split="val", image_size=image_size),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    threshold_sums = {
        threshold: {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "count": 0}
        for threshold in THRESHOLDS
    }

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)

        if logits.shape != masks.shape:
            raise ValueError(f"Shape mismatch giữa logits {tuple(logits.shape)} và masks {tuple(masks.shape)}")

        for threshold in THRESHOLDS:
            threshold_sums[threshold]["iou"] += float(binary_iou(logits, masks, threshold=threshold).detach().cpu())
            threshold_sums[threshold]["f1"] += float(binary_f1(logits, masks, threshold=threshold).detach().cpu())
            threshold_sums[threshold]["precision"] += float(binary_precision(logits, masks, threshold=threshold).detach().cpu())
            threshold_sums[threshold]["recall"] += float(binary_recall(logits, masks, threshold=threshold).detach().cpu())
            threshold_sums[threshold]["count"] += 1

    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        count = max(threshold_sums[threshold]["count"], 1)
        rows.append(
            {
                "threshold": f"{threshold:.2f}",
                "iou": f"{threshold_sums[threshold]['iou'] / count:.6f}",
                "f1": f"{threshold_sums[threshold]['f1'] / count:.6f}",
                "precision": f"{threshold_sums[threshold]['precision'] / count:.6f}",
                "recall": f"{threshold_sums[threshold]['recall'] / count:.6f}",
            }
        )

    output_path = PROJECT_ROOT / config["experiments"]["root"] / "segmenter" / args.run_id / "threshold_sweep.csv"
    write_csv(output_path, rows)

    best_iou_row = max(rows, key=lambda row: float(row["iou"]))
    best_f1_row = max(rows, key=lambda row: float(row["f1"]))

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset_root: {dataset_root}")
    print(f"threshold_sweep_csv: {output_path}")
    print(
        "best_by_iou: "
        f"threshold={best_iou_row['threshold']} "
        f"iou={best_iou_row['iou']} "
        f"f1={best_iou_row['f1']} "
        f"precision={best_iou_row['precision']} "
        f"recall={best_iou_row['recall']}"
    )
    print(
        "best_by_f1: "
        f"threshold={best_f1_row['threshold']} "
        f"iou={best_f1_row['iou']} "
        f"f1={best_f1_row['f1']} "
        f"precision={best_f1_row['precision']} "
        f"recall={best_f1_row['recall']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
