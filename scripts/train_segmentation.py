from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmentation training entrypoint (dry-run only).")
    parser.add_argument("--config", default="configs/data.yaml", help="Đường dẫn file cấu hình YAML.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size cho dry-run.")
    parser.add_argument("--num-workers", type=int, default=0, help="Số worker cho DataLoader.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ chạy 1 batch forward/backward để kiểm tra pipeline.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit("Giai đoạn này chỉ hỗ trợ --dry-run. Huấn luyện thật chưa được bật.")

    config = load_config(args.config)
    dataset_root = PROJECT_ROOT / config["processed"]["root"] / config["processed"]["active_dataset"]
    image_size = int(config["build"]["image_size"])

    transform = get_segmentation_transforms(split="train", image_size=image_size)
    dataset = CrackSegDataset(dataset_root=dataset_root, split="train", transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    batch = next(iter(dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CrackSegmenter().to(device)
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    model.train()
    logits = model(images)
    loss = bce_dice_loss(logits, masks)
    loss.backward()

    print(f"dataset_root: {dataset_root}")
    print(f"batch_shape: {tuple(images.shape)}")
    print(f"logits_shape: {tuple(logits.shape)}")
    print(f"dry_run_loss: {float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
