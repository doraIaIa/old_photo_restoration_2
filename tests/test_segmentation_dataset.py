from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import CrackSegDataset
from src.data.transforms import get_segmentation_transforms


def _load_active_dataset_root() -> Path:
    config_path = Path("configs/data.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return Path(config["processed"]["root"]) / config["processed"]["active_dataset"]


def test_segmentation_dataset_and_dataloader_smoke() -> None:
    dataset_root = _load_active_dataset_root()
    transform = get_segmentation_transforms(split="train", image_size=512)
    dataset = CrackSegDataset(dataset_root=dataset_root, split="train", transform=transform)

    assert len(dataset) > 0

    sample = dataset[0]
    assert sample["image"].shape == (3, 512, 512)
    assert sample["mask"].shape == (1, 512, 512)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    assert float(sample["mask"].min()) >= 0.0
    assert float(sample["mask"].max()) <= 1.0

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    assert batch["image"].shape == (2, 3, 512, 512)
    assert batch["mask"].shape == (2, 1, 512, 512)


def test_segmentation_dataset_strong_augmentation_smoke() -> None:
    dataset_root = _load_active_dataset_root()
    transform = get_segmentation_transforms(split="train", image_size=512, aug_profile="strong")
    dataset = CrackSegDataset(dataset_root=dataset_root, split="train", transform=transform)

    sample = dataset[0]
    assert sample["image"].shape == (3, 512, 512)
    assert sample["mask"].shape == (1, 512, 512)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    assert float(sample["mask"].min()) >= 0.0
    assert float(sample["mask"].max()) <= 1.0
