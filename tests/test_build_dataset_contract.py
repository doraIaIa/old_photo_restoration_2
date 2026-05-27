from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import yaml


def _load_active_dataset_root() -> Path:
    config_path = Path("configs/data.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return Path(config["processed"]["root"]) / config["processed"]["active_dataset"]


def _count_images(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png")


def test_dataset_contract_for_active_dataset() -> None:
    dataset_root = _load_active_dataset_root()

    assert dataset_root.exists()
    for relative_dir in (
        "train/images",
        "train/masks",
        "train/gt",
        "val/images",
        "val/masks",
        "val/gt",
    ):
        assert (dataset_root / relative_dir).exists()

    manifest_path = dataset_root / "manifest.csv"
    stats_path = dataset_root / "stats.json"
    metadata_path = dataset_root / "dataset_metadata.json"
    config_snapshot_path = dataset_root / "config_snapshot.yaml"

    assert manifest_path.exists()
    assert stats_path.exists()
    assert metadata_path.exists()
    assert config_snapshot_path.exists()

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows

    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    assert stats["dataset_id"] == dataset_root.name

    train_images = _count_images(dataset_root / "train/images")
    train_masks = _count_images(dataset_root / "train/masks")
    train_gt = _count_images(dataset_root / "train/gt")
    val_images = _count_images(dataset_root / "val/images")
    val_masks = _count_images(dataset_root / "val/masks")
    val_gt = _count_images(dataset_root / "val/gt")

    assert train_images == train_masks == train_gt
    assert val_images == val_masks == val_gt

    sample_mask_path = dataset_root / rows[0]["mask_path"]
    sample_mask = cv2.imread(str(sample_mask_path), cv2.IMREAD_GRAYSCALE)
    assert sample_mask is not None
    assert int((sample_mask > 0).sum()) > 0
