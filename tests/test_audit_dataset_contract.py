from __future__ import annotations

import json
from pathlib import Path

import yaml


def _load_active_dataset_root() -> Path:
    config_path = Path("configs/data.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return Path(config["processed"]["root"]) / config["processed"]["active_dataset"]


def test_audit_report_for_active_dataset_is_valid() -> None:
    dataset_root = _load_active_dataset_root()
    audit_report_path = dataset_root / "audit" / "audit_report.json"

    assert audit_report_path.exists()
    with audit_report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    assert report["num_missing_files"] == 0
    assert report["num_empty_masks"] == 0
    assert report["num_shape_mismatch"] == 0
    assert report["num_reject_ratio"] == 0
    assert report["mean_mask_ratio"] > 0
    assert report["mean_mask_ratio"] < 0.10
