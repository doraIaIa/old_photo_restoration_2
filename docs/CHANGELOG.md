# Changelog

## v0.1.0 — Phase 1 Degradation Core
- Added `src/data/degradation.py`
- Implemented `compute_heightmap`, `compute_normal_map`, `apply_phong_illumination`
- Self-test passed

## v0.1.1 — Data Preparation
- Copied DIV2K subset: 50 train, 10 val
- Copied CrackForest subset: 20 crack images
- Moved external CrackForest repo outside project

## v0.1.2 — Project Governance Foundation
- Added storage convention, skill matrix, data audit checklist, testing strategy
- Added codebase graph plan for future repo visualization
- Standardized `configs/data.yaml` for Phase 1 artifact flow

## v0.1.3 — Registry and Decision Logs
- Added changelog and ADR structure
- Added result registry templates for dataset, experiment, metric tracking
- Added report evidence index and metadata standard for reproducibility

## v0.1.4 — Dataset Builder
- Added `scripts/build_dataset.py` for synthetic dataset generation from DIV2K + CrackBank

## v0.1.5 — Dataset Audit Rejection
- Ran `scripts/audit_dataset.py` on `ds-crack3d-512-n0200-v001`
- Dataset rejected due to excessive mask ratio
- Next action: preprocess CrackForest into RGBA crack bank

## v0.1.6 — Dataset v002 Accepted
- Built `ds-crack3d-512-n0200-v002` using processed RGBA crack bank
- Audit passed with `mean_mask_ratio = 0.008334`
- Marked dataset as accepted for segmentation smoke tests

## v0.1.7 — Segmentation Data Pipeline Smoke
- Added `tests/test_degradation.py`, `tests/test_build_dataset_contract.py`, and `tests/test_audit_dataset_contract.py`
- Added `src/data/dataset.py` and `src/data/transforms.py` for segmentation loading
- Added `src/losses/segmentation.py`, `src/utils/metrics.py`, `src/models/attention_gate.py`, and `src/models/segmenter.py`
- Added `tests/test_segmentation_dataset.py` and `tests/test_segmentation_model_smoke.py`
- Added `scripts/train_segmentation.py --dry-run` to validate one batch forward/backward without full training

## v0.1.8 — Dataset n1000 Accepted
- Built `ds-crack3d-512-n1000-v001` with 800 train samples and 200 val samples
- Audit passed with `mean_mask_ratio = 0.007063` and `num_reject_ratio = 0`
- Marked `ds-crack3d-512-n1000-v001` as accepted and switched `active_dataset` to `n1000`

## v0.1.9 — Segmentation Smoke Training
- Extended `scripts/train_segmentation.py` with `--smoke-run`, checkpoint saving, experiment artifact saving, and registry updates
- Completed smoke training run `seg-unet-attn-r001-s42` on `ds-crack3d-512-n1000-v001`
- Best smoke metric reached `val_iou = 0.197967` after 5 epochs on the current lightweight U-Net attention skeleton
