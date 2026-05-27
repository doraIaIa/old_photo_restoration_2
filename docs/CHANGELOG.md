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
