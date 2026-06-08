# Scripts Overview

Thư mục `scripts/` hiện giữ nguyên flat layout để tránh làm gãy command, subprocess path và import đang phụ thuộc đường dẫn `scripts/...`.

## CORE_RUNTIME

- `run_demo.py`
- `run_restoration_pipeline.py`

Hai script này là entry point runtime chính cho demo và pipeline phục hồi hiện tại.

## TRAINING_REPRODUCTION

- `train_segmentation.py`
- `finetune_real_segmentation.py`
- `train_r013_finetune.py`

Nhóm này dùng để tái lập train hoặc fine-tune segmentation, bao gồm candidate r013.

## DATASET_PREP

- `build_dataset.py`
- `validate_r013_finetune_set.py`
- `fix_r013_finetune_masks.py`
- `build_repair_mask_dataset.py`
- các script `audit_*`, `prepare_*`, `fix_*` khác

Nhóm này phục vụ build, kiểm tra, sửa và chuẩn hóa dataset hoặc workflow dữ liệu phụ trợ.

## EVALUATION_METRICS

- `compare_r011_r013.py`
- `evaluate_r013_threshold_sweep.py`
- `run_final_pipeline_suite.py`
- các script `compare_*`, `evaluate_*`, `run_*benchmark*`, `run_*review*` khác

Nhóm này dùng để so sánh checkpoint, sweep threshold, chạy regression suite, benchmark và review kết quả.

## REPORT_ASSET_BUILDER

- `build_blueprint*`
- `build_codeformer*`
- `build_final*`
- `build_demo*`

Nhóm này dựng ảnh, bảng, grid và asset phục vụ báo cáo, slide hoặc review thủ công.

## DIAGNOSTIC_RECOVERY

- `check_lama_completion_readiness.py`
- `check_face_restoration_dependencies.py`
- `predict_r013_demo_masks.py`
- `run_mask_refinement_suite.py`
- `run_post_commit_validation.py`

Nhóm này phục vụ readiness check, smoke test, debug và recovery-safe validation.

## LEGACY_CANDIDATE

- các script liên quan `r010`
- các script liên quan `r012`
- các script `demo3 oracle`

Nhóm này được giữ lại chủ yếu để truy vết lịch sử thí nghiệm, không phải runtime chính hiện tại.

## PLACEHOLDER

- `infer.py`
- `train_restoration.py`

Hai file này hiện là placeholder/backlog và vẫn còn được tài liệu tham chiếu.

## Important Note

Do not move scripts mechanically. Many commands, docs and imports are path-sensitive.
