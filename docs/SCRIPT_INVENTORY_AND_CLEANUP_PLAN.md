# SCRIPT INVENTORY AND CLEANUP PLAN

## Phạm vi

- Chế độ làm việc: đọc, phân loại, đề xuất.
- Không xóa file.
- Không move/rename file.
- Không sửa code runtime.
- Không commit/push.

## Trạng thái kiểm tra

- Stable commit: `fda268a6c41aa7961e307577c561f561f3130f16`
- `python -m compileall scripts src app_gradio.py`: pass
- `git diff --check`: pass
- Tổng số `scripts/*.py`: `56`
- Tất cả `scripts/*.py` hiện đều là file tracked.

## Tóm tắt theo category

| Category | Count |
| --- | ---: |
| `CORE_RUNTIME` | 2 |
| `TRAINING_REPRODUCTION` | 3 |
| `EVALUATION_METRICS` | 13 |
| `DATASET_PREP` | 14 |
| `REPORT_ASSET_BUILDER` | 10 |
| `DIAGNOSTIC_RECOVERY` | 5 |
| `PLACEHOLDER_OR_EMPTY` | 2 |
| `LEGACY_CANDIDATE` | 7 |

## Quy ước action

- `KEEP_IN_PLACE`
- `KEEP_BUT_DOCUMENT`
- `MOVE_TO_SCRIPTS_SUBFOLDER_LATER`
- `ARCHIVE_CANDIDATE_NEEDS_APPROVAL`
- `DELETE_NOT_ALLOWED`
- `PLACEHOLDER_KEEP_OR_REMOVE_AFTER_DOC_UPDATE`

## Inventory

### A. CORE_RUNTIME

| Script | Tracked? | Size | Has main/argparse? | Referenced by docs/code? | Category | Current purpose | Still useful? | Risk if moved | Recommended action |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `run_demo.py` | Yes | 42376 | Yes / Yes | Docs + code nhiều | `CORE_RUNTIME` | Chạy suy luận mask, inpaint, overlay, comparison grid | Yes | High | `KEEP_IN_PLACE` |
| `run_restoration_pipeline.py` | Yes | 31298 | Yes / Yes | Docs + code nhiều | `CORE_RUNTIME` | Wrapper pipeline cuối, metadata, face restore, backend fallback | Yes | High | `KEEP_IN_PLACE` |

### B. TRAINING_REPRODUCTION

| Script | Tracked? | Size | Has main/argparse? | Referenced by docs/code? | Category | Current purpose | Still useful? | Risk if moved | Recommended action |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `finetune_real_segmentation.py` | Yes | 25186 | Yes / Yes | Ít reference trực tiếp | `TRAINING_REPRODUCTION` | Fine-tune segmenter trên dữ liệu thực | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `train_r013_finetune.py` | Yes | 23466 | Yes / Yes | Reference gần đây trong r013 flow | `TRAINING_REPRODUCTION` | Tái lập fine-tune r013 candidate | Yes | Medium | `KEEP_BUT_DOCUMENT` |
| `train_segmentation.py` | Yes | 32537 | Yes / Yes | Docs + tests nhiều | `TRAINING_REPRODUCTION` | Entry point train segmentation tổng quát | Yes | High | `KEEP_BUT_DOCUMENT` |

### C. EVALUATION_METRICS

| Script | Tracked? | Size | Has main/argparse? | Referenced by docs/code? | Category | Current purpose | Still useful? | Risk if moved | Recommended action |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `compare_auto_vs_manual_masks.py` | Yes | 5310 | Yes / Yes | Ít | `EVALUATION_METRICS` | So sánh mask auto với manual | Maybe | Low | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `compare_r011_r013.py` | Yes | 10478 | Yes / Yes | Docs/code gần đây | `EVALUATION_METRICS` | So sánh công bằng r011 và r013 | Yes | Medium | `KEEP_BUT_DOCUMENT` |
| `compare_segmentation_checkpoints.py` | Yes | 6770 | Yes / Yes | Ít | `EVALUATION_METRICS` | So sánh nhiều checkpoint segmentation | Maybe | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `evaluate_r013_threshold_sweep.py` | Yes | 5319 | Yes / Yes | Docs/code gần đây | `EVALUATION_METRICS` | Sweep threshold cho r013 | Yes | Low | `KEEP_BUT_DOCUMENT` |
| `evaluate_real_segmentation.py` | Yes | 15297 | Yes / Yes | Ít | `EVALUATION_METRICS` | Đánh giá segmentation trên ảnh thực | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `evaluate_segmentation_thresholds.py` | Yes | 8344 | Yes / Yes | Docs + code | `EVALUATION_METRICS` | Threshold metric sweep tổng quát | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `export_segmentation_predictions.py` | Yes | 7202 | Yes / Yes | Ít | `EVALUATION_METRICS` | Export panel dự đoán segmentation | Yes | Low | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `run_ablation_smoke.py` | Yes | 8299 | Yes / Yes | Docs + code | `EVALUATION_METRICS` | Chạy ablation nhỏ để review nhanh | Maybe | Medium | `KEEP_BUT_DOCUMENT` |
| `run_batch_review.py` | Yes | 12057 | Yes / Yes | Docs + code | `EVALUATION_METRICS` | Chạy review hàng loạt trên input thực | Yes | Medium | `KEEP_BUT_DOCUMENT` |
| `run_final_demo_bottleneck_benchmark.py` | Yes | 17365 | Yes / Yes | Ít | `EVALUATION_METRICS` | Benchmark bottleneck cho final demo | Maybe | Medium | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `run_final_pipeline_suite.py` | Yes | 9974 | Yes / Yes | Docs + code | `EVALUATION_METRICS` | Regression suite cho final pipeline | Yes | Medium | `KEEP_BUT_DOCUMENT` |
| `run_inpainting_completion_benchmark.py` | Yes | 11561 | Yes / Yes | Ít | `EVALUATION_METRICS` | Benchmark completion quality/runtime | Maybe | Medium | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `run_official_lama_final_validation.py` | Yes | 11271 | Yes / Yes | Ít | `EVALUATION_METRICS` | Validation cho official LaMa path | Yes | Medium | `KEEP_BUT_DOCUMENT` |

### D. DATASET_PREP

| Script | Tracked? | Size | Has main/argparse? | Referenced by docs/code? | Category | Current purpose | Still useful? | Risk if moved | Recommended action |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `audit_dataset.py` | Yes | 13081 | Yes / Yes | Docs + tests | `DATASET_PREP` | Audit dataset đã build | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `audit_real_old_photo_masks.py` | Yes | 16235 | Yes / Yes | Ít | `DATASET_PREP` | Audit mask cho ảnh cũ thực | Maybe | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `audit_repair_manual_masks.py` | Yes | 8844 | Yes / Yes | Ít | `DATASET_PREP` | Audit bộ manual repair masks | Maybe | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `build_dataset.py` | Yes | 22359 | Yes / Yes | Docs + tests nhiều | `DATASET_PREP` | Sinh synthetic dataset chính | Yes | High | `KEEP_BUT_DOCUMENT` |
| `build_repair_mask_dataset.py` | Yes | 7483 | Yes / Yes | Ít | `DATASET_PREP` | Build dataset cho repair-mask workflow | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `fix_manual_masks_binary_size.py` | Yes | 6628 | Yes / Yes | Ít | `DATASET_PREP` | Chuẩn hóa binary/size của manual masks | Maybe | Low | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `fix_r013_finetune_masks.py` | Yes | 20918 | Yes / Yes | Docs/code gần đây | `DATASET_PREP` | Fix mask cho bộ fine-tune r013 | Yes | Medium | `KEEP_BUT_DOCUMENT` |
| `make_manual_mask_overlays.py` | Yes | 5594 | Yes / Yes | Ít | `DATASET_PREP` | Tạo overlay kiểm tra manual masks | Maybe | Low | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `prepare_crack_bank.py` | Yes | 12787 | Yes / Yes | Ít | `DATASET_PREP` | Chuẩn bị nguồn CrackBank | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `prepare_lama_finetune_dataset_check.py` | Yes | 5240 | Yes / Yes | Docs ít | `DATASET_PREP` | Kiểm tra dataset cho LaMa fine-tune | Yes | Low | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `prepare_lama_finetune_workspace.py` | Yes | 2914 | Yes / Yes | Docs ít | `DATASET_PREP` | Tạo workspace cho LaMa fine-tune | Yes | Low | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `prepare_phase1_data.py` | Yes | 5673 | Yes / Yes | Ít | `DATASET_PREP` | Chuẩn bị dữ liệu phase 1 | Maybe | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `prepare_repair_manual_workflow.py` | Yes | 6795 | Yes / Yes | Ít | `DATASET_PREP` | Chuẩn bị cây thư mục manual repair | Maybe | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `validate_r013_finetune_set.py` | Yes | 22443 | Yes / Yes | Docs/code gần đây | `DATASET_PREP` | Validate bộ fine-tune r013 | Yes | Medium | `KEEP_BUT_DOCUMENT` |

### E. REPORT_ASSET_BUILDER

| Script | Tracked? | Size | Has main/argparse? | Referenced by docs/code? | Category | Current purpose | Still useful? | Risk if moved | Recommended action |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `build_acceleration_status_report.py` | Yes | 12792 | Yes / No | Ít | `REPORT_ASSET_BUILDER` | Tạo status report cho tăng tốc LaMa/pipeline | Maybe | Medium | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `build_blueprint21_final_assets.py` | Yes | 9463 | Yes / No | Docs + code ít | `REPORT_ASSET_BUILDER` | Dựng asset final cho blueprint21 | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `build_blueprint21_final_report_assets.py` | Yes | 9518 | Yes / Yes | Docs + code | `REPORT_ASSET_BUILDER` | Tạo asset cho báo cáo cuối blueprint21 | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `build_codeformer_activation_summary.py` | Yes | 6377 | Yes / No | Ít | `REPORT_ASSET_BUILDER` | Tóm tắt bật/tắt CodeFormer | Maybe | Low | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `build_codeformer_face_comparison_grid.py` | Yes | 8514 | Yes / No | Ít | `REPORT_ASSET_BUILDER` | Dựng grid so sánh khuôn mặt | Maybe | Low | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `build_codeformer_fidelity_sweep_summary.py` | Yes | 4507 | Yes / No | Ít | `REPORT_ASSET_BUILDER` | Tóm tắt sweep fidelity CodeFormer | Maybe | Low | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `build_demo3_sensitive_final_assets.py` | Yes | 13320 | Yes / No | Code import từ `run_demo` | Yes | `REPORT_ASSET_BUILDER` | Dựng asset final cho demo3 sensitive | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `build_final_demo_review_sheets.py` | Yes | 7034 | Yes / Yes | Docs + code ít | `REPORT_ASSET_BUILDER` | Dựng review sheets cho final demos | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `build_final_demo_shortlist_review.py` | Yes | 9651 | Yes / No | Ít | `REPORT_ASSET_BUILDER` | Shortlist review cho final demo | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `build_final_pipeline_candidate_assets.py` | Yes | 9530 | Yes / No | Docs + code ít | `REPORT_ASSET_BUILDER` | Dựng asset cho candidate pipeline | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |

### F. DIAGNOSTIC_RECOVERY

| Script | Tracked? | Size | Has main/argparse? | Referenced by docs/code? | Category | Current purpose | Still useful? | Risk if moved | Recommended action |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `check_face_restoration_dependencies.py` | Yes | 7871 | Yes / Yes | Docs + code ít | `DIAGNOSTIC_RECOVERY` | Probe dependency face restoration | Yes | Low | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `check_lama_completion_readiness.py` | Yes | 7236 | Yes / Yes | Docs + code ít | `DIAGNOSTIC_RECOVERY` | Probe readiness của official LaMa | Yes | Low | `KEEP_BUT_DOCUMENT` |
| `predict_r013_demo_masks.py` | Yes | 5799 | Yes / Yes | Docs/code gần đây | `DIAGNOSTIC_RECOVERY` | Predict mask r013 để smoke/recovery | Yes | Medium | `KEEP_BUT_DOCUMENT` |
| `run_mask_refinement_suite.py` | Yes | 8855 | Yes / Yes | Code ít | `DIAGNOSTIC_RECOVERY` | Chạy suite tinh chỉnh mask | Yes | Medium | `MOVE_TO_SCRIPTS_SUBFOLDER_LATER` |
| `run_post_commit_validation.py` | Yes | 8937 | Yes / Yes | Code ít | `DIAGNOSTIC_RECOVERY` | Validation nhanh sau thay đổi code | Yes | Medium | `KEEP_BUT_DOCUMENT` |

### G. PLACEHOLDER_OR_EMPTY

| Script | Tracked? | Size | Has main/argparse? | Referenced by docs/code? | Category | Current purpose | Still useful? | Risk if moved | Recommended action |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `infer.py` | Yes | 0 | No / No | Docs nhiều | `PLACEHOLDER_OR_EMPTY` | Placeholder cho full inference pipeline | No current runtime value | Medium | `PLACEHOLDER_KEEP_OR_REMOVE_AFTER_DOC_UPDATE` |
| `train_restoration.py` | Yes | 0 | No / No | Docs nhiều | `PLACEHOLDER_OR_EMPTY` | Placeholder cho fine-tune Module 2 | No current runtime value | Medium | `PLACEHOLDER_KEEP_OR_REMOVE_AFTER_DOC_UPDATE` |

### H. LEGACY_CANDIDATE

| Script | Tracked? | Size | Has main/argparse? | Referenced by docs/code? | Category | Current purpose | Still useful? | Risk if moved | Recommended action |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `build_demo3_oracle_diagnosis_assets.py` | Yes | 9546 | Yes / No | Ít | `LEGACY_CANDIDATE` | Build asset cho oracle diagnosis của demo3 | Low | Medium | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `build_r012_visual_review.py` | Yes | 11217 | Yes / Yes | Ít | `LEGACY_CANDIDATE` | Dựng visual review riêng cho r012 | Low | Medium | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `experiment_line_linked_mask_demo3.py` | Yes | 22272 | Yes / No | Code import từ `run_demo` | `LEGACY_CANDIDATE` | Thử nghiệm line-linked mask riêng cho demo3 | Low | High | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `fix_demo3_oracle_mask_size.py` | Yes | 14001 | Yes / Yes | Ít | `LEGACY_CANDIDATE` | Sửa kích thước oracle mask cho demo3 | Low | Medium | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `run_r012_repair_manual_suite.py` | Yes | 8281 | Yes / Yes | Code ít | `LEGACY_CANDIDATE` | Suite đánh giá nhánh r012 manual repair | Low | Medium | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `run_r012_restoration_comparison.py` | Yes | 6204 | Yes / Yes | Code ít | `LEGACY_CANDIDATE` | So sánh phục hồi r012 với baseline cũ | Low | Medium | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |
| `run_real_domain_r010_suite.py` | Yes | 23140 | Yes / Yes | Ít | `LEGACY_CANDIDATE` | Suite lịch sử cho real-domain r010 | Maybe | High | `ARCHIVE_CANDIDATE_NEEDS_APPROVAL` |

## Ghi chú đặc biệt cho file rỗng

### `scripts/infer.py`

- Hiện là file rỗng.
- Vẫn được nhắc trong:
  - `ARCHITECTURE.md`
  - `ROADMAP_AND_TASKS.md`
  - `SETUP_ENV.md`
  - `docs/CODEBASE_GRAPH_PLAN.md`
  - `docs/TESTING_STRATEGY.md`
  - `docs/SKILL_MATRIX.md`
- Kết luận:
  - Đây là placeholder/backlog.
  - Không nên xóa nếu chưa update docs.
  - Hướng an toàn hơn là hoặc:
    - biến thành stub có thông báo rõ ràng, hoặc
    - cập nhật docs để bỏ reference rồi mới cân nhắc remove.

### `scripts/train_restoration.py`

- Hiện là file rỗng.
- Vẫn được nhắc trong:
  - `ARCHITECTURE.md`
  - `ROADMAP_AND_TASKS.md`
  - `SETUP_ENV.md`
  - `docs/CODEBASE_GRAPH_PLAN.md`
  - `docs/SKILL_MATRIX.md`
- Kết luận:
  - Đây là placeholder/backlog.
  - Không nên xóa nếu chưa update docs.
  - Có thể đổi sang stub thông báo “chưa implement” hoặc dọn docs trước.

## Script đang path-sensitive, chưa nên move ngay

Các script sau đang dùng subprocess/import với path hard-code kiểu `scripts\\...` hoặc `from scripts.run_demo import ...`, nên nếu move folder sẽ vỡ command/reference:

- `app_gradio.py` gọi `scripts\\run_restoration_pipeline.py`
- `run_ablation_smoke.py`
- `run_batch_review.py`
- `run_final_demo_bottleneck_benchmark.py`
- `run_final_pipeline_suite.py`
- `run_inpainting_completion_benchmark.py`
- `run_official_lama_final_validation.py`
- `run_r012_repair_manual_suite.py`
- `run_r012_restoration_comparison.py`
- `run_real_domain_r010_suite.py`
- `build_demo3_sensitive_final_assets.py`
- `experiment_line_linked_mask_demo3.py`

Kết luận:

- Việc move `run_demo.py`, `run_restoration_pipeline.py`, `finetune_real_segmentation.py` hoặc các script import chúng cần làm theo migration nhiều bước, không nên làm cơ học.

## Top 10 script có khả năng dư cao nhất

1. `build_demo3_oracle_diagnosis_assets.py`
2. `experiment_line_linked_mask_demo3.py`
3. `fix_demo3_oracle_mask_size.py`
4. `build_r012_visual_review.py`
5. `run_r012_repair_manual_suite.py`
6. `run_r012_restoration_comparison.py`
7. `run_real_domain_r010_suite.py`
8. `build_codeformer_fidelity_sweep_summary.py`
9. `build_codeformer_face_comparison_grid.py`
10. `build_acceleration_status_report.py`

## Top 10 script không nên đụng

1. `run_restoration_pipeline.py`
2. `run_demo.py`
3. `train_segmentation.py`
4. `build_dataset.py`
5. `train_r013_finetune.py`
6. `compare_r011_r013.py`
7. `evaluate_r013_threshold_sweep.py`
8. `fix_r013_finetune_masks.py`
9. `validate_r013_finetune_set.py`
10. `check_lama_completion_readiness.py`

## Script runtime bắt buộc

- `run_restoration_pipeline.py`
- `run_demo.py`

## Script chỉ phục vụ báo cáo/asset

- `build_acceleration_status_report.py`
- `build_blueprint21_final_assets.py`
- `build_blueprint21_final_report_assets.py`
- `build_codeformer_activation_summary.py`
- `build_codeformer_face_comparison_grid.py`
- `build_codeformer_fidelity_sweep_summary.py`
- `build_demo3_sensitive_final_assets.py`
- `build_final_demo_review_sheets.py`
- `build_final_demo_shortlist_review.py`
- `build_final_pipeline_candidate_assets.py`
- `build_demo3_oracle_diagnosis_assets.py`
- `build_r012_visual_review.py`

## Script one-off / recovery / diagnostic

- `check_face_restoration_dependencies.py`
- `check_lama_completion_readiness.py`
- `predict_r013_demo_masks.py`
- `run_mask_refinement_suite.py`
- `run_post_commit_validation.py`
- `experiment_line_linked_mask_demo3.py`
- `fix_demo3_oracle_mask_size.py`

## Đề xuất cấu trúc mới

Chỉ là đề xuất, chưa tạo folder:

```text
scripts/
  runtime/
  training/
  evaluation/
  data_prep/
  report_assets/
  diagnostics/
  legacy/
```

### Mapping đề xuất

- `scripts/runtime/`
  - `run_demo.py`
  - `run_restoration_pipeline.py`

- `scripts/training/`
  - `train_segmentation.py`
  - `finetune_real_segmentation.py`
  - `train_r013_finetune.py`

- `scripts/evaluation/`
  - `compare_auto_vs_manual_masks.py`
  - `compare_r011_r013.py`
  - `compare_segmentation_checkpoints.py`
  - `evaluate_r013_threshold_sweep.py`
  - `evaluate_real_segmentation.py`
  - `evaluate_segmentation_thresholds.py`
  - `export_segmentation_predictions.py`
  - `run_ablation_smoke.py`
  - `run_batch_review.py`
  - `run_final_demo_bottleneck_benchmark.py`
  - `run_final_pipeline_suite.py`
  - `run_inpainting_completion_benchmark.py`
  - `run_official_lama_final_validation.py`

- `scripts/data_prep/`
  - `audit_dataset.py`
  - `audit_real_old_photo_masks.py`
  - `audit_repair_manual_masks.py`
  - `build_dataset.py`
  - `build_repair_mask_dataset.py`
  - `fix_manual_masks_binary_size.py`
  - `fix_r013_finetune_masks.py`
  - `make_manual_mask_overlays.py`
  - `prepare_crack_bank.py`
  - `prepare_lama_finetune_dataset_check.py`
  - `prepare_lama_finetune_workspace.py`
  - `prepare_phase1_data.py`
  - `prepare_repair_manual_workflow.py`
  - `validate_r013_finetune_set.py`

- `scripts/report_assets/`
  - `build_acceleration_status_report.py`
  - `build_blueprint21_final_assets.py`
  - `build_blueprint21_final_report_assets.py`
  - `build_codeformer_activation_summary.py`
  - `build_codeformer_face_comparison_grid.py`
  - `build_codeformer_fidelity_sweep_summary.py`
  - `build_demo3_sensitive_final_assets.py`
  - `build_final_demo_review_sheets.py`
  - `build_final_demo_shortlist_review.py`
  - `build_final_pipeline_candidate_assets.py`

- `scripts/diagnostics/`
  - `check_face_restoration_dependencies.py`
  - `check_lama_completion_readiness.py`
  - `predict_r013_demo_masks.py`
  - `run_mask_refinement_suite.py`
  - `run_post_commit_validation.py`

- `scripts/legacy/`
  - `build_demo3_oracle_diagnosis_assets.py`
  - `build_r012_visual_review.py`
  - `experiment_line_linked_mask_demo3.py`
  - `fix_demo3_oracle_mask_size.py`
  - `run_r012_repair_manual_suite.py`
  - `run_r012_restoration_comparison.py`
  - `run_real_domain_r010_suite.py`
  - `infer.py`
  - `train_restoration.py`

## Migration plan đề xuất

### Pha 1: chỉ tài liệu hóa

- Thêm section trong `README.md` hoặc `docs/FINAL_PIPELINE_USAGE.md` mô tả vai trò từng nhóm script.
- Không move file.
- Đánh dấu placeholder rõ trong docs cho `infer.py` và `train_restoration.py`.

### Pha 2: tách diagnostics và report assets

- Move nhóm ít rủi ro trước:
  - `check_*`
  - `run_post_commit_validation.py`
  - `build_codeformer_*`
  - `build_final_*`
- Cập nhật các docs:
  - `docs/BLUEPRINT21_IMPLEMENTATION_STATUS.md`
  - `docs/FINAL_PIPELINE_USAGE.md`
  - `docs/EVALUATION_PROTOCOL.md`
  - `docs/ABLATION_STUDY_PLAN.md`
  - `docs/CODEFORMER_ACTIVATION_PLAN.md`
  - `docs/LAMA_FINETUNE_ACCELERATION_PLAN.md`

### Pha 3: tách data prep và evaluation

- Move các script không bị subprocess path hard-code trước.
- Update command trong:
  - `SETUP_ENV.md`
  - `ROADMAP_AND_TASKS.md`
  - `ARCHITECTURE.md`
  - `docs/CODEBASE_GRAPH_PLAN.md`
  - `docs/TESTING_STRATEGY.md`
  - `docs/SKILL_MATRIX.md`

### Pha 4: xử lý runtime

- Chỉ move `run_demo.py` và `run_restoration_pipeline.py` khi đã:
  - thay toàn bộ subprocess path cứng,
  - thay import `from scripts.run_demo import ...`,
  - chạy lại smoke test runtime.

## Command sẽ đổi nếu move

Ví dụ:

- `python scripts/run_demo.py`
  -> `python scripts/runtime/run_demo.py`

- `python scripts/run_restoration_pipeline.py`
  -> `python scripts/runtime/run_restoration_pipeline.py`

- `python scripts/train_segmentation.py`
  -> `python scripts/training/train_segmentation.py`

- `python scripts/build_dataset.py`
  -> `python scripts/data_prep/build_dataset.py`

Rủi ro:

- Docs sẽ lệch ngay nếu đổi path mà chưa update.
- Subprocess trong code sẽ fail nếu vẫn gọi `scripts\\run_demo.py` hoặc `scripts\\run_restoration_pipeline.py`.
- Import kiểu `from scripts.run_demo import ...` sẽ vỡ nếu chưa đổi package path.

## Kết luận

1. Repo không cần cleanup mạnh ở source code lúc này; cần inventory + tài liệu hóa trước.
2. Hai file runtime thật sự bắt buộc chỉ là `run_demo.py` và `run_restoration_pipeline.py`.
3. `infer.py` và `train_restoration.py` là placeholder/backlog, không nên xóa khi docs còn tham chiếu.
4. Nhóm phù hợp nhất để dọn cấu trúc trước là:
   - `REPORT_ASSET_BUILDER`
   - `DIAGNOSTIC_RECOVERY`
5. Nhóm phù hợp để archive sau khi duyệt là:
   - `LEGACY_CANDIDATE`
6. Với runtime và các script bị path-sensitive, chỉ nên move sau khi có migration plan và test hồi quy.
