# QUARANTINE AUDIT PROPOSAL

## Phạm vi

- Chế độ làm việc: đọc, kiểm tra, phân loại.
- Không xóa file.
- Không move file.
- Không refactor code trong lượt này.
- Không commit/push.

## Trạng thái audit

- Git HEAD tại thời điểm audit: `fda268a6c41aa7961e307577c561f561f3130f16`
- `python -m compileall scripts src app_gradio.py`: pass
- `git diff --check`: pass

## A. KEEP

Giữ nguyên vì đang phục vụ runtime/demo/report chính hoặc là source tracked còn nằm trong luồng tài liệu và vận hành:

- `app_gradio.py`
- `scripts/run_restoration_pipeline.py`
- `scripts/run_demo.py`
- `src/postprocess/mask_refinement.py`
- `src/restoration/codeformer_adapter.py`
- `src/restoration/dependency_checks.py`
- `src/restoration/face_restoration.py`
- `src/restoration/official_lama_adapter.py`
- `src/models/segmenter.py`
- `src/data/*`
- `configs/*`
- `docs/FINAL_PIPELINE_USAGE.md`
- `docs/FINAL_PIPELINE_STATUS.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/REPORT_EVIDENCE.md`
- `scripts/check_lama_completion_readiness.py`
- `scripts/build_blueprint21_final_assets.py`
- `scripts/build_final_pipeline_candidate_assets.py`
- `scripts/build_final_demo_review_sheets.py`
- `scripts/build_final_demo_shortlist_review.py`
- `scripts/build_demo3_sensitive_final_assets.py`
- `scripts/compare_r011_r013.py`
- `scripts/evaluate_r013_threshold_sweep.py`
- `scripts/fix_r013_finetune_masks.py`
- `scripts/predict_r013_demo_masks.py`
- `scripts/train_r013_finetune.py`
- `scripts/validate_r013_finetune_set.py`

Ghi chú:

- `scripts/infer.py` và `scripts/train_restoration.py` hiện là file tracked rỗng, nhưng vẫn được tài liệu kiến trúc/roadmap/test strategy tham chiếu. Đây là nợ cấu trúc, không phải candidate quarantine.

## B. KEEP_BUT_IGNORE

Giữ local, không commit, không archive vội vì là artifact vận hành hoặc đầu ra cần cho recovery/demo:

- `checkpoints/`
- `outputs/r013_finetune/`
- `outputs/blueprint21_final_assets/`
- `outputs/final_pipeline_candidate/`
- `outputs/final_pipeline_assets/`
- `outputs/blueprint21_final_validation/`
- `outputs/report_assets/`
- `outputs/final_demo_benchmark/`
- `outputs/r013_hybrid_ablation_demo/`
- `data/demo_inputs/`
- `venv/`
- `F:\deeplearning\_share\old_photo_restoration_2_r013_checkpoint_package`
- `F:\deeplearning\_share\old_photo_restoration_2_r013_checkpoint_package.zip`

Lý do:

- Các path này chứa checkpoint, output review, input demo, package chia sẻ, hoặc môi trường chạy. Chúng không nên nằm trong Git nhưng cũng không nên quarantine/xóa khi chưa có kế hoạch vận hành thay thế.

## C. CANDIDATE_QUARANTINE

Các mục dưới đây phù hợp để copy sang khu archive/quarantine ở bước sau, sau khi duyệt thủ công. Không mục nào trong nhóm này là tracked source code.

| original path | reason | git status | referenced_by | risk | proposed archive path |
| --- | --- | --- | --- | --- | --- |
| `.tracked_files_snapshot.txt` | Snapshot tạm tạo để audit tracked files, không thuộc runtime/report chính | untracked | chỉ được nhắc trong `cleanup_inventory.md`, `repo_status_short.txt.tmp` | low | `F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\files\.tracked_files_snapshot.txt` |
| `cleanup_inventory.md` | Báo cáo sự cố cleanup tạm thời, không thuộc runtime chính | untracked | không có reference runtime; chỉ tự mô tả | low | `F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\files\cleanup_inventory.md` |
| `repo_tracked_files.txt.tmp` | File tạm sinh ra riêng cho audit hiện tại | untracked | không có reference runtime | low | `F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\files\repo_tracked_files.txt.tmp` |
| `repo_status_short.txt.tmp` | File tạm sinh ra riêng cho audit hiện tại | untracked | không có reference runtime | low | `F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\files\repo_status_short.txt.tmp` |
| `outputs/mask_refinement_diagnosis/` | Debug/diagnosis trung gian cho tinh chỉnh mask, rất nhiều ảnh overlay/debug, không phải output runtime chính | ignored | không thấy reference trong code/docs hiện tại | medium | `F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\files\outputs\mask_refinement_diagnosis\` |
| `outputs/r013_mask_tuning_demo3/` | Kết quả tuning riêng cho demo3, chủ yếu dùng review thủ công | ignored | không thấy reference runtime; chỉ phục vụ audit chất lượng | medium | `F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\files\outputs\r013_mask_tuning_demo3\` |
| `outputs/r013_mask_tuning_demo3_reference/` | Kết quả reference bổ sung cho tuning demo, không phải output runtime chính | ignored | không thấy reference runtime | medium | `F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\files\outputs\r013_mask_tuning_demo3_reference\` |
| `outputs/recovery_r013_smoke/` | Smoke output một ảnh để xác nhận recovery mode, không phải artifact chính thức | ignored | không thấy reference runtime | low | `F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\files\outputs\recovery_r013_smoke\` |
| `outputs/recovery_lama_readiness_20260608/` | Output readiness check cục bộ, dùng một lần để xác nhận môi trường | ignored | không thấy reference runtime | low | `F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\files\outputs\recovery_lama_readiness_20260608\` |
| `outputs/recovery_r013_demo_assets/` | Output recovery cho demo assets, có thể giữ tạm nhưng không phải core runtime | ignored | không thấy reference runtime | medium | `F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\files\outputs\recovery_r013_demo_assets\` |

## D. DO_NOT_TOUCH

Tuyệt đối không đụng trong đợt quarantine/refactor đầu tiên:

- `.git/`
- `venv/`
- `data/`
- `checkpoints/`
- `outputs/r013_finetune/`
- `outputs/blueprint21_final_assets/`
- `F:\deeplearning\external_models\lama`
- `F:\deeplearning\external_models\CodeFormer`
- `F:\deeplearning\_share\old_photo_restoration_2_r013_checkpoint_package`
- `F:\deeplearning\_share\old_photo_restoration_2_r013_checkpoint_package.zip`
- `app_gradio.py`
- `scripts/run_restoration_pipeline.py`
- `scripts/run_demo.py`
- `src/postprocess/mask_refinement.py`

## Ghi chú về script “nghi dư” nhưng chưa được kết luận quarantine

### 1. `scripts/infer.py`

- Trạng thái: tracked, file rỗng.
- Reference còn tồn tại trong:
  - `ARCHITECTURE.md`
  - `ROADMAP_AND_TASKS.md`
  - `docs/CODEBASE_GRAPH_PLAN.md`
  - `docs/TESTING_STRATEGY.md`
  - `docs/SKILL_MATRIX.md`
- Kết luận: là placeholder hoặc backlog dang dở, chưa nên quarantine.

### 2. `scripts/train_restoration.py`

- Trạng thái: tracked, file rỗng.
- Reference còn tồn tại trong:
  - `ARCHITECTURE.md`
  - `ROADMAP_AND_TASKS.md`
  - `docs/CODEBASE_GRAPH_PLAN.md`
  - `docs/SKILL_MATRIX.md`
- Kết luận: là placeholder hoặc backlog dang dở, chưa nên quarantine.

### 3. Các script build/audit/experiment theo tên chuyên biệt

Ví dụ:

- `scripts/build_demo3_oracle_diagnosis_assets.py`
- `scripts/fix_demo3_oracle_mask_size.py`
- `scripts/experiment_line_linked_mask_demo3.py`
- `scripts/run_mask_refinement_suite.py`
- `scripts/run_ablation_smoke.py`

Kết luận hiện tại:

- Tên gọi cho thấy tính chất chiến dịch hoặc diagnosis.
- Tuy nhiên đây đều là tracked source code.
- Chưa đủ cơ sở để archive/quarantine source tracked.
- Nếu muốn giảm tải repo, hướng đúng là lập kế hoạch refactor hoặc module hóa, không phải move sang archive.

## Đề xuất cấu trúc archive

Chỉ là đề xuất, chưa tạo:

`F:\deeplearning\_archive\old_photo_restoration_2\quarantine_20260608_211500\`

Nội dung dự kiến:

- `MANIFEST.csv`
- `MANIFEST.md`
- `RESTORE_INSTRUCTIONS.md`
- `restore_files.ps1`
- `files\...`

Quy tắc đề xuất:

1. Copy trước.
2. Tính và ghi SHA256 nguồn/đích.
3. Verify số lượng file, size, checksum.
4. Chỉ sau verify mới xem xét move ra khỏi repo.
5. Move là bước riêng, cần lệnh riêng và cần phê duyệt thủ công.

## Refactor audit

### Duplicate đang thấy rõ

1. Hàm đọc/ghi ảnh và chuẩn hóa ảnh lặp ở nhiều nơi:
   - `app_gradio.py`
   - `scripts/run_restoration_pipeline.py`
   - `scripts/run_demo.py`
   - `src/postprocess/mask_refinement.py`
   - `src/restoration/face_restoration.py`
   - `src/restoration/official_lama_adapter.py`

2. Mapping checkpoint/model version/mask mode bị lặp giữa:
   - `app_gradio.py`
   - `scripts/run_restoration_pipeline.py`

3. Logic readiness/backends bị phân tán giữa:
   - `app_gradio.py`
   - `src/restoration/dependency_checks.py`
   - `scripts/run_restoration_pipeline.py`

4. Logic dựng comparison grid/preview lặp giữa:
   - `scripts/run_demo.py`
   - `scripts/run_restoration_pipeline.py`

### Constant/path nên gom về config

1. `R011_CHECKPOINT`
2. `R012_CHECKPOINT`
3. `R013_CHECKPOINT`
4. Path `outputs/...` cho gradio/demo/runtime
5. Path external LaMa/CodeFormer
6. Danh sách mode và metadata đi kèm:
   - `mask_source`
   - `mask_refine`
   - `threshold`
   - `fallback_threshold`
   - `warning`
   - `model_version`

### Mode config đang lặp

- `auto_r011*`, `auto_r012*`, `auto_r013*` trong `scripts/run_restoration_pipeline.py`
- `MASK_MODE_CHOICES`, `available_mask_modes()`, `default_mask_mode()`, `checkpoint_for_mask_mode()`, `segmentation_version_for_mode()` trong `app_gradio.py`

Hiện trạng này tạo rủi ro lệch UI và runtime khi thêm mode mới.

### Function quá dài hoặc ghép nhiều trách nhiệm

- `app_gradio.py::run_restoration`
- `app_gradio.py::build_demo`
- `scripts/run_restoration_pipeline.py::main`
- `scripts/run_demo.py::build_cv_crack_mask`
- `scripts/run_demo.py::main`

### Refactor an toàn

1. Tách constants/checkpoint paths sang một module cấu hình dùng chung.
2. Tách registry cho mask modes dùng chung cho app và pipeline.
3. Tách helper I/O ảnh và helper metadata dùng chung.
4. Tách helper dựng preview/comparison grid dùng chung.
5. Tách helper readiness summary để app không tự dựng logic riêng.

### Refactor rủi ro, chưa nên làm ngay

1. Đụng vào thuật toán `build_cv_crack_mask` trong `scripts/run_demo.py`.
2. Đổi schema metadata hoặc tên file output hiện có.
3. Đổi luồng fallback/backends cho official LaMa và CodeFormer.
4. Gom mạnh các script tracked “mang tính chiến dịch” khi chưa có test hồi quy.
5. Chạm vào mode r011/r012 khi checkpoint local đang thiếu.

### Test nên chạy nếu refactor sau này

1. `python -m compileall scripts src app_gradio.py`
2. `python scripts/run_restoration_pipeline.py --help`
3. `python scripts/run_demo.py --help`
4. `python -c "import app_gradio; print('ok')"`
5. Smoke test 1 ảnh với:
   - `auto_r013_union_refined`
   - `auto_r013_union_repair_wide`
6. Kiểm tra metadata có đủ:
   - `segmentation_model_version`
   - `segmentation_checkpoint`
   - `segmentation_threshold`
   - `mask_mode`
   - `r011_checkpoint_available`
   - `r013_checkpoint_available`
7. `scripts/check_lama_completion_readiness.py`

## Kết luận ngắn

- Candidate quarantine an toàn nhất hiện tại là nhóm untracked audit files và một số output recovery/debug gần đây.
- Không có cơ sở để quarantine source tracked trong lượt này.
- Refactor có ích, nhưng chưa nên làm ngay trong cùng đợt với quarantine vì runtime hiện vừa ổn định lại sau recovery.
