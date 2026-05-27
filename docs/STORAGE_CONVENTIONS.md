# Quy ước lưu trữ chính thức

## 1. Nguyên tắc chung

- Không tạo output trực tiếp ở root project.
- Không overwrite dataset, checkpoint, experiment cũ.
- Mỗi dataset phải có `dataset_id`.
- Mỗi training, evaluation, inference run phải có `run_id`.
- Không hard-code path ngoài `configs/`.
- `data/`, `checkpoints/`, `outputs/`, `experiments/`, `wandb/`, `venv/` không được commit lên git.
- `src/`, `configs/`, `docs/`, `scripts/`, `tests/` được commit bình thường.

## 2. Quy ước `dataset_id`

Format:

```text
ds-<degradation>-<image_size>-n<num_samples>-v<version>
```

Ví dụ:

```text
ds-crack3d-512-n0200-v001
ds-crack3d-512-n5000-v001
ds-crack3d-512-n10000-v002
```

Giải thích:

- `ds`: dataset
- `crack3d`: loại degradation pipeline
- `512`: image size
- `n0200`: số lượng sample
- `v001`: version dataset

Quy tắc:

- Không dùng timestamp trong tên folder dataset.
- Nếu thay đổi source data, logic build, seed, augmentation hoặc cấu trúc output thì tăng `version`.
- Không tạo dataset mới bằng cách ghi đè dataset cũ.

## 3. Quy ước `run_id`

Format:

```text
<stage>-<model>-<tag>-r<run_number>-s<seed>
```

Ví dụ:

```text
seg-unet-attn-r001-s42
lama-ft-r001-s42
eval-ablation-r001-s42
infer-demo-r001-s42
```

Giải thích:

- `stage`: giai đoạn như `seg`, `lama`, `eval`, `infer`
- `model`: model hoặc biến thể chính
- `tag`: nhãn ngắn mô tả mục tiêu run
- `r001`: số thứ tự run
- `s42`: random seed

Quy tắc:

- Không dùng timestamp trong tên folder run.
- Không ghi đè run cũ.

## 4. Ngày giờ và metadata

- Không đưa ngày giờ vào tên folder.
- Lưu ngày giờ trong metadata file:
  - `dataset_metadata.json`
  - `run_metadata.json`

Dataset metadata:

```text
data/processed/<dataset_id>/dataset_metadata.json
```

Run metadata:

```text
experiments/<stage>/<run_id>/run_metadata.json
checkpoints/<module>/<run_id>/run_metadata.json
```

### Chuẩn `run_metadata.json`

Mọi training, evaluation, inference run bắt buộc phải có các field:

- `run_id`
- `stage`
- `created_at`
- `dataset_id`
- `seed`
- `config_path`
- `config_snapshot`
- `git_commit` nếu lấy được
- `status`
- `best_metric`
- `checkpoint_path` nếu có
- `notes`

### Chuẩn `dataset_metadata.json`

Mọi dataset build ra bắt buộc phải có các field:

- `dataset_id`
- `created_at`
- `image_size`
- `num_samples`
- `num_train`
- `num_val`
- `seed`
- `clean_sources`
- `crack_sources`
- `generation_script`
- `config_snapshot`
- `status`
- `notes`

## 5. Cấu trúc dataset output

```text
data/processed/<dataset_id>/
  train/
    images/
    masks/
    gt/
  val/
    images/
    masks/
    gt/
  previews/
  manifest.csv
  stats.json
  config_snapshot.yaml
  dataset_metadata.json
```

Ý nghĩa:

- `images/`: ảnh degraded input
- `masks/`: crack mask ground truth
- `gt/`: ảnh sạch ground truth
- `previews/`: preview để kiểm tra nhanh
- `manifest.csv`: mapping từng sample
- `stats.json`: thống kê dataset
- `config_snapshot.yaml`: snapshot config lúc build
- `dataset_metadata.json`: metadata của dataset

## 6. Quy ước `manifest.csv`

Mỗi dòng là một sample.

Các cột bắt buộc:

- `sample_id`
- `split`
- `degraded_path`
- `mask_path`
- `gt_path`
- `clean_source`
- `crack_source`
- `seed`
- `image_size`
- `mask_pixels`

## 7. Quy ước `stats.json`

Phải có tối thiểu:

- `dataset_id`
- `num_samples`
- `num_train`
- `num_val`
- `image_size`
- `seed`
- `mean_mask_pixels`
- `min_mask_pixels`
- `max_mask_pixels`
- `created_at`

## 8. Cấu trúc checkpoint

```text
checkpoints/
  segmenter/<run_id>/
  lama_finetuned/<run_id>/
  pretrained/lama/
  pretrained/codeformer/
```

Mỗi thư mục checkpoint theo `run_id` nên có:

- checkpoint chính như `best_iou.ckpt` hoặc `best_lpips.ckpt`
- `last.ckpt` nếu có
- `config_snapshot.yaml`
- `run_metadata.json`
- `metrics.json`

Không lưu checkpoint trực tiếp kiểu:

```text
checkpoints/best.ckpt
checkpoints/model.pth
best.pth
last.pth
```

## 9. Cấu trúc experiment

```text
experiments/
  segmenter/<run_id>/
  restoration/<run_id>/
  evaluation/<run_id>/
```

Mỗi thư mục run trong `experiments/` phải có `run_metadata.json`.

## 10. Cấu trúc output

```text
outputs/
  debug/degradation/
  inference/<run_id>/
  figures/report/
```

Mọi file debug như `degradation_test_output.png` phải nằm trong `outputs/debug/`, không nằm ở root project.

## 11. Active dataset

Có thể dùng:

```text
data/processed/latest.txt
```

Ví dụ:

```text
ds-crack3d-512-n0200-v001
```

Tuy nhiên `configs/data.yaml` vẫn là nguồn chính để script đọc dataset đang active.

## 12. Config path

Mọi path phải đi qua `configs/data.yaml`.

Field chuẩn cho Phase 1:

```yaml
raw:
  div2k_train: "data/clean/div2k/train"
  div2k_val: "data/clean/div2k/val"
  crack_bank_raw: "data/crack_bank/raw"

processed:
  root: "data/processed"
  active_dataset: "ds-crack3d-512-n0200-v001"

build:
  degradation: "crack3d"
  image_size: 512
  num_samples: 200
  val_ratio: 0.2
  seed: 42
  version: "v001"

outputs:
  debug_root: "outputs/debug"
  inference_root: "outputs/inference"
  figures_root: "outputs/figures"

checkpoints:
  root: "checkpoints"

experiments:
  root: "experiments"
```

## 13. Quy ước git

Các mục sau phải bị ignore:

```text
data/
checkpoints/
outputs/
experiments/
wandb/
venv/
__pycache__/
*.pyc
*.ckpt
*.pth
*.pt
temp_*/
*_tmp/
CrackForest-dataset/
DeepCrack/
```

## 14. Áp dụng cho Phase 1 hiện tại

- Clean source hiện có:
  - `data/clean/div2k/train/`: 50 ảnh
  - `data/clean/div2k/val/`: 10 ảnh
- Crack source hiện có:
  - `data/crack_bank/raw/`: 20 ảnh crack
- Dataset build kế tiếp phải ghi vào:
  - `data/processed/ds-crack3d-512-n0200-v001/`
- Không viết `scripts/build_dataset.py` trong tài liệu này.
