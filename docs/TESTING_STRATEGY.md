# Chiến lược kiểm thử

## Mục tiêu

Kiểm thử theo từng lớp để chặn lỗi shape, dtype, range và lỗi artifact path trước khi đụng tới training thật. Với Phase 1 hiện tại, ưu tiên test unit và test build dataset trên data nhỏ.

## 1. Nguyên tắc chung

- Test `shape`, `dtype`, `range` trước khi test metric.
- Test trên data nhỏ trước.
- Không train full khi smoke test chưa pass.
- Mọi test tạo output phải ghi vào thư mục quy ước, không ghi ra root project.

## 2. Unit tests cho `src/data/degradation.py`

File đích:

- [tests/test_degradation.py](F:/deeplearning/old_photo_restoration_2/tests/test_degradation.py)

Các ca test bắt buộc:

- `compute_heightmap` trả đúng shape và range
- `compute_normal_map` trả vector đơn vị
- `apply_phong_illumination` giữ đúng shape và range hợp lệ
- `alpha_blend` không làm đổi shape
- `generate_degraded_pair` trả `degraded` và `mask` đúng `dtype`, `shape`, `range`

Kiểm thử nên bám các tiêu chí sau:

- Input RGB/RGBA nhỏ, dễ kiểm tra tay
- Không phụ thuộc vào file lớn
- Có thể tái lập bằng seed cố định nếu hàm có random

## 3. Dataset tests cho `build_dataset.py`

Mặc dù chưa viết `scripts/build_dataset.py` trong task này, chiến lược test cần được chốt trước.

Các ca test bắt buộc:

- `build_dataset.py` sinh đủ folder theo `dataset_id`
- Số lượng file đúng giữa `images`, `masks`, `gt`
- `manifest.csv` đọc được
- `stats.json` đọc được
- Mask không rỗng với đa số sample

Ca test tối thiểu nên kiểm:

- Build trên tập rất nhỏ, ví dụ vài sample
- Kiểm tra `train/` và `val/` đều có dữ liệu
- Kiểm tra `config_snapshot.yaml` và `dataset_metadata.json` được tạo
- Kiểm tra output đi vào `data/processed/<dataset_id>/`

## 4. Training smoke tests sau này

Các smoke test này chưa triển khai ở Phase 1 nhưng cần được chốt để tránh viết code huấn luyện mù.

### Segmentation smoke tests

- `CrackSegDataset` load được 1 sample
- `DataLoader` load được 1 batch
- `Segmenter` forward pass với dummy tensor
- Loss trả về scalar
- Checkpoint save/load được

File liên quan sau này:

- [src/data/dataset.py](F:/deeplearning/old_photo_restoration_2/src/data/dataset.py)
- [src/models/segmenter.py](F:/deeplearning/old_photo_restoration_2/src/models/segmenter.py)
- [src/losses/segmentation.py](F:/deeplearning/old_photo_restoration_2/src/losses/segmentation.py)
- [scripts/train_segmentation.py](F:/deeplearning/old_photo_restoration_2/scripts/train_segmentation.py)

## 5. Inference tests sau này

Các ca test bắt buộc:

- `infer.py` nhận được input ảnh hợp lệ
- Output nằm đúng `outputs/inference/<run_id>/`
- Không ghi output vào root project
- Không overwrite run cũ

File liên quan sau này:

- [scripts/infer.py](F:/deeplearning/old_photo_restoration_2/scripts/infer.py)
- [docs/STORAGE_CONVENTIONS.md](F:/deeplearning/old_photo_restoration_2/docs/STORAGE_CONVENTIONS.md)

## 6. Thứ tự ưu tiên test

1. Unit test cho các hàm ảnh nền tảng
2. Test audit dữ liệu nguồn
3. Test build dataset trên sample nhỏ
4. Test manifest, stats, metadata
5. Smoke test dataset loader
6. Smoke test model forward
7. Smoke test training loop
8. Smoke test inference path

## 7. Chính sách chặn tiến độ

Không nên làm bước tiếp theo nếu bước trước chưa pass:

- Nếu unit test của `degradation.py` chưa ổn, không build dataset hàng loạt
- Nếu dataset audit chưa ổn, không huấn luyện segmentation
- Nếu smoke test loader/model chưa ổn, không chạy training dài
- Nếu inference path chưa đúng quy ước output, không dựng demo

## 8. Áp dụng cho trạng thái hiện tại

Ở thời điểm hiện tại:

- Đã có `src/data/degradation.py` self-test pass
- Chưa có `scripts/build_dataset.py`
- Chưa có model Phase 2

Việc kiểm thử tiếp theo nên tập trung vào:

- Chuẩn hóa unit test hiện có cho `degradation.py`
- Thiết kế test contract cho `build_dataset.py`
- Chạy dataset audit trước khi sinh `ds-crack3d-512-n0200-v001`
