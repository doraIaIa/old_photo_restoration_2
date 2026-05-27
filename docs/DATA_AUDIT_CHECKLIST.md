# Checklist kiểm tra dữ liệu

## Mục tiêu

Checklist này dùng để kiểm tra dữ liệu trước và sau khi build dataset cho Phase 1. Mục tiêu là chặn lỗi sớm trước khi viết hoặc chạy `scripts/build_dataset.py`.

## 1. Trước build dataset

### Kiểm tra thư mục nguồn

- [ ] `data/clean/div2k/train/` tồn tại
- [ ] `data/clean/div2k/val/` tồn tại
- [ ] `data/crack_bank/raw/` tồn tại
- [ ] Số lượng file clean train đúng theo subset hiện tại
- [ ] Số lượng file clean val đúng theo subset hiện tại
- [ ] Có ảnh crack sẵn trong `data/crack_bank/raw/`

### Kiểm tra khả năng load ảnh

- [ ] Clean images load được bằng OpenCV
- [ ] Clean images load được bằng Pillow nếu cần kiểm tra chéo
- [ ] Crack images load được
- [ ] Không có ảnh trả về `None` khi đọc bằng OpenCV
- [ ] Không có ảnh 0 byte

### Kiểm tra định dạng và nội dung file

- [ ] Không có file lạ trong raw folders như `.txt`, `.db`, `.zip`, `.json` không liên quan
- [ ] Đuôi file ảnh nằm trong tập cho phép như `.png`, `.jpg`, `.jpeg`
- [ ] Crack có alpha channel hoặc có thể tạo alpha hợp lý từ darkness
- [ ] Không có ảnh có width hoặc height bằng 0
- [ ] Không có ảnh bị hỏng header

### Kiểm tra sơ bộ bằng mắt

- [ ] Một vài ảnh DIV2K mở được bình thường
- [ ] Một vài ảnh crack mở được bình thường
- [ ] Không có hiện tượng màu bị sai RGB/BGR ngay từ raw input
- [ ] Crack đủ nhìn thấy bằng mắt, không quá mờ

## 2. Sau build dataset

### Kiểm tra cấu trúc thư mục

- [ ] Dataset folder đúng theo `dataset_id`
- [ ] `train/images/` tồn tại
- [ ] `train/masks/` tồn tại
- [ ] `train/gt/` tồn tại
- [ ] `val/images/` tồn tại
- [ ] `val/masks/` tồn tại
- [ ] `val/gt/` tồn tại
- [ ] `previews/` tồn tại
- [ ] `manifest.csv` tồn tại
- [ ] `stats.json` tồn tại
- [ ] `config_snapshot.yaml` tồn tại
- [ ] `dataset_metadata.json` tồn tại

### Kiểm tra số lượng file

- [ ] Số ảnh trong `train/images` khớp với `train/masks`
- [ ] Số ảnh trong `train/images` khớp với `train/gt`
- [ ] Số ảnh trong `val/images` khớp với `val/masks`
- [ ] Số ảnh trong `val/images` khớp với `val/gt`
- [ ] Tổng số sample khớp với `manifest.csv`
- [ ] Tổng số sample khớp với `stats.json`

### Kiểm tra chất lượng mask

- [ ] Mask không rỗng với đa số sample
- [ ] `mask_pixels` có `min`, `mean`, `max` hợp lý
- [ ] Không có quá nhiều mask quá nhỏ
- [ ] Không có quá nhiều mask quá lớn
- [ ] Mask overlap đúng vùng crack
- [ ] Mask không bị lệch vị trí so với degraded image

### Kiểm tra degraded/gt

- [ ] `gt` là ảnh sạch đúng kích thước mục tiêu
- [ ] `degraded` là ảnh đã có crack
- [ ] `degraded` đã có global degradation nếu config yêu cầu
- [ ] `degraded`, `mask`, `gt` cùng kích thước
- [ ] 10 preview đầu nhìn đúng bằng mắt
- [ ] Không có output bị crop hoặc méo bất thường
- [ ] Không có ảnh màu sai do nhầm RGB/BGR

### Kiểm tra file mô tả dataset

- [ ] `manifest.csv` đọc được
- [ ] `manifest.csv` có đủ cột bắt buộc:
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
- [ ] `stats.json` parse được
- [ ] `stats.json` có đủ field bắt buộc:
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
- [ ] `config_snapshot.yaml` khớp config build thực tế
- [ ] `dataset_metadata.json` có `id`, `created_at`, `seed`, `config_path`, `config_snapshot`

## 3. Tiêu chí reject dataset

Reject dataset nếu có một hoặc nhiều điều sau:

- [ ] Quá nhiều mask rỗng
- [ ] Mask quá nhỏ hoặc quá lớn bất thường trên tỷ lệ mẫu đáng kể
- [ ] Màu ảnh bị sai RGB/BGR
- [ ] Crack không visible bằng mắt trong degraded output
- [ ] Mask lệch khỏi vị trí crack
- [ ] Ảnh output bị crop hoặc méo bất thường
- [ ] `manifest.csv` thiếu cột bắt buộc
- [ ] `stats.json` không hợp lệ hoặc không phản ánh đúng dữ liệu
- [ ] Thiếu `config_snapshot.yaml` hoặc `dataset_metadata.json`

## 4. Gợi ý audit tối thiểu cho Phase 1 hiện tại

Với dataset đầu tiên `ds-crack3d-512-n0200-v001`, audit tối thiểu nên gồm:

- Kiểm tra toàn bộ source raw folders
- Kiểm tra thống kê `mask_pixels`
- Mở 10 preview đầu tiên
- Mở ngẫu nhiên ít nhất 10 sample train và 5 sample val
- Đối chiếu ngược ít nhất 3 sample về `clean_source` và `crack_source` trong `manifest.csv`
