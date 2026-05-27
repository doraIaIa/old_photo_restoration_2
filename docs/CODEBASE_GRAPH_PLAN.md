# Kế hoạch codebase graph

## Mục tiêu

- Giúp nhóm hiểu file nào phụ thuộc file nào
- Giúp AI coding assistant không sửa nhầm file
- Giúp cấu trúc repo rõ ràng hơn cho báo cáo và demo
- Chưa cần graph nghiên cứu phức tạp

## Phạm vi của tài liệu này

Đây là plan, không phải task implement.

- Chưa implement `scripts/visualize_repo_graph.py`
- Chưa sinh `docs/CODEBASE_GRAPH.md`
- Chưa sinh `outputs/figures/repo_graph.png`

## 1. Graph logic cần có sau khi repo đủ file

### Pipeline build dataset

```text
scripts/build_dataset.py
  -> src/data/degradation.py
  -> configs/data.yaml
```

Ý nghĩa:

- `build_dataset.py` là entry point
- `degradation.py` chứa logic core của Phase 1
- `configs/data.yaml` là nguồn path và thông số build

### Pipeline train segmentation

```text
scripts/train_segmentation.py
  -> src/data/dataset.py
  -> src/models/segmenter.py
  -> src/losses/segmentation.py
  -> src/utils/metrics.py
```

Ý nghĩa:

- Script train phụ thuộc dataset loader
- Dataset feed vào model
- Model dùng loss segmentation
- Metrics phục vụ validation và logging

### Pipeline train restoration

```text
scripts/train_restoration.py
  -> src/models/lama_gan.py
  -> src/losses/restoration.py
  -> src/utils/checkpoint.py
```

Ý nghĩa:

- Restoration training tách riêng khỏi segmentation training
- Checkpoint helper là dependency quản lý lưu/đọc model

### Pipeline inference

```text
scripts/infer.py
  -> Module 1
  -> Module 2
  -> Module 3
  -> outputs/inference/<run_id>/
```

Ý nghĩa:

- Inference full pipeline phải thể hiện rõ đường đi từ segmentation sang restoration rồi face restoration
- Output cuối phải đi đúng thư mục theo `run_id`

## 2. Vì sao cần graph

### Cho nhóm phát triển

- Biết file nào là entry point
- Biết file nào là thư viện lõi
- Giảm sửa nhầm file không liên quan

### Cho AI coding assistant

- Xác định nhanh vùng ảnh hưởng trước khi sửa
- Tránh đụng nhầm Phase 2 khi đang làm Phase 1
- Giúp reasoning theo module thay vì đoán mò cả repo

### Cho báo cáo và demo

- Dễ minh họa kiến trúc repo
- Dễ giải thích luồng dữ liệu từ raw input đến output

## 3. Cấp độ graph nên có

Ưu tiên theo thứ tự:

1. Graph mức file
2. Graph mức module
3. Graph mức script entry point

Chưa cần ở giai đoạn này:

- Graph call stack chi tiết từng hàm
- Graph AST hoặc static analysis phức tạp
- Graph nghiên cứu kiểu dependency mining

## 4. Đề xuất file sau này

- `scripts/visualize_repo_graph.py`
- `docs/CODEBASE_GRAPH.md`
- `outputs/figures/repo_graph.png`

## 5. Đề xuất cách làm sau này

### Bước 1

Liệt kê các script entry point:

- `scripts/build_dataset.py`
- `scripts/train_segmentation.py`
- `scripts/train_restoration.py`
- `scripts/infer.py`

### Bước 2

Map import chính của từng script về:

- `src/data/`
- `src/models/`
- `src/losses/`
- `src/utils/`
- `configs/`

### Bước 3

Sinh tài liệu graph ở dạng:

- sơ đồ text
- mermaid graph
- hoặc ảnh `repo_graph.png`

### Bước 4

Đưa graph tóm tắt vào tài liệu:

- `docs/CODEBASE_GRAPH.md`

## 6. Quy tắc khi làm graph sau này

- Chỉ vẽ dependency thực sự tồn tại trong code
- Không vẽ dependency tưởng tượng
- Ưu tiên rõ ràng hơn là đẹp
- Cập nhật graph khi thêm script hoặc module lớn

## 7. Áp dụng cho Phase 1 hiện tại

Ở thời điểm hiện tại:

- File lõi đang có thật là [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py)
- Config lõi đang có thật là [configs/data.yaml](F:/deeplearning/old_photo_restoration_2/configs/data.yaml)
- `scripts/build_dataset.py` chưa được viết xong nên graph hiện chỉ là plan

Việc tiếp theo sau tài liệu này là:

- Viết `scripts/build_dataset.py` bám đúng dependency tối thiểu:
  - config từ `configs/data.yaml`
  - logic degrade từ `src/data/degradation.py`
  - output theo `docs/STORAGE_CONVENTIONS.md`
