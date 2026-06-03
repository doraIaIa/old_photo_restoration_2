# Evaluation Protocol

Tài liệu này mô tả cách review kết quả phục hồi ảnh cũ cho Blueprint 2.1. Hiện tại protocol này dùng cho demo data cũ và batch review thủ công; chưa dùng để train/val/test bộ data mới khi bộ data đó chưa sẵn sàng.

## Mục tiêu

- Tách lỗi theo đúng module để tránh kết luận sai nguyên nhân.
- Ghi nhận backend thật sự đã chạy, fallback và metadata đi kèm.
- Không kết luận GPU nhanh hơn CPU nếu chưa có benchmark rộng.
- Không chốt baseline mới nếu chưa có review trực quan và metric đủ tin cậy.

## Phân loại lỗi

`mask_error`: mask bỏ sót vết hỏng, crack mảnh, vùng rách hoặc bắt nhầm vùng thật như mặt, tóc, quần áo, viền đồ vật.

`inpainting_error`: mask nhìn hợp lý nhưng LaMa/OpenCV lấp bệt, nhòe, sai texture, tạo mảng màu giả hoặc làm mất cấu trúc nền.

`codeformer_error`: ảnh trước CodeFormer ổn hơn, nhưng sau CodeFormer mặt bị giả, méo, lệch nhận dạng, sai texture da hoặc tạo artifact mới.

Nếu một ảnh có nhiều lỗi, đánh dấu tất cả cột liên quan và ghi rõ trong `notes` lỗi nào là nguyên nhân chính.

## Cột `review_sheet`

| Cột | Ý nghĩa |
| --- | --- |
| `image_id` | ID ảnh, thường lấy từ tên file không có phần mở rộng. |
| `config_name` | Tên cấu hình review hoặc tên thí nghiệm. |
| `mask_mode` | Mode segmentation/mask trong pipeline, ví dụ `auto_r011_union_refined`. |
| `backend_requested` | Backend yêu cầu, ví dụ `official_lama`, `simple_lama`, `opencv`. |
| `backend_actual` | Backend thực tế sau fallback. |
| `fallback_applied` | Có fallback hay không. |
| `mask_ratio` | Tỷ lệ pixel mask cuối cùng. |
| `face_restoration` | `on` hoặc `off`. |
| `codeformer_fidelity` | Fidelity CodeFormer nếu bật face restoration. |
| `status` | `done`, `failed`, `skip` hoặc trạng thái review tương ứng. |
| `mask_error` | `0/1` hoặc mô tả ngắn lỗi mask. |
| `inpainting_error` | `0/1` hoặc mô tả ngắn lỗi inpainting. |
| `codeformer_error` | `0/1` hoặc mô tả ngắn lỗi CodeFormer. |
| `overall_quality_1_5` | Điểm review trực quan từ 1 đến 5. |
| `notes` | Ghi chú ngắn, ưu tiên nguyên nhân và vùng ảnh bị lỗi. |

## Cột `dataset_index`

| Cột | Ý nghĩa |
| --- | --- |
| `image_id` | ID ảnh duy nhất. |
| `filename` | Tên file ảnh. |
| `split` | `demo`, `train`, `val`, `test` hoặc để trống nếu chưa chia. |
| `source` | Nguồn ảnh. |
| `has_manual_mask` | Có manual/oracle mask hay không. |
| `mask_path` | Đường dẫn mask thủ công nếu có. |
| `has_face` | Ảnh có mặt người đáng phục hồi hay không. |
| `damage_level` | Nhẹ, vừa, nặng hoặc thang nội bộ. |
| `damage_type` | Crack, scratch, stain, missing region, blur, noise... |
| `hard_negative_type` | Vùng dễ nhầm như tóc, nếp áo, viền vật thể, họa tiết nền. |
| `notes` | Ghi chú thêm. |

## Cách dùng script

Ghi template khi chưa có data mới:

```powershell
python scripts/run_batch_review.py --write-template --output-dir outputs/batch_review_templates
```

Chạy smoke trên demo data cũ:

```powershell
python scripts/run_batch_review.py --input-dir data/demo_inputs/real_manual_3 --max-images 3 --backend official_lama --mask-modes auto_r011_union_refined,auto_r011_sensitive_low_threshold --face-restoration off
```

Runtime output nằm trong `outputs/` và không được commit.
