# ADR-003: LaMa Prototype Policy

## Bối cảnh

Module 2 hiện đang ở mức prototype inference, chưa fine-tune generator và chưa đưa vào pipeline restoration hoàn chỉnh. Ở giai đoạn này, mục tiêu chính là chốt policy thực dụng cho backend inpainting, cách chọn mask từ Module 1 segmentation, và quy ước dilation trước khi chuyển sang các thử nghiệm lớn hơn trên Kaggle.

Nguồn bằng chứng trực tiếp đến từ focused review `n=30` của bundle ngoài repo:

- `F:\deeplearning\_kaggle_downloads\lama-proto-r009-focused-review-n30_kaggle_bundle.zip`

Focused review này so sánh các cấu hình oracle/pred, nhiều mức threshold, và backend `simple_lama` so với `opencv`.

## Evidence

Các cấu hình chính được đối chiếu:

- `simple_lama | oracle | dilate 0`
- `simple_lama | pred | threshold 0.90 | dilate 0`
- `simple_lama | pred | threshold 0.70 | dilate 0`
- `simple_lama | pred | threshold 0.50 | dilate 0`
- `opencv | pred | threshold 0.70 | dilate 0`

Kết quả nổi bật:

- `simple_lama | pred | 0.90 | d0` đứng đầu theo các metric focused review dùng cho vùng sửa:
  - `masked_mae_oracle = 16.719551`
  - `bbox_ssim_oracle = 0.962560`
  - `masked_mae_used = 25.183271`
  - `bbox_ssim_used = 0.975841`
- `simple_lama | pred | 0.70 | d0` xếp sau nhưng vẫn là cấu hình cân bằng tốt hơn rõ rệt so với `opencv`.
- `simple_lama | pred | 0.50 | d0` không phá ảnh rõ rệt nhưng kém hơn `0.70` và `0.90` theo các metric focused.
- Các thử nghiệm trước đó với `dilate 1` và `dilate 2` cho thấy xu hướng xấu đi nhất quán: `SSIM` giảm và `MAE` tăng khi dilation tăng.
- `opencv` chỉ phù hợp làm baseline sanity check, không đủ mạnh để dùng làm backend chính.

## Decision

Chốt policy tạm thời cho Module 2 prototype như sau:

- Backend chính: `simple_lama`
- Không dùng dilation mặc định: `dilate = 0`
- Default / metric-best mode: `pred threshold = 0.90`
- Coverage / safe-removal fallback: `pred threshold = 0.70`
- Coverage/debug only: `pred threshold = 0.50`
- Baseline đối chiếu: `opencv pred threshold = 0.70`
- Oracle reference: `simple_lama oracle dilate 0`

## Policy Table

| Mục | Policy | Vai trò |
|---|---|---|
| Backend chính | `simple_lama` | Backend mặc định cho prototype Module 2 |
| Dilation mặc định | `0` | Không mở rộng mask nếu chưa có lý do rõ ràng |
| Default mode | `pred threshold 0.90` | Dùng khi cần metric tốt nhất theo focused review |
| Fallback coverage | `pred threshold 0.70` | Dùng khi ưu tiên coverage và giảm nguy cơ bỏ sót crack |
| Debug coverage | `pred threshold 0.50` | Chỉ dùng để kiểm tra over-mask / coverage |
| Baseline sanity check | `opencv pred threshold 0.70` | Mốc đối chiếu backend đơn giản |
| Oracle reference | `simple_lama oracle dilate 0` | Mốc tham chiếu để so sánh với mask dự đoán |

## Risks / Limitations

- Focused review mới dùng `n=30`, chưa đủ lớn để coi là kết luận cuối cùng cho mọi loại vết nứt.
- `pred threshold 0.90` có rủi ro bỏ sót crack mảnh, dù đang cho metric focused tốt nhất.
- `pred threshold 0.70` an toàn hơn về coverage, nhưng có thể sửa rộng hơn mức cần thiết.
- Metric toàn ảnh như `PSNR` hoặc `SSIM` không đủ để kết luận riêng cho vùng crack; chúng chỉ nên dùng phụ.
- Kết quả tốt hơn của `pred` so với `oracle` trên một số metric focused không đồng nghĩa `pred` “đúng hơn”, vì policy mask hẹp hơn có thể làm bài toán inpaint dễ hơn.

## Next Steps

1. Review thủ công các grid trọng yếu của focused review `n=30`, ưu tiên so sánh trực tiếp `0.90` với `0.70`.
2. Chạy thêm focused review với số mẫu lớn hơn, tối thiểu `n=50`, để kiểm tra độ ổn định của policy.
3. Khi chạy LaMa prototype tiếp theo:
   - dùng `simple_lama`
   - mặc định `pred threshold 0.90`, `dilate 0`
   - luôn xuất thêm nhánh đối chiếu `pred threshold 0.70`, `dilate 0`
4. Chỉ xem xét mở dilation khi có bằng chứng trực quan rõ rằng crack mảnh đang bị bỏ sót sau inpainting.
