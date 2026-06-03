# Baseline Decision Draft

Tài liệu này là bản nháp quyết định baseline, không chốt thay r011. Mọi thay đổi baseline cần có metric và visual review mới.

## Baseline tạm thời

- Segmentation default: `r011` với mode `auto_r011_union_refined`.
- Sensitive mode: `auto_r011_sensitive_low_threshold`, chỉ dùng cho ảnh hỏng nặng hoặc crack mảnh khi cần tăng recall và chấp nhận rủi ro false positive.
- Inpainting default: `official_lama` nếu environment sẵn sàng.
- Fallback cuối: `opencv`.
- Face restoration: CodeFormer với fidelity mặc định `0.7` khi bật Module 3.

## Điều kiện để model mới thay r011

Model mới chỉ nên thay r011 khi có bằng chứng nhất quán:

- Mean IoU tăng đáng kể, ví dụ khoảng `+0.03` đến `+0.05` trên tập đánh giá phù hợp.
- F1 tăng rõ, không chỉ tăng ở một vài ảnh dễ.
- Recall tăng nhưng precision không sụp.
- Visual review tốt hơn trên ảnh thật, đặc biệt ở crack mảnh và vùng hỏng nặng.
- Không tăng false positive nghiêm trọng trên mặt, tóc, quần áo, cạnh vật thể hoặc texture nền.
- Metadata và fallback behavior vẫn ổn định trong pipeline đầy đủ.

## Trạng thái r012

`r012` vẫn là experimental nếu chưa có bằng chứng mới. Không dùng `r012` làm default chỉ vì checkpoint tồn tại hoặc một vài case nhìn ổn hơn.

## Ghi chú về GPU

GPU readiness hoặc GPU smoke pass chỉ chứng minh môi trường có thể chạy. Không ghi GPU nhanh hơn CPU nếu chưa có benchmark rộng, cùng input, cùng backend và cùng điều kiện đo.
