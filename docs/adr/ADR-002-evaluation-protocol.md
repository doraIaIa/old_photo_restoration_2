# ADR-002: Evaluation Protocol

## Bối cảnh

PSNR và SSIM không đủ cho bài toán generative inpainting. Chúng hữu ích như metric tham chiếu, nhưng không nên là metric chính cho restoration.

## Quyết định

- dùng `LPIPS` và `FID` làm metric chính cho restoration
- dùng `IoU`, `F1`, `Precision`, `Recall` cho segmentation
- thêm `oracle-mask evaluation` để tách lỗi segmentation và lỗi inpainting

## Hệ quả tích cực

- đánh giá restoration gần hơn với chất lượng cảm nhận
- đánh giá segmentation bám sát chất lượng mask
- dễ tách nguồn lỗi giữa Module 1 và Module 2

## Trade-off

- pipeline evaluation phức tạp hơn
- cần lưu thêm metric và evidence
- chi phí tính toán cao hơn so với chỉ dùng PSNR/SSIM
