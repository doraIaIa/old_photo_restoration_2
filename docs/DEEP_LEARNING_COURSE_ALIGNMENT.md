# Deep Learning Course Alignment

Đồ án này liên quan trực tiếp đến các chủ đề chính của môn Deep Learning, nhưng cần trình bày trung thực: LaMa và CodeFormer là pretrained models, project không claim tự train hai model này từ đầu.

## Supervised Learning

Module 1 học ánh xạ từ ảnh cũ sang repair mask. Đầu vào là ảnh, đầu ra là mask nhị phân cho vùng cần sửa. Đây là bài toán supervised learning nếu có mask ground truth hoặc manual mask.

## Image Segmentation

Repair mask là pixel-wise binary prediction: mỗi pixel được dự đoán thuộc vùng hỏng hoặc vùng giữ nguyên. Metric phù hợp gồm IoU, F1, precision và recall.

## Loss và Optimization

Nếu fine-tune r011 sau này, có thể thảo luận BCE, Dice, Focal hoặc Tversky loss. BCE ổn cho phân loại nhị phân từng pixel; Dice/Focal/Tversky hữu ích khi vùng crack nhỏ và mất cân bằng lớp mạnh. Lựa chọn loss phải được kiểm chứng bằng metric và visual review.

## Transfer Learning

Project tận dụng transfer learning ở nhiều nơi:

- `r011` là checkpoint segmentation đã có, dùng làm baseline hiện tại.
- LaMa pretrained dùng cho generative inpainting vùng mask.
- CodeFormer pretrained dùng cho face restoration sau inpainting.

## Fine-Tuning

Hướng chính là fine-tune segmentation từ `r011` sang `r012/r013` khi có data mới tốt hơn. Fine-tune LaMa là future work vì cần clean target tương ứng, quy trình chuẩn bị data phức tạp hơn và rủi ro đánh giá sai cao hơn.

## Generative Inpainting

LaMa sinh nội dung thay thế trong vùng mask dựa trên ngữ cảnh xung quanh. Đây là phần generative của pipeline, nhưng chất lượng phụ thuộc mạnh vào mask: mask thiếu làm còn vết hỏng, mask dư dễ làm mất chi tiết thật.

## Face Restoration

CodeFormer xử lý khuôn mặt sau inpainting. Fidelity mặc định hiện dùng là `0.7`, cần review vì fidelity quá thấp hoặc quá cao đều có thể làm mặt giả, méo hoặc khác nhận dạng.

## Evaluation

Evaluation gồm hai tầng:

- Mask: IoU, F1, precision, recall nếu có ground truth.
- Full pipeline: visual review, phân loại `mask_error`, `inpainting_error`, `codeformer_error`, và ablation study.

## MLOps nhẹ

Project có metadata, fallback, readiness check và script tái lập kết quả. Đây là phần MLOps nhẹ giúp debug đúng nguyên nhân: backend nào được yêu cầu, backend nào chạy thật, có fallback không, mask ratio bao nhiêu và CodeFormer có áp dụng không.

## Giới hạn hiện tại

- Không claim LaMa hoặc CodeFormer được train từ đầu trong project.
- GPU smoke pass không đồng nghĩa GPU nhanh hơn CPU.
- `r012` cũ chưa được xem là cải thiện đáng kể nếu chưa có kết quả mới.
- Chưa dùng train/val/test mới khi user chưa hoàn tất bộ data mới.
