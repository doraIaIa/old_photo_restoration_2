# LaMa Fine-tune Acceleration Plan

## Trạng thái hiện tại

Baseline Module 2 hiện tại là `simple_lama` pretrained, có OpenCV fallback. Chưa có bằng chứng repo này đã fine-tune LaMa thành công.

Không được đánh dấu `fine_tuned_lama` là completed nếu chưa có đủ:

- Checkpoint LaMa fine-tuned.
- Inference output trên demo/test image.
- Báo cáo định lượng như LPIPS/FID hoặc masked-region LPIPS.
- Metadata xác nhận backend thực sự là fine-tuned LaMa.

## Input cần chuẩn bị

- Clean images: DIV2K, Flickr hoặc nguồn clean image nội bộ.
- Masks: crack bank, repair masks hoặc manual repair masks.
- Train pairs: degraded image + mask + clean target.
- Validation/test pairs tách riêng với train.

## Nguyên tắc môi trường

- Không cài package nặng vào venv chính nếu có nguy cơ xung đột.
- Ưu tiên workspace riêng hoặc environment riêng cho LaMa.
- Không overwrite checkpoint r009/r010/r011/r012 segmentation.
- Không copy dataset lớn vào Git.

## Workspace đề xuất

Tạo bằng:

```powershell
python scripts\prepare_lama_finetune_workspace.py
```

Output mặc định:

`outputs\blueprint21_acceleration\lama_finetune_workspace`

Nội dung gồm checklist, config template và cấu trúc thư mục placeholder để gắn dữ liệu ngoài repo.

## Milestones

1. Chuẩn hóa dataset contract: `image`, `mask`, `clean_target`.
2. Sinh synthetic degraded pairs từ clean images và crack/repair mask bank.
3. Chạy inference baseline `simple_lama` trên cùng split.
4. Fine-tune LaMa trong environment riêng.
5. Đánh giá LPIPS/FID hoặc masked-region LPIPS.
6. Chỉ tích hợp vào pipeline chính khi có checkpoint, output và metadata xác nhận.
