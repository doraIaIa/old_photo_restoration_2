# R012 Repair Manual Plan

## Mục tiêu

r012 nhằm học repair mask thủ công thật thay vì chỉ học morphology từ thin masks. Mục tiêu là tạo mask phù hợp hơn cho LaMa: phủ lõi nứt, viền xám/trắng, shadow, vùng giấy gãy và rách sát viền.

## Vì sao r011 chưa đủ

r011 là current automatic baseline tốt hơn r009/r010 về metric, nhưng target `repair_v1` vẫn được sinh bằng morphology từ thin masks. Vì vậy r011 chưa học được toàn bộ hình học repair region do người annotate.

## Dữ liệu cần chuẩn bị

Trước tiên annotate 10-15 ảnh tại:

`F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq\masks_repair_manual`

Filename convention:

- `001_mask.png`
- `002_mask.png`

Mask convention:

- background = `0`
- repair region = `255`

## Điều kiện trước khi train r012

- Có đủ manual repair masks tối thiểu 10-15 ảnh.
- Audit mask pass: binary, đúng size, không inverted, ratio hợp lý.
- Visual review xác nhận mask phủ repair region, không chỉ centerline.

## Training plan

- Init từ r011: `checkpoints\segmenter\seg-unet-attn-r011-repair-ft-s42\best_iou.ckpt`
- Target: `masks_repair_manual`
- Nếu manual set nhỏ, cân nhắc mix `masks_repair_repair_v1` + `masks_repair_manual`.
- Không train r012 khi chưa có manual repair masks.

## Evaluation plan

- Evaluate r012 trên manual repair validation nếu có.
- Evaluate thêm against thin GT để kiểm tra model còn phủ được crack gốc.
- Demo so sánh r011 automatic, r012 automatic và manual upper bound.
