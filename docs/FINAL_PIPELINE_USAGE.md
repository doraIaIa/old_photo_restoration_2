# Final Pipeline Usage

## Mục tiêu

Old photo restoration using automatic crack/repair mask generation + LaMa inpainting.

## Kiến trúc hiện tại

Input image
→ r011 repair-mask segmentation
→ optional CV union
→ optional Module 1.5 mask refinement
→ simple_lama inpainting
→ restored output

## Checkpoint khuyến nghị

`checkpoints\segmenter\seg-unet-attn-r011-repair-ft-s42\best_iou.ckpt`

## Modes

`auto_r011`: fully automatic DL repair mask.

`auto_r011_union`: DL + CV `notebook_v7_candidate`.

`auto_r011_refined`: DL + Module 1.5 refinement.

`auto_r011_union_refined`: DL + CV + Module 1.5.

`external`: manual/external mask upper bound, không phải automatic.

## Chạy một ảnh

```powershell
python scripts\run_restoration_pipeline.py ^
  --image data\demo_inputs\real_manual_3\demo3.png ^
  --mode auto_r011_union_refined ^
  --output-dir outputs\final_pipeline_assets\single_demo
```

## Chạy batch demo

```powershell
python scripts\run_final_pipeline_suite.py ^
  --demo-dir data\demo_inputs\real_manual_3 ^
  --external-mask-dir data\demo_masks\real_manual_3 ^
  --output-root outputs\final_pipeline_assets\demo_suite
```

## Hạn chế hiện tại

- Mask tự động vẫn có thể bỏ sót vết nứt dài/mờ.
- CV có thể thêm false positive.
- LaMa chỉ xóa vùng nằm trong mask.
- External/manual mask chỉ dùng làm ceiling/diagnosis.

## Kết quả kỹ thuật chính

- r009 real test: IoU `0.002222`, F1 `0.004434`
- r010 real test: IoU `0.292728`, F1 `0.452884`
- r011 repair_v1 test: IoU `0.447877`, F1 `0.618667`

## Hướng phát triển

- Tạo `masks_repair_manual`.
- Train r012 từ r011 sau khi manual repair masks audit pass.
- Nâng architecture bằng ResNet34 encoder/deep supervision nếu còn thời gian.
