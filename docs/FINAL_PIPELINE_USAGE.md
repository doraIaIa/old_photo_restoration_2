# Final Pipeline Usage

## Mục tiêu

Old photo restoration dùng automatic crack/repair mask generation, inpainting và tùy chọn face restoration dependency-gated.

## Kiến trúc hiện tại

Input image
→ r011 repair-mask segmentation
→ optional CV union
→ optional Module 1.5 mask refinement
→ inpainting backend
→ optional Module 3 face restoration
→ restored output

## Checkpoint khuyến nghị

`checkpoints\segmenter\seg-unet-attn-r011-repair-ft-s42\best_iou.ckpt`

## Modes

`auto_r011`: fully automatic DL repair mask.

`auto_r011_union`: DL + CV `notebook_v7_candidate`.

`auto_r011_refined`: DL + Module 1.5 refinement.

`auto_r011_union_refined`: DL + CV + Module 1.5.

`auto_r011_union_refined_face_auto`: `auto_r011_union_refined` + Module 3 face auto nếu dependency/adapter sẵn sàng.

`external`: manual/external mask upper bound, không phải automatic.

`external_face_auto`: `external` + Module 3 face auto nếu dependency/adapter sẵn sàng.

## Face Restoration

Module 3 được bọc qua `src.restoration.face_restoration.apply_face_restoration`.

Các lựa chọn:

- `--face-mode off`: tắt face restoration.
- `--face-mode auto`: tự kiểm tra dependency CodeFormer/GFPGAN/facexlib; nếu chưa đủ điều kiện thì giữ nguyên ảnh và ghi metadata.
- `--face-mode codeformer_if_available`: chỉ dùng khi CodeFormer đã được cấu hình adapter ổn định.

Wrapper hiện không tự cài package, không copy model, không giả lập kết quả face restoration. Khi dependency hoặc adapter chưa sẵn sàng, output `restored_final.png` giữ nguyên từ `restored_before_face.png`, còn `metadata.json` ghi `face_reason`.

## Chạy một ảnh

```powershell
python scripts\run_restoration_pipeline.py ^
  --image data\demo_inputs\real_manual_3\demo3.png ^
  --mode auto_r011_union_refined ^
  --output-dir outputs\blueprint21_final_assets\single_demo ^
  --checkpoint checkpoints\segmenter\seg-unet-attn-r011-repair-ft-s42\best_iou.ckpt ^
  --face-mode off
```

## Chạy face-auto variant

```powershell
python scripts\run_restoration_pipeline.py ^
  --image data\demo_inputs\real_manual_3\demo3.png ^
  --mode auto_r011_union_refined_face_auto ^
  --output-dir outputs\blueprint21_final_assets\single_demo ^
  --checkpoint checkpoints\segmenter\seg-unet-attn-r011-repair-ft-s42\best_iou.ckpt ^
  --face-mode auto
```

## Chạy batch demo

```powershell
python scripts\run_final_pipeline_suite.py ^
  --demo-dir data\demo_inputs\real_manual_3 ^
  --external-mask-dir data\demo_masks\real_manual_3 ^
  --output-root outputs\blueprint21_final_assets\demo_suite ^
  --checkpoint checkpoints\segmenter\seg-unet-attn-r011-repair-ft-s42\best_iou.ckpt
```

Suite chạy các variant:

- `auto_r011`
- `auto_r011_union`
- `auto_r011_refined`
- `auto_r011_union_refined`
- `auto_r011_union_refined_face_auto`
- `external` nếu có mask tương ứng
- `external_face_auto` nếu có mask tương ứng

## Gradio demo

```powershell
python app_gradio.py
```

Nếu chưa cài Gradio, script chỉ báo `gradio not installed`. Smoke test import vẫn không bị fail:

```powershell
python -c "import app_gradio; print('gradio app import ok')"
```

## Output chính

Mỗi mode ghi vào:

`<output-root>\<image-stem>\<mode>\`

Các file quan trọng:

- `input.png`
- `final_mask.png`
- `overlay_final.png`
- `restored_before_face.png`
- `restored_final.png`
- `comparison_grid.png`
- `metadata.json`
- `face_module\face_metadata.json` nếu Module 3 được gọi

## Hạn chế hiện tại

- Mask tự động vẫn có thể bỏ sót vết nứt dài/mờ.
- CV union có thể thêm false positive.
- Inpainting chỉ xóa vùng nằm trong mask.
- Face restoration đang là dependency-gated wrapper; cần adapter CodeFormer/GFPGAN ổn định trước khi xem là bước phục hồi mặt thật sự.
- External/manual mask chỉ dùng làm ceiling/diagnosis, không phải automatic pipeline.

## Kết quả kỹ thuật chính

- r009 real test: IoU `0.002222`, F1 `0.004434`
- r010 real test: IoU `0.292728`, F1 `0.452884`
- r011 repair_v1 test: IoU `0.447877`, F1 `0.618667`

## Hướng phát triển

- Review visual để chọn automatic variant cuối.
- Nếu chọn face restoration, pin dependency và thêm adapter inference xác định cho CodeFormer/GFPGAN.
- Chỉ train r012 sau khi có manual repair masks được audit, không nằm trong Blueprint 2.1 hiện tại.
