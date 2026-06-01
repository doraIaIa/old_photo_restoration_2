# Blueprint 2.1 Implementation Status

## Phạm vi

Blueprint 2.1 hoàn thiện packaging/demo/report assets cho pipeline hiện có. Không train model mới, không fine-tune LaMa, không copy dataset, checkpoint hoặc output image vào Git.

## Dataset

- Real dataset: `F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq`
- Số pair: 60
- Split: train 42, val 9, test 9

## Module trạng thái

| Module | Trạng thái | Ghi chú |
|---|---|---|
| Module 1 segmentation r011 | Implemented | Dùng checkpoint `seg-unet-attn-r011-repair-ft-s42\best_iou.ckpt`. |
| Module 1.5 mask refinement | Implemented | Mode chính: `repair_v3_conservative`. |
| Module 2 inpainting | Implemented | Gọi qua `scripts/run_demo.py`, backend auto/simple_lama/opencv. |
| Module 3 face restoration | Dependency-gated wrapper | Không tự cài CodeFormer/GFPGAN; nếu thiếu dependency hoặc adapter thì giữ nguyên ảnh và ghi metadata. |
| Gradio demo | Implemented minimal | `app_gradio.py`, import không fail nếu thiếu Gradio. |
| Report assets builder | Implemented | `scripts/build_blueprint21_final_report_assets.py`. |

## Pipeline variants

- `auto_r011`
- `auto_r011_union`
- `auto_r011_refined`
- `auto_r011_union_refined`
- `auto_r011_union_refined_face_auto`
- `external` nếu có mask
- `external_face_auto` nếu có mask

## Metric mốc

| Experiment | Split | IoU | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|
| r009 synthetic-only | real test | 0.002222 | 0.004434 | 0.083464 | 0.002277 |
| r010 real-ft | real test | 0.292728 | 0.452884 | 0.509123 | 0.407834 |
| r011 repair-ft | repair_v1 test | 0.447877 | 0.618667 | 0.613738 | 0.623676 |
| r011 repair-ft | thin GT test | 0.371838 | 0.542102 | 0.493531 | 0.601276 |

## Kết luận

Bottleneck chính là mask generation. Fine-tune real-domain cải thiện mạnh so với r009 synthetic-only, còn các external/manual mask chỉ nên dùng như ceiling hoặc diagnosis.
