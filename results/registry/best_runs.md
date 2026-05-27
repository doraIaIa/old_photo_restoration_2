# Best Runs

Các kết quả dưới đây là **controlled smoke experiments** trên cùng dataset `ds-crack3d-512-n1000-v001`. Đây **không phải final model selection**.

## Segmentation Controlled Smoke Experiments

| run_id | epochs | bce_weight | dice_weight | base_channels | best_val_iou | best_val_f1 | precision | recall | best_threshold_iou | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `seg-unet-attn-r001-s42` | 5 | 0.5 | 0.5 | 8 | 0.197967 | 0.306015 | 0.442651 | 0.262256 | 0.20 | Baseline smoke run 5 epoch; under-segmentation rõ |
| `seg-unet-attn-r002-s42` | 15 | 0.5 | 0.5 | 8 | 0.321141 | 0.455175 | 0.513190 | 0.444333 | 0.65 | Best overall theo validation IoU; threshold tối ưu cao hơn 0.5 |
| `seg-unet-attn-r003-dice07-s42` | 15 | 0.3 | 0.7 | 8 | 0.319957 | 0.448927 | 0.592571 | 0.401046 | 0.20 | Dice-heavy loss tăng precision nhưng không vượt r002 theo IoU |
| `seg-unet-attn-r004-ch32-s42` | 15 | 0.5 | 0.5 | 32 | 0.306828 | 0.437265 | 0.597821 | 0.390610 | 0.20 | Tăng capacity lên 32 channels chưa cải thiện hơn r002 |

## Current Best Smoke Run

- Run tốt nhất hiện tại: `seg-unet-attn-r002-s42`
- Dataset: `ds-crack3d-512-n1000-v001`
- Lý do chọn: tốt nhất theo `best_val_iou = 0.321141` trong nhóm controlled smoke experiments hiện có
- Checkpoint: `F:/deeplearning/old_photo_restoration_2/checkpoints/segmenter/seg-unet-attn-r002-s42/best_iou.ckpt`
- Ghi chú: đây vẫn chỉ là smoke run, chưa phải final training run

## Restoration

- Chưa có run nào.

## Evaluation

- Chưa có run nào.
