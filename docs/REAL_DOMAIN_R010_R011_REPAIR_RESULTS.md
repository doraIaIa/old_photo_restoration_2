# Real Domain r010-r011 Repair Results

## Dataset

- Real dataset: `F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq`
- Total pairs: `60`
- Split: train `42`, val `9`, test `9`
- Original mask convention: background `0`, crack/scratch/fold/tear `255`

## Baseline Metrics

| Model | Target | Split | Threshold | IoU | F1 | Precision | Recall |
|---|---|---|---:|---:|---:|---:|---:|
| r009 synthetic-only | thin mask | real test | 0.10 | 0.002222 | 0.004434 | 0.083464 | 0.002277 |
| r010 real fine-tuned | thin mask | real val | 0.70 | 0.237356 | 0.383650 | 0.415410 | 0.356402 |
| r010 real fine-tuned | thin mask | real test | 0.70 | 0.292728 | 0.452884 | 0.509123 | 0.407834 |

## Problem

r010 cải thiện rõ mask generation so với r009, nhưng output demo vẫn xóa nứt chưa sạch vì model chủ yếu tạo detection mask mảnh quanh crack centerline. LaMa/simple_lama cần repair mask rộng hơn, bao gồm lõi nứt, viền xám/trắng hai bên, shadow quanh nứt, vùng giấy gãy và các đoạn đứt cần bridge.

## Module 1.5 Repair Mask

Đã thêm `src/postprocess/mask_refinement.py` với các mode:

- `none`
- `dilate1`
- `dilate2`
- `dilate3`
- `close_dilate1`
- `repair_v1`
- `repair_v2`
- `repair_v3_conservative`

`run_demo.py` đã hỗ trợ `--mask-refine`, mặc định `none` để giữ backward compatibility.

## Auto vs Manual Demo3

Kết quả so với `data\demo_masks\real_manual_3\demo3_mask.png` cho thấy auto masks còn thiếu vùng repair mask thủ công. Một số dòng chính:

| Variant | IoU vs manual | F1 | Precision | Recall | Missing ratio | Extra ratio |
|---|---:|---:|---:|---:|---:|---:|
| r010_dl_t070 | 0.026871 | 0.052335 | 0.405628 | 0.027972 | 0.103222 | 0.004353 |
| r010_union_cv_t070_dilate1 | 0.133243 | 0.235153 | 0.316775 | 0.186976 | 0.086337 | 0.042824 |
| r010_union_cv_t070_dilate2 | 0.151459 | 0.263073 | 0.313198 | 0.226778 | 0.082110 | 0.052809 |
| r010_union_cv_t050_dilate1 | 0.147939 | 0.257747 | 0.307037 | 0.222094 | 0.082608 | 0.053229 |

Chi tiết nằm tại `outputs\report_assets\repair_mask_r011_suite\analysis\demo3_auto_vs_manual`.

## Repair Dataset

Repair masks được tạo ngoài repo tại `F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq\masks_repair_*`. Không copy dataset vào repo.

| Mode | Ratio min | Ratio mean | Ratio max | Warning >0.30 |
|---|---:|---:|---:|---:|
| dilate1 | 0.017690 | 0.069094 | 0.174642 | 0 |
| dilate2 | 0.024635 | 0.092614 | 0.225176 | 0 |
| close_dilate1 | 0.018056 | 0.070161 | 0.177282 | 0 |
| repair_v1 | 0.019339 | 0.071793 | 0.192846 | 0 |
| repair_v2 | 0.027830 | 0.100301 | 0.271408 | 0 |
| repair_v3_conservative | 0.017369 | 0.065452 | 0.176313 | 0 |

`repair_v1` được chọn cho r011 vì phủ rộng hơn thin mask nhưng không quá aggressive.

## r011 Training

- Init checkpoint: `checkpoints\segmenter\seg-unet-attn-r010-real-ft-s42\best_iou.ckpt`
- Target masks: `masks_repair_repair_v1`
- Smoke train: pass, best val IoU `0.271481`
- Overfit2: pass, best threshold `0.80`, IoU `0.683282`
- Full train: đã chạy
- Run ID: `seg-unet-attn-r011-repair-ft-s42`
- Best epoch: `76`
- Best val IoU trong training: `0.337490`
- Checkpoint: `checkpoints\segmenter\seg-unet-attn-r011-repair-ft-s42\best_iou.ckpt`

## r011 Evaluation

| Model | Target | Split | Threshold | IoU | F1 | Precision | Recall |
|---|---|---|---:|---:|---:|---:|---:|
| r011 repair fine-tuned | repair_v1 | val | 0.70 | 0.361050 | 0.530547 | 0.518625 | 0.543029 |
| r011 repair fine-tuned | repair_v1 | test | 0.70 | 0.447877 | 0.618667 | 0.613738 | 0.623676 |
| r011 repair fine-tuned | thin original | test | 0.80 | 0.371838 | 0.542102 | 0.493531 | 0.601276 |

## Demo Outputs

r011 demo outputs:

- `outputs\report_assets\repair_mask_r011_suite\demo\r011_repair_dl`
- `outputs\report_assets\repair_mask_r011_suite\demo\r011_repair_union`
- `outputs\report_assets\repair_mask_r011_suite\demo\r011_repair_union_refine_v3`

Summary files:

- `outputs\report_assets\repair_mask_r011_suite\SUMMARY.md`
- `outputs\report_assets\repair_mask_r011_suite\metrics_summary.csv`
- `outputs\report_assets\repair_mask_r011_suite\demo_index.csv`
- `outputs\report_assets\repair_mask_r011_suite\repair_dataset_summary.csv`

## Recommendation

- Current automatic baseline: `r010_dl_t070`
- Improved automatic candidates: `r010/r011 + union + Module 1.5`
- `manual_upper_bound` chỉ là ceiling/diagnosis, không phải automatic result.
- Nếu r011 vẫn chưa đủ, bước tiếp theo là r012 với encoder mạnh hơn như ResNet34/pretrained encoder/deep supervision hoặc tăng real dataset vượt 60 ảnh.
