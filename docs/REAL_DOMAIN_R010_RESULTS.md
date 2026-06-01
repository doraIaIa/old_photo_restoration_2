# Real Domain R010 Results

## Dataset

- Real dataset: `F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq`
- Total pairs: `60`
- Split:
  - Train: `42`
  - Val: `9`
  - Test: `9`

## Checkpoints

- Baseline synthetic-only r009: `checkpoints\segmenter\seg-unet-attn-r009-aug-tversky03-07-ep60-s42\best_iou.ckpt`
- Real-domain fine-tuned r010: `checkpoints\segmenter\seg-unet-attn-r010-real-ft-s42\best_iou.ckpt`

## Evaluation Metrics

| Model | Split | Best threshold | IoU | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| r009 synthetic-only | Real test | 0.10 | 0.002222 | 0.004434 | 0.083464 | 0.002277 |
| r010 real fine-tuned | Real val | 0.70 | 0.237356 | 0.383650 | 0.415410 | 0.356402 |
| r010 real fine-tuned | Real test | 0.70 | 0.292728 | 0.452884 | 0.509123 | 0.407834 |

## Demo Variants

Các demo variants đã chạy:

- `r009_dl_t090`
- `r010_dl_t070`
- `r010_union_cv_t070_dilate0`
- `r010_union_cv_t070_dilate1`
- `r010_union_cv_t070_dilate2`
- `r010_union_cv_t050_dilate1`
- `manual_upper_bound`

Variant automatic được khuyến nghị cho báo cáo là `r010_union_cv_t070_dilate1` nếu visual review chọn nó. `manual_upper_bound` chỉ là ceiling/diagnosis, không được gọi là automatic result.

## Conclusion

Fine-tune real-domain cải thiện mạnh so với baseline r009 synthetic-only trên real old-photo test set. IoU tăng từ `0.002222` lên `0.292728`, F1 tăng từ `0.004434` lên `0.452884`.

Bottleneck chính vẫn là mask generation: khi mask đủ chính xác, LaMa/simple_lama có thể phục hồi tốt hơn; khi mask bỏ sót crack hoặc phủ sai texture/local contrast, chất lượng restoration bị giới hạn.
