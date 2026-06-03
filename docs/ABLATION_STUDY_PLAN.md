# Ablation Study Plan

Kế hoạch này dùng để so sánh vai trò của mask mode, backend inpainting và CodeFormer. Khi chưa có data mới, chỉ chạy smoke trên demo input cũ và không kết luận case nào tốt nhất.

## Case đề xuất

| Case | Cấu hình | Mục tiêu |
| --- | --- | --- |
| A | OpenCV fallback/baseline | Mốc thấp, kiểm tra fallback luôn có đường chạy. |
| B | `r011` default + `official_lama`, face off | Baseline pipeline chính không có Module 3. |
| C | `r011` sensitive + `official_lama`, face off | Kiểm tra lợi ích và rủi ro của recall-sensitive mask. |
| D | `r011` default hoặc sensitive + `official_lama` + CodeFormer fidelity `0.7` | Đánh giá tác động Module 3 lên mặt. |
| E | External/oracle mask + `official_lama` nếu manual mask demo có sẵn | Upper-bound chẩn đoán: nếu mask tốt mà ảnh vẫn lỗi, bottleneck nghiêng về inpainting. |

## Metric và review

- Với mask có ground truth: IoU, F1, precision, recall.
- Với pipeline output: visual review theo `docs/EVALUATION_PROTOCOL.md`.
- Với runtime: chỉ ghi `runtime_seconds` như quan sát smoke; chưa dùng để claim speedup.
- Với fallback: luôn ghi `backend_actual` và `fallback_applied`.

## Script smoke

Ghi template:

```powershell
python scripts/run_ablation_smoke.py --write-template --output-dir outputs/ablation_templates
```

Chạy demo smoke:

```powershell
python scripts/run_ablation_smoke.py --input-dir data/demo_inputs/real_manual_3 --manual-mask-dir data/demo_inputs/real_manual_3/manual_masks --max-images 3 --backend official_lama
```

Case thiếu manual mask phải được skip rõ trong `notes`, không crash mơ hồ.
