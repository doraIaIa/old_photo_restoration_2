# Project Milestone: r011 Pipeline and r012 Plan

## Current milestone

- r011 repair-mask model là current automatic baseline.
- Pipeline end-to-end đã có CLI chính cho single image và batch demo.
- Không gọi r011 là final solved result.

## What works

- r009 → r010 → r011 cải thiện metric rõ.
- `simple_lama` chạy được trong pipeline.
- External/manual mask cho thấy upper bound tốt hơn automatic mask.

## What remains weak

- Visual vẫn còn sót vết nứt dài/mờ.
- Mask tự động chưa giống repair mask thủ công.
- CV chỉ hỗ trợ bắt thêm candidate crack, không phải giải pháp chính.

## Final pipeline CLI

Single image:

```powershell
python scripts\run_restoration_pipeline.py ^
  --image data\demo_inputs\real_manual_3\demo3.png ^
  --mode auto_r011_union_refined ^
  --output-dir outputs\final_pipeline_assets\single_demo
```

Batch demo:

```powershell
python scripts\run_final_pipeline_suite.py ^
  --demo-dir data\demo_inputs\real_manual_3 ^
  --external-mask-dir data\demo_masks\real_manual_3 ^
  --output-root outputs\final_pipeline_assets\demo_suite
```

## Next research step

- Annotate `masks_repair_manual`.
- Audit manual repair masks.
- Train r012 từ r011 sau khi audit pass.
- Optional architecture upgrade: ResNet34 encoder / pretrained encoder / deep supervision.

## Do not overclaim

- Current pipeline is functional but not perfect.
- Manual mask is upper bound, not automatic result.
- r011 is current automatic baseline, not final solved result.
