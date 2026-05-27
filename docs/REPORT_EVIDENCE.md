# Report Evidence Index

Mục tiêu của file này là lưu mapping giữa claim trong báo cáo và bằng chứng tương ứng trong repo.

| Claim | Evidence path | Type | Notes |
|---|---|---|---|
| Degradation pipeline tạo mask GT chính xác | `data/processed/<dataset_id>/previews/` | image preview | Overlay degraded/mask để kiểm tra trực quan |
| Dataset statistics hợp lệ | `data/processed/<dataset_id>/stats.json` | json | Thống kê phân bố mask pixel |
| Synthetic dataset manifest phục vụ truy vết sample | `data/processed/<dataset_id>/manifest.csv` | csv | Mapping giữa sample, clean source, crack source |
| Audit phát hiện dataset v001 không phù hợp để train segmentation | `data/processed/ds-crack3d-512-n0200-v001/audit/audit_report.json` | json | Failed/rejected dataset evidence, không dùng để train |
| Overlay audit xác nhận mask quá lớn trên dataset v001 | `data/processed/ds-crack3d-512-n0200-v001/audit/overlays/` | image overlay | Failed/rejected dataset evidence, không dùng để train |
| Dataset v002 đã được accept cho segmentation smoke tests | `data/processed/ds-crack3d-512-n0200-v002/stats.json` | json | Accepted dataset statistics |
| Manifest của dataset v002 sẵn sàng cho segmentation smoke tests | `data/processed/ds-crack3d-512-n0200-v002/manifest.csv` | csv | Accepted dataset sample mapping |
| Audit report của dataset v002 đã pass | `data/processed/ds-crack3d-512-n0200-v002/audit/audit_report.json` | json | Accepted dataset evidence for smoke tests |
| Overlay audit của dataset v002 xác nhận mask ratio hợp lệ | `data/processed/ds-crack3d-512-n0200-v002/audit/overlays/` | image overlay | Accepted dataset evidence for smoke tests |
| Dataset n1000 đã được accept cho segmentation smoke training | `data/processed/ds-crack3d-512-n1000-v001/stats.json` | json | Accepted dataset statistics for larger smoke training set |
| Manifest của dataset n1000 sẵn sàng cho segmentation smoke training | `data/processed/ds-crack3d-512-n1000-v001/manifest.csv` | csv | Accepted dataset sample mapping |
| Audit report của dataset n1000 đã pass | `data/processed/ds-crack3d-512-n1000-v001/audit/audit_report.json` | json | `num_missing_files = 0`, `num_reject_ratio = 0`, `mean_mask_ratio < 0.10` |
| Overlay audit của dataset n1000 xác nhận crack placement hợp lệ | `data/processed/ds-crack3d-512-n1000-v001/audit/overlays/` | image overlay | Dùng để kiểm tra trực quan trước smoke training |
| Unit test xác nhận các hàm degradation core đúng shape, dtype, range | `tests/test_degradation.py` | test | Kiểm tra `compute_heightmap`, `compute_normal_map`, `alpha_blend`, `generate_degraded_pair` |
| Contract test xác nhận active dataset giữ đúng cấu trúc lưu trữ | `tests/test_build_dataset_contract.py` | test | Kiểm tra manifest, stats, metadata, config snapshot và số lượng file |
| Audit contract test xác nhận active dataset không có reject ratio | `tests/test_audit_dataset_contract.py` | test | Ràng buộc `num_missing_files == 0`, `num_reject_ratio == 0`, `mean_mask_ratio < 0.10` |
| Dataset/DataLoader segmentation load được sample và batch chuẩn | `tests/test_segmentation_dataset.py` | test | Xác nhận tensor image `[3,512,512]`, mask `[1,512,512]`, DataLoader batch size 2 |
| Segmentation skeleton forward, loss và metric chạy được | `tests/test_segmentation_model_smoke.py` | test | Smoke test cho `CrackSegmenter`, `bce_dice_loss`, IoU/F1/Precision/Recall |
| Smoke training segmentation đã lưu metric, checkpoint và registry đúng quy ước | `checkpoints/segmenter/seg-unet-attn-r001-s42/metrics.json` | json | Lịch sử 5 epoch và best metric của smoke run |
| Smoke training segmentation đã ghi log epoch theo experiment artifact | `experiments/segmenter/seg-unet-attn-r001-s42/metrics.csv` | csv | `train_loss`, `val_loss`, `val_iou`, `val_f1`, `val_precision`, `val_recall` theo từng epoch |
| Prediction export của smoke run r001 hỗ trợ kiểm tra trực quan under/over-segmentation | `outputs/debug/segmentation/seg-unet-attn-r001-s42/` | image panels | Mỗi panel gồm degraded input, ground-truth mask, predicted mask, overlay |
| Threshold sweep của smoke run r001 cho thấy ngưỡng tối ưu khác 0.5 | `experiments/segmenter/seg-unet-attn-r001-s42/threshold_sweep.csv` | csv | So sánh IoU/F1/Precision/Recall trên val split |
| Registry experiment đã ghi nhận smoke run hoàn tất | `results/registry/experiment_registry.csv` | csv | `status = smoke_completed`, chưa phải final training |
| Registry metric đã ghi nhận best metric của smoke run | `results/registry/metric_registry.csv` | csv | Best validation IoU/F1/Precision/Recall của `seg-unet-attn-r001-s42` |
| Best smoke run hiện tại đã được đánh dấu nhưng chưa phải final run | `results/registry/best_runs.md` | markdown | Chỉ định rõ đây là best smoke run hiện tại |
| Prediction export của smoke run r002 hỗ trợ so sánh trực quan với r001 | `outputs/debug/segmentation/seg-unet-attn-r002-s42/` | image panels | Dùng cùng val split để so sánh chất lượng dự đoán |
| Threshold sweep của smoke run r002 cho thấy ngưỡng tối ưu theo IoU/F1 | `experiments/segmenter/seg-unet-attn-r002-s42/threshold_sweep.csv` | csv | Dùng để quyết định threshold inference phù hợp hơn 0.5 |
| LaMa cải thiện chất lượng ảnh | `outputs/figures/report/` | figure | Chưa áp dụng ở giai đoạn này |
