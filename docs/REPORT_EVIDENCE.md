# Report Evidence Index

Mục tiêu của file này là lưu mapping giữa claim trong báo cáo và bằng chứng tương ứng trong repo.

| Claim | Evidence path | Type | Notes |
|---|---|---|---|
| Degradation pipeline tạo mask GT chính xác | `data/processed/<dataset_id>/previews/` | image preview | overlay degraded/mask |
| Dataset statistics hợp lệ | `data/processed/<dataset_id>/stats.json` | json | mask pixel distribution |
| Synthetic dataset manifest phục vụ truy vết sample | `data/processed/<dataset_id>/manifest.csv` | csv | clean source, crack source, sample mapping |
| Audit phát hiện dataset v001 không phù hợp để train segmentation | `data/processed/ds-crack3d-512-n0200-v001/audit/audit_report.json` | json | failed/rejected dataset evidence, không dùng để train |
| Overlay audit xác nhận mask quá lớn trên dataset v001 | `data/processed/ds-crack3d-512-n0200-v001/audit/overlays/` | image overlay | failed/rejected dataset evidence, không dùng để train |
| Segmentation model đạt IoU tốt | `results/registry/metric_registry.csv` | metric | val IoU/F1 |
| LaMa cải thiện chất lượng ảnh | `outputs/figures/report/` | figure | before/after comparison |
