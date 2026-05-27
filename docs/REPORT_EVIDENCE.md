# Report Evidence Index

Mục tiêu của file này là lưu mapping giữa claim trong báo cáo và bằng chứng tương ứng trong repo.

| Claim | Evidence path | Type | Notes |
|---|---|---|---|
| Degradation pipeline tạo mask GT chính xác | `data/processed/<dataset_id>/previews/` | image preview | overlay degraded/mask |
| Dataset statistics hợp lệ | `data/processed/<dataset_id>/stats.json` | json | mask pixel distribution |
| Segmentation model đạt IoU tốt | `results/registry/metric_registry.csv` | metric | val IoU/F1 |
| LaMa cải thiện chất lượng ảnh | `outputs/figures/report/` | figure | before/after comparison |
