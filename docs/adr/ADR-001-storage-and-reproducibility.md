# ADR-001: Storage and Reproducibility

## Bối cảnh

Project dễ bị bẩn nếu output, checkpoint, dataset và log được lưu rải rác hoặc ghi đè lẫn nhau. Điều này làm giảm khả năng audit, khó tái lập kết quả và gây nhầm lẫn khi báo cáo.

## Quyết định

Áp dụng các quy ước sau:

- dùng `dataset_id` cho mọi dataset versioned
- dùng `run_id` cho mọi training, evaluation, inference run
- lưu `manifest.csv` để truy vết sample
- lưu `stats.json` để tóm tắt dataset
- lưu metadata file cho dataset và run
- lưu `config_snapshot.yaml` để tái lập cấu hình

## Hệ quả tích cực

- dễ audit dữ liệu
- dễ viết báo cáo
- dễ reproduce
- giảm nguy cơ ghi đè artifact cũ

## Trade-off

- ban đầu mất thời gian setup hơn
- script phải tạo thêm file metadata và registry
- cần kỷ luật hơn khi đặt tên dataset và run
