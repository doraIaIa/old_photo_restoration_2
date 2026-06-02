# CodeFormer Activation Plan

## Trạng thái hiện tại

Module 3 đã có wrapper dependency-gated trong `src/restoration/face_restoration.py`, nhưng CodeFormer/GFPGAN chưa chạy thật trong môi trường hiện tại nếu metadata còn ghi `dependency_not_available` hoặc `adapter_not_configured`.

Không được claim CodeFormer completed nếu chưa có:

- Dependency và checkpoint face restoration sẵn sàng.
- Adapter inference ổn định.
- Output thật trên demo/test image.
- Metadata xác nhận `face_restoration_applied: true` và backend tương ứng.

## Nguyên tắc activation

- Không cài dependency nặng vào venv chính nếu có nguy cơ xung đột.
- Ưu tiên environment riêng cho CodeFormer/GFPGAN.
- Không tự tải/copy checkpoint vào Git.
- Khi thiếu dependency, pipeline phải giữ nguyên ảnh và ghi rõ lý do trong metadata.

## Dependency check

```powershell
python scripts\check_face_restoration_dependencies.py
```

Script chỉ kiểm tra import:

- `torch`
- `cv2`
- `basicsr`
- `facexlib`
- `gfpgan`
- `codeformer`

## Các bước tiếp theo

1. Tạo environment riêng cho CodeFormer/GFPGAN.
2. Pin version dependency và checkpoint.
3. Viết adapter inference có input/output contract rõ ràng.
4. Smoke test trên một ảnh demo.
5. Ghi metadata xác nhận backend và trạng thái applied.
6. Chỉ sau đó mới bật làm Module 3 thật trong pipeline.
