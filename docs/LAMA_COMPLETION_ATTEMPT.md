# LaMa Completion Attempt

Tài liệu này ghi lại trạng thái thử chuẩn bị official/pretrained LaMa cho Blueprint 2.1 Module 2. Không có checkpoint, output ảnh, dataset hoặc external model nào được copy vào repo.

## Phạm vi

- Workspace ngoài repo: `F:\deeplearning\external_models\lama`
- Conda env riêng: `lama`
- Mục tiêu trước mắt: kiểm tra official/pretrained LaMa có thể chạy inference demo3 với mask sensitive/oracle hay không.
- Không cài dependency nặng vào môi trường chính của project.

## Trạng thái hiện tại

- Workspace ngoài repo đã có: `F:\deeplearning\external_models\lama`
- Source official LaMa đã clone ngoài repo: `F:\deeplearning\external_models\lama\lama`
- Conda env riêng `lama` chạy được Python 3.8.
- PyTorch trong env `lama` đang chạy CPU: `torch 2.1.2+cpu`, `torchvision 0.16.2+cpu`.
- Requirements official LaMa đã cài trong env `lama`; có chỉnh dependency bằng cách hạ `numpy` xuống `1.23.5` để tránh lỗi `np.bool` từ `scikit-image 0.17.2`.
- Pretrained `big-lama` đã tải từ link Hugging Face được README official nêu và giải nén ngoài repo.
- Official/pretrained LaMa đã chạy inference thật trên demo3 oracle mask bằng `device=cpu`.

## Điều kiện để chạy official/pretrained LaMa

- Có source official LaMa đầy đủ trong `F:\deeplearning\external_models\lama`.
- Có pretrained checkpoint hợp lệ (`.ckpt`, `.pth` hoặc `.pt`) ngoài repo.
- Env `lama` import được PyTorch và các dependency của LaMa.
- Có command inference đã smoke-test trên Windows, hoặc chạy qua WSL/Colab/Kaggle nếu dependency Windows rủi ro.

## Điều kiện để fine-tune LaMa

- Official/pretrained inference phải pass trước.
- Có manifest clean images và mask bank để tạo synthetic inpainting pairs.
- Có tiny overfit/smoke-test trước khi chạy fine-tune thật.
- Chưa claim LaMa fine-tune completed khi chưa có log training, checkpoint và visual/metric chứng minh.

## Kết luận tạm thời

Official/pretrained LaMa hiện đã runnable trong env riêng `lama`, nhưng backend decision chưa đổi tự động cho project cho tới khi có human visual review so sánh output official LaMa với `simple_lama`. Chưa fine-tune LaMa.

## Final validation update

- Official/pretrained LaMa đã được tích hợp vào pipeline chính qua subprocess adapter `src/restoration/official_lama_adapter.py`.
- Backend CLI hiện hỗ trợ `--inpaint-backend official_lama`.
- Final validation trên demo1/demo2/demo3 đã chạy 9 case và pass `9/9`.
- Backend decision hiện tại: `official_lama_pretrained` là runnable candidate, `simple_lama` vẫn là stable fallback.
- Không đổi default toàn project cho tới khi human visual review đủ rõ.
- LaMa fine-tune vẫn chưa thực hiện.
