# Skill Matrix theo phase

## Cách đọc tài liệu này

Mỗi skill ghi rõ:

- Mục tiêu cần hiểu
- Dấu hiệu đã hiểu
- File liên quan trong project
- Giai đoạn cần dùng

## Phase 0 — Environment & Repo Basics

### Python virtual environment
- Mục tiêu cần hiểu: tách dependency theo project, biết tạo và kích hoạt môi trường cục bộ.
- Dấu hiệu đã hiểu: cài package không làm bẩn Python hệ thống, chạy đúng interpreter trong `venv`.
- File liên quan: [requirements.txt](F:/deeplearning/old_photo_restoration_2/requirements.txt), [SETUP_ENV.md](F:/deeplearning/old_photo_restoration_2/SETUP_ENV.md)
- Giai đoạn cần dùng: Phase 0

### pip/requirements
- Mục tiêu cần hiểu: cài đúng phiên bản thư viện, đọc và cập nhật `requirements.txt`.
- Dấu hiệu đã hiểu: cài được toàn bộ dependency và giải thích được package nào phục vụ phần nào.
- File liên quan: [requirements.txt](F:/deeplearning/old_photo_restoration_2/requirements.txt)
- Giai đoạn cần dùng: Phase 0

### Git basics
- Mục tiêu cần hiểu: add, commit, status, phân biệt file nên và không nên commit.
- Dấu hiệu đã hiểu: không đẩy data/checkpoint lên git và đọc được trạng thái worktree.
- File liên quan: [.gitignore](F:/deeplearning/old_photo_restoration_2/.gitignore)
- Giai đoạn cần dùng: Phase 0 trở đi

### YAML/OmegaConf
- Mục tiêu cần hiểu: cấu trúc config dạng key-value lồng nhau và cách tách path khỏi code.
- Dấu hiệu đã hiểu: mọi path/hyperparameter nằm trong `configs/`, không hard-code trong script.
- File liên quan: [configs/data.yaml](F:/deeplearning/old_photo_restoration_2/configs/data.yaml)
- Giai đoạn cần dùng: Phase 0, Phase 1

### .gitignore
- Mục tiêu cần hiểu: kiểm soát artifact nào bị ignore.
- Dấu hiệu đã hiểu: biết vì sao `data/`, `outputs/`, `checkpoints/` không được commit.
- File liên quan: [.gitignore](F:/deeplearning/old_photo_restoration_2/.gitignore)
- Giai đoạn cần dùng: Phase 0, Phase 0.5

### VS Code/terminal workflow
- Mục tiêu cần hiểu: chạy script, đọc log, kiểm tra file, điều hướng repo.
- Dấu hiệu đã hiểu: dùng terminal để kiểm tra output thay vì thao tác tay thiếu kiểm soát.
- File liên quan: toàn repo
- Giai đoạn cần dùng: mọi phase

## Phase 0.5 — Project Hygiene & Artifact Management

### dataset_id/run_id
- Mục tiêu cần hiểu: nhận diện dataset và run ổn định, không dùng timestamp.
- Dấu hiệu đã hiểu: đặt tên đúng format và không ghi đè artifact cũ.
- File liên quan: [docs/STORAGE_CONVENTIONS.md](F:/deeplearning/old_photo_restoration_2/docs/STORAGE_CONVENTIONS.md), [configs/data.yaml](F:/deeplearning/old_photo_restoration_2/configs/data.yaml)
- Giai đoạn cần dùng: trước mọi script sinh dữ liệu, train, infer

### manifest.csv
- Mục tiêu cần hiểu: lưu mapping từng sample để truy vết source.
- Dấu hiệu đã hiểu: truy được một sample về clean source và crack source gốc.
- File liên quan: `data/processed/<dataset_id>/manifest.csv`, [docs/STORAGE_CONVENTIONS.md](F:/deeplearning/old_photo_restoration_2/docs/STORAGE_CONVENTIONS.md)
- Giai đoạn cần dùng: Phase 0.5, Phase 1

### stats.json
- Mục tiêu cần hiểu: tổng hợp thống kê dataset và sanity-check mask distribution.
- Dấu hiệu đã hiểu: đọc min/mean/max `mask_pixels` để phát hiện dataset bất thường.
- File liên quan: `data/processed/<dataset_id>/stats.json`
- Giai đoạn cần dùng: Phase 0.5, Phase 1.5

### metadata.json
- Mục tiêu cần hiểu: lưu `created_at`, seed, config snapshot, git commit, notes.
- Dấu hiệu đã hiểu: có thể tái tạo dataset hoặc run dựa trên metadata.
- File liên quan: `dataset_metadata.json`, `run_metadata.json`
- Giai đoạn cần dùng: mọi phase sinh artifact

### config_snapshot.yaml
- Mục tiêu cần hiểu: chụp lại config tại thời điểm chạy để đảm bảo tái lập.
- Dấu hiệu đã hiểu: mỗi dataset/run đều có snapshot config riêng.
- File liên quan: `config_snapshot.yaml` trong dataset, checkpoint, experiment
- Giai đoạn cần dùng: Phase 0.5 trở đi

### checkpoint convention
- Mục tiêu cần hiểu: checkpoint phải nằm theo `run_id`, không để file rời.
- Dấu hiệu đã hiểu: biết nơi lưu `best_iou.ckpt`, `best_lpips.ckpt`, `last.ckpt`.
- File liên quan: [docs/STORAGE_CONVENTIONS.md](F:/deeplearning/old_photo_restoration_2/docs/STORAGE_CONVENTIONS.md)
- Giai đoạn cần dùng: Phase 2, Phase 3

### output convention
- Mục tiêu cần hiểu: debug, inference, figure phải có root riêng.
- Dấu hiệu đã hiểu: không còn ảnh output nằm ở root project.
- File liên quan: [configs/data.yaml](F:/deeplearning/old_photo_restoration_2/configs/data.yaml), [docs/STORAGE_CONVENTIONS.md](F:/deeplearning/old_photo_restoration_2/docs/STORAGE_CONVENTIONS.md)
- Giai đoạn cần dùng: Phase 1, Phase 4, Phase 5

### reproducibility
- Mục tiêu cần hiểu: seed, config, source data, manifest, metadata phải đủ để chạy lại.
- Dấu hiệu đã hiểu: giải thích được vì sao hai lần build có khác nhau hay không.
- File liên quan: [configs/data.yaml](F:/deeplearning/old_photo_restoration_2/configs/data.yaml), `manifest.csv`, `stats.json`, metadata files
- Giai đoạn cần dùng: mọi phase

## Phase 1 — Degradation Pipeline

### NumPy array shape/range/dtype
- Mục tiêu cần hiểu: quản lý shape HWC/CHW, dtype uint8/float32, range [0,255]/[0,1].
- Dấu hiệu đã hiểu: không nhầm shape khi blend hay tạo mask.
- File liên quan: [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py)
- Giai đoạn cần dùng: Phase 1

### OpenCV image I/O
- Mục tiêu cần hiểu: đọc/ghi ảnh bằng OpenCV, biết cách kiểm tra lỗi đọc ảnh.
- Dấu hiệu đã hiểu: phát hiện được ảnh hỏng hoặc load `None`.
- File liên quan: [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py)
- Giai đoạn cần dùng: Phase 1, Phase 1.5

### RGB/BGR/RGBA differences
- Mục tiêu cần hiểu: khác biệt giữa định dạng màu và tác động lên blending/preview.
- Dấu hiệu đã hiểu: nhìn ra ngay khi màu bị đảo kênh.
- File liên quan: [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py)
- Giai đoạn cần dùng: Phase 1, Phase 1.5

### Alpha channel
- Mục tiêu cần hiểu: vai trò của alpha trong crack asset và mask ground truth.
- Dấu hiệu đã hiểu: tạo được mask đáng tin từ alpha hoặc logic thay thế.
- File liên quan: [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py)
- Giai đoạn cần dùng: Phase 1

### Alpha blending
- Mục tiêu cần hiểu: công thức blend crack lên clean image.
- Dấu hiệu đã hiểu: blend đúng shape, không lệch vị trí, không làm vỡ dtype.
- File liên quan: [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py)
- Giai đoạn cần dùng: Phase 1

### Heightmap
- Mục tiêu cần hiểu: chuyển crack thành bản đồ độ cao.
- Dấu hiệu đã hiểu: phân biệt được vai trò của invert, blur và normalize.
- File liên quan: [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py)
- Giai đoạn cần dùng: Phase 1

### Sobel gradient
- Mục tiêu cần hiểu: lấy gradient theo x/y để dựng normal map.
- Dấu hiệu đã hiểu: giải thích được vì sao không lấy Sobel trực tiếp từ luminance để thay normal map.
- File liên quan: [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py)
- Giai đoạn cần dùng: Phase 1

### Normal map
- Mục tiêu cần hiểu: vector pháp tuyến đơn vị từ heightmap.
- Dấu hiệu đã hiểu: kiểm được norm gần 1 trên toàn ảnh.
- File liên quan: [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py), [tests/test_degradation.py](F:/deeplearning/old_photo_restoration_2/tests/test_degradation.py)
- Giai đoạn cần dùng: Phase 1

### Phong illumination
- Mục tiêu cần hiểu: mô phỏng ánh sáng cho crack 3D.
- Dấu hiệu đã hiểu: output có vùng highlight và shadow hợp lý, range không vỡ.
- File liên quan: [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py)
- Giai đoạn cần dùng: Phase 1

### Global degradation: noise, blur, sepia
- Mục tiêu cần hiểu: các thành phần suy thoái toàn cục ngoài crack.
- Dấu hiệu đã hiểu: phân biệt được crack artifact và global degradation artifact.
- File liên quan: [src/data/degradation.py](F:/deeplearning/old_photo_restoration_2/src/data/degradation.py)
- Giai đoạn cần dùng: Phase 1

### Data preview
- Mục tiêu cần hiểu: kiểm tra nhanh degraded, mask, ground truth bằng panel.
- Dấu hiệu đã hiểu: nhìn preview phát hiện được lỗi màu, lỗi mask, lỗi crop.
- File liên quan: `previews/`, notebook preview sau này, `outputs/debug/`
- Giai đoạn cần dùng: Phase 1, Phase 1.5

## Phase 1.5 — Data Quality Audit

### mask not empty
- Mục tiêu cần hiểu: mask rỗng làm hỏng training supervision.
- Dấu hiệu đã hiểu: có ngưỡng chấp nhận cho số mẫu mask rỗng.
- File liên quan: `manifest.csv`, `stats.json`, preview grid
- Giai đoạn cần dùng: sau khi build dataset

### mask pixel ratio
- Mục tiêu cần hiểu: tỷ lệ pixel mask quá nhỏ hoặc quá lớn đều đáng ngờ.
- Dấu hiệu đã hiểu: đọc được phân phối `mask_pixels` và gắn cờ outlier.
- File liên quan: `stats.json`, `manifest.csv`
- Giai đoạn cần dùng: Phase 1.5

### mask overlay
- Mục tiêu cần hiểu: xác minh mask chồng đúng lên vùng crack.
- Dấu hiệu đã hiểu: phát hiện lệch vị trí hoặc mask sai hình.
- File liên quan: `previews/`
- Giai đoạn cần dùng: Phase 1.5

### degraded/gt/mask preview grid
- Mục tiêu cần hiểu: so sánh ba thành phần theo sample.
- Dấu hiệu đã hiểu: phát hiện nhanh sample bất thường bằng mắt.
- File liên quan: `previews/`
- Giai đoạn cần dùng: Phase 1.5

### detecting broken images
- Mục tiêu cần hiểu: file ảnh hỏng, 0 byte, đọc lỗi.
- Dấu hiệu đã hiểu: script audit bắt được file lỗi trước khi build.
- File liên quan: raw folders, processed folders
- Giai đoạn cần dùng: trước và sau build

### detecting wrong RGB/BGR colors
- Mục tiêu cần hiểu: ảnh bị đảo kênh màu có thể vẫn hợp lệ về shape nhưng sai nội dung.
- Dấu hiệu đã hiểu: nhìn preview phát hiện được tông da/cây trời bị sai.
- File liên quan: `previews/`, output debug
- Giai đoạn cần dùng: Phase 1.5

### detecting excessive/too small masks
- Mục tiêu cần hiểu: xác định crack visibility hợp lý.
- Dấu hiệu đã hiểu: có tiêu chí reject dataset nếu mask bất thường quá nhiều.
- File liên quan: `stats.json`, `manifest.csv`
- Giai đoạn cần dùng: Phase 1.5

## Phase 2 — Segmentation

### PyTorch tensors
- Mục tiêu cần hiểu: BCHW, dtype tensor, device, sigmoid/logits.
- Dấu hiệu đã hiểu: forward pass không lỗi shape.
- File liên quan: [src/data/dataset.py](F:/deeplearning/old_photo_restoration_2/src/data/dataset.py), [src/models/segmenter.py](F:/deeplearning/old_photo_restoration_2/src/models/segmenter.py)
- Giai đoạn cần dùng: Phase 2

### Dataset/DataLoader
- Mục tiêu cần hiểu: dataset trả đúng cặp image/mask và loader batch hóa đúng.
- Dấu hiệu đã hiểu: load được 1 batch ổn định.
- File liên quan: [src/data/dataset.py](F:/deeplearning/old_photo_restoration_2/src/data/dataset.py)
- Giai đoạn cần dùng: Phase 2

### U-Net
- Mục tiêu cần hiểu: encoder-decoder với skip connection.
- Dấu hiệu đã hiểu: giải thích được đường đi feature map.
- File liên quan: [src/models/segmenter.py](F:/deeplearning/old_photo_restoration_2/src/models/segmenter.py)
- Giai đoạn cần dùng: Phase 2

### ResNet encoder
- Mục tiêu cần hiểu: backbone pretrained cho segmentation.
- Dấu hiệu đã hiểu: nối đúng feature stages với decoder.
- File liên quan: [src/models/segmenter.py](F:/deeplearning/old_photo_restoration_2/src/models/segmenter.py)
- Giai đoạn cần dùng: Phase 2

### Attention Gate
- Mục tiêu cần hiểu: lọc skip feature theo tín hiệu gating.
- Dấu hiệu đã hiểu: giải thích được đầu vào/đầu ra của gate.
- File liên quan: [src/models/attention_gate.py](F:/deeplearning/old_photo_restoration_2/src/models/attention_gate.py)
- Giai đoạn cần dùng: Phase 2

### Deep Supervision
- Mục tiêu cần hiểu: auxiliary heads ở nhiều scale.
- Dấu hiệu đã hiểu: biết cộng loss nhiều mức và resize đúng.
- File liên quan: [src/models/segmenter.py](F:/deeplearning/old_photo_restoration_2/src/models/segmenter.py)
- Giai đoạn cần dùng: Phase 2

### BCE Loss
- Mục tiêu cần hiểu: loss pixel-wise cho bài toán nhị phân.
- Dấu hiệu đã hiểu: biết khi nào dùng logits và khi nào dùng probability.
- File liên quan: [src/losses/segmentation.py](F:/deeplearning/old_photo_restoration_2/src/losses/segmentation.py)
- Giai đoạn cần dùng: Phase 2

### Dice Loss
- Mục tiêu cần hiểu: xử lý class imbalance cho crack segmentation.
- Dấu hiệu đã hiểu: giải thích được intersection/union mềm.
- File liên quan: [src/losses/segmentation.py](F:/deeplearning/old_photo_restoration_2/src/losses/segmentation.py)
- Giai đoạn cần dùng: Phase 2

### IoU, F1, Precision, Recall
- Mục tiêu cần hiểu: metric đánh giá mask prediction.
- Dấu hiệu đã hiểu: đọc được trade-off giữa precision và recall.
- File liên quan: [src/utils/metrics.py](F:/deeplearning/old_photo_restoration_2/src/utils/metrics.py)
- Giai đoạn cần dùng: Phase 2, Phase 5

### PyTorch Lightning training loop
- Mục tiêu cần hiểu: `training_step`, `validation_step`, `configure_optimizers`, callbacks.
- Dấu hiệu đã hiểu: chạy được smoke training ngắn với checkpoint/logging.
- File liên quan: [scripts/train_segmentation.py](F:/deeplearning/old_photo_restoration_2/scripts/train_segmentation.py)
- Giai đoạn cần dùng: Phase 2

## Phase 3 — Restoration / LaMa

### Inpainting
- Mục tiêu cần hiểu: phục hồi vùng khuyết dựa trên mask.
- Dấu hiệu đã hiểu: phân biệt input masked image với target clean image.
- File liên quan: [src/models/lama_gan.py](F:/deeplearning/old_photo_restoration_2/src/models/lama_gan.py)
- Giai đoạn cần dùng: Phase 3

### Masked image
- Mục tiêu cần hiểu: cách kết hợp ảnh và mask làm input cho restoration model.
- Dấu hiệu đã hiểu: dựng đúng tensor đầu vào 4 kênh hoặc dạng masked expected.
- File liên quan: [src/data/dataset.py](F:/deeplearning/old_photo_restoration_2/src/data/dataset.py)
- Giai đoạn cần dùng: Phase 3

### Fast Fourier Convolution
- Mục tiêu cần hiểu: receptive field toàn cục trong LaMa.
- Dấu hiệu đã hiểu: giải thích được local branch và global branch.
- File liên quan: [src/models/ffc.py](F:/deeplearning/old_photo_restoration_2/src/models/ffc.py)
- Giai đoạn cần dùng: Phase 3

### Perceptual Loss VGG-19
- Mục tiêu cần hiểu: loss cảm nhận thay vì chỉ pixel loss.
- Dấu hiệu đã hiểu: giải thích được vì sao L1 một mình chưa đủ.
- File liên quan: [src/losses/restoration.py](F:/deeplearning/old_photo_restoration_2/src/losses/restoration.py)
- Giai đoạn cần dùng: Phase 3

### PatchGAN
- Mục tiêu cần hiểu: discriminator đánh giá patch-level realism.
- Dấu hiệu đã hiểu: hiểu vì sao dùng patch thay vì toàn ảnh.
- File liên quan: [src/models/lama_gan.py](F:/deeplearning/old_photo_restoration_2/src/models/lama_gan.py)
- Giai đoạn cần dùng: Phase 3

### GAN training loop
- Mục tiêu cần hiểu: alternating update generator/discriminator.
- Dấu hiệu đã hiểu: log được loss của cả hai mạng và không vỡ training ngay từ đầu.
- File liên quan: [scripts/train_restoration.py](F:/deeplearning/old_photo_restoration_2/scripts/train_restoration.py)
- Giai đoạn cần dùng: Phase 3

### Mixed precision
- Mục tiêu cần hiểu: tăng tốc và giảm memory trên GPU.
- Dấu hiệu đã hiểu: dùng đúng precision mode mà không phá ổn định loss.
- File liên quan: scripts train sau này
- Giai đoạn cần dùng: Phase 3

### LPIPS, FID, PSNR, SSIM
- Mục tiêu cần hiểu: metric cảm nhận và tham chiếu cho restoration.
- Dấu hiệu đã hiểu: không nhầm metric chính và metric phụ.
- File liên quan: [src/utils/metrics.py](F:/deeplearning/old_photo_restoration_2/src/utils/metrics.py)
- Giai đoạn cần dùng: Phase 3, Phase 5

## Phase 4 — Face Restoration

### RetinaFace
- Mục tiêu cần hiểu: detect face trước khi đưa vào face restoration.
- Dấu hiệu đã hiểu: biết khi nào bỏ qua Module 3 nếu không có mặt.
- File liên quan: [scripts/infer.py](F:/deeplearning/old_photo_restoration_2/scripts/infer.py)
- Giai đoạn cần dùng: Phase 4

### CodeFormer
- Mục tiêu cần hiểu: phục hồi mặt ở chế độ inference.
- Dấu hiệu đã hiểu: dùng đúng face crop và merge lại vào ảnh tổng.
- File liên quan: [scripts/infer.py](F:/deeplearning/old_photo_restoration_2/scripts/infer.py)
- Giai đoạn cần dùng: Phase 4

### Face crop/align/paste-back
- Mục tiêu cần hiểu: căn chỉnh và gắn mặt phục hồi về ảnh gốc.
- Dấu hiệu đã hiểu: không bị lệch vị trí hoặc khác tỷ lệ.
- File liên quan: [scripts/infer.py](F:/deeplearning/old_photo_restoration_2/scripts/infer.py)
- Giai đoạn cần dùng: Phase 4

### Identity caution
- Mục tiêu cần hiểu: face restoration có thể làm thay đổi đặc trưng nhận dạng.
- Dấu hiệu đã hiểu: mô tả được rủi ro khi demo kết quả mặt người.
- File liên quan: docs demo và infer
- Giai đoạn cần dùng: Phase 4, Phase 5

## Phase 5 — Evaluation & Demo

### Ablation study
- Mục tiêu cần hiểu: so sánh từng thành phần pipeline.
- Dấu hiệu đã hiểu: thiết kế được bảng so sánh có kiểm soát.
- File liên quan: notebook eval, docs report
- Giai đoạn cần dùng: Phase 5

### Oracle mask evaluation
- Mục tiêu cần hiểu: tách chất lượng Module 2 khỏi lỗi Module 1.
- Dấu hiệu đã hiểu: biết khi nào dùng GT mask thay vì predicted mask.
- File liên quan: scripts eval/infer sau này
- Giai đoạn cần dùng: Phase 5

### Gradio demo
- Mục tiêu cần hiểu: đóng gói inference thành demo có input/output rõ.
- Dấu hiệu đã hiểu: output đi đúng `outputs/inference/<run_id>/`.
- File liên quan: demo script sau này
- Giai đoạn cần dùng: Phase 5

### report figures
- Mục tiêu cần hiểu: lưu figure có cấu trúc để đưa vào báo cáo.
- Dấu hiệu đã hiểu: figure không nằm rải rác ở root.
- File liên quan: `outputs/figures/report/`
- Giai đoạn cần dùng: Phase 5

### qualitative comparison
- Mục tiêu cần hiểu: so sánh trực quan giữa baseline, từng module, và full pipeline.
- Dấu hiệu đã hiểu: dựng panel so sánh nhất quán giữa các mẫu.
- File liên quan: report figures, notebook eval
- Giai đoạn cần dùng: Phase 5
