# ROADMAP_AND_TASKS.md
# Kế hoạch Thực thi & Ma trận Kỹ năng — Blueprint 2.1

> **Hướng dẫn dùng file này với AI Coding Assistant (Codex/Cursor):**
> Trước mỗi session, paste đoạn sau vào chat:
> *"Đọc ARCHITECTURE.md và ROADMAP_AND_TASKS.md trong thư mục gốc. Chúng ta đang ở [Phase X - Task Y]. Hãy giúp tôi viết code cho [tên file]."*

---

## Tổng quan Timeline

```
Phase 0 │ Tuần 1      │ Môi trường & Cấu trúc dự án
Phase 1 │ Tuần 1–2    │ Degradation Pipeline (Data Engine)
Phase 2 │ Tuần 3–5    │ Crack Segmentation (Module 1)
Phase 3 │ Tuần 6–10   │ LaMa Inpainting (Module 2)
Phase 4 │ Tuần 11     │ Tích hợp CodeFormer (Module 3)
Phase 5 │ Tuần 12–13  │ Evaluation, Demo & Báo cáo
```

**Trạng thái:** Dùng ký hiệu `[ ]` = chưa làm, `[x]` = xong, `[~]` = đang làm.

---

## Phase 0 — Môi trường & Khung dự án
**Thời gian:** 2–3 ngày | **GPU cần:** Không

### Tasks

```
[ ] 0.1  Cài Miniconda + tạo virtual environment `old_photo`
[ ] 0.2  Tạo cấu trúc thư mục theo ARCHITECTURE.md
[ ] 0.3  Tạo requirements.txt và cài đặt dependencies
[ ] 0.4  Khởi tạo W&B project
[ ] 0.5  Tạo .gitignore (bỏ qua data/, checkpoints/, __pycache__)
[ ] 0.6  Viết configs/data.yaml, configs/segmenter.yaml, configs/generator.yaml
```

### Files cần tạo

| File | Nội dung |
|------|----------|
| `requirements.txt` | Toàn bộ dependencies (xem mục Terminal Commands) |
| `configs/data.yaml` | Paths tới data/, split ratio, image size |
| `configs/segmenter.yaml` | lr, batch_size, epochs, loss weights (bce_weight, dice_weight) |
| `configs/generator.yaml` | lambda_l1, lambda_perceptual, lambda_adversarial |
| `configs/wandb.yaml` | project, entity, tags |
| `.gitignore` | data/, checkpoints/, *.ckpt, __pycache__, .env |

### 📚 Skill Matrix — Phase 0

> Những gì bạn cần hiểu để **review code AI sinh ra** ở phase này:

- **YAML syntax cơ bản:** key: value, nested dict, list. Đọc thêm: [yaml.org](https://yaml.org/start.html)
- **Virtual environment là gì:** Tại sao cần isolate dependencies giữa các project
- **Git cơ bản:** `git init`, `git add`, `git commit`, `git push`

---

## Phase 1 — Degradation Pipeline (Data Engine)
**Thời gian:** 3–5 ngày | **GPU cần:** Không (CPU only)

### Mục tiêu
Tạo ra hàm `generate_degraded_pair(clean_img, crack_rgba)` sinh ra cặp `(degraded_image, crack_mask)` với mask chính xác 100% và hiệu ứng vật lý thật.

### Tasks

```
[ ] 1.1  Thu thập Crack Bank: download CrackForest dataset
         Link: https://github.com/cuilimeng/CrackForest-dataset
         Lưu vào: data/crack_bank/raw/

[ ] 1.2  Viết src/data/degradation.py — core algorithm:
         - compute_heightmap()
         - compute_normal_map()        ← QUAN TRỌNG: công thức 3D đúng
         - apply_phong_illumination()
         - alpha_blend()
         - add_global_degradation()
         - generate_degraded_pair()    ← hàm tổng hợp

[ ] 1.3  Viết src/data/transforms.py — Albumentations pipeline
         - CrackAugment: rotate, scale, flip, elastic
         - ImageAugment: color jitter, random crop

[ ] 1.4  Viết scripts/build_dataset.py
         - Đọc clean images từ data/clean/
         - Đọc cracks từ data/crack_bank/processed/
         - Sinh pairs → lưu vào data/processed/train/ và data/processed/val/
         - Log stats: số lượng pairs, phân phối mask size

[ ] 1.5  Kiểm tra bằng notebooks/02_degradation_preview.ipynb
         - Hiển thị 10 cặp (degraded, mask) ngẫu nhiên
         - Verify: mask phải overlap đúng với vùng crack trên ảnh
```

### Thuật toán cốt lõi — `src/data/degradation.py`

```python
# ============================================================
# THUẬT TOÁN NORMAL MAPPING 3D — ĐÂY LÀ PHẦN QUAN TRỌNG NHẤT
# ============================================================

# Bước 1: Heightmap
# Mục đích: chuyển ảnh xám của crack thành "bản đồ độ cao"
# Crack tối → invert → crack là đỉnh cao nhất
heightmap = GaussianBlur(255 - crack_gray, sigma=3)
heightmap = normalize(heightmap)  # về [0, 1]

# Bước 2: Normal Map 3D  ← GEMINI ĐÃ SAI Ở ĐÂY
# Công thức đúng: N = normalize([-dH/dx * s, -dH/dy * s, 1])
# KHÔNG phải: N = Sobel(luminance) trực tiếp
dH_dx = Sobel(heightmap, dx=1, dy=0)   # gradient theo x
dH_dy = Sobel(heightmap, dx=0, dy=1)   # gradient theo y
N = normalize(stack([-dH_dx*s, -dH_dy*s, ones]))  # unit vector (H,W,3)

# Bước 3: Phong Illumination
# Mô phỏng nguồn sáng từ góc 45 độ → bóng đổ + viền sáng
L = normalize([0.5, 0.5, 1.0])         # hướng ánh sáng
I = 0.3 + 0.7 * clamp(dot(N, L), 0, 1) # I ∈ [0.3, 1.0]
crack_3d = crack_rgb * I               # apply illumination

# Bước 4: Alpha Blend
# out[y:y+h, x:x+w] = crack_3d * α + clean[y:y+h, x:x+w] * (1-α)

# Bước 5: Mask GT (Ground Truth)
mask = (alpha_channel > 30).astype(uint8) * 255  # binary, 100% chính xác

# Bước 6: Global Degradation
# + Gaussian noise: sigma ~ Uniform(5, 25)
# + Motion blur: kernel 3/5/7, random angle, p=0.5
# + Sepia fading: sepia_matrix @ rgb, p=0.6
```

### 📚 Skill Matrix — Phase 1

> Những gì bạn cần hiểu để **review và debug** code ở phase này:

**NumPy (bắt buộc):**
- `np.stack([a, b, c], axis=-1)` → stack arrays theo chiều cuối
- `np.linalg.norm(x, axis=-1, keepdims=True)` → tính độ dài vector, giữ shape
- Broadcasting: `(H, W, 3) * (H, W, 1)` → multiply từng channel độc lập
- `np.clip(x, 0, 1)` → cắt giá trị về khoảng [0, 1]
- Indexing: `arr[y:y+h, x:x+w]` → crop vùng ảnh

**OpenCV:**
- `cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)` → gradient theo x, kết quả float64
- `cv2.GaussianBlur(img, (0,0), sigmaX=3)` → blur với sigma thay vì kernel size
- `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` → chuyển sang grayscale
- `cv2.resize(img, (new_w, new_h))` → lưu ý: resize nhận (width, height)!

**Kiểm tra nhanh:**
```python
# Test normal map output hợp lệ
normal = compute_normal_map(heightmap)
assert normal.shape == (*heightmap.shape, 3)      # phải có 3 channels
norms = np.linalg.norm(normal, axis=-1)
assert np.allclose(norms, 1.0, atol=1e-5)         # phải là unit vectors
```

---

## Phase 2 — Crack Segmentation (Module 1)
**Thời gian:** 2 tuần | **GPU cần:** Kaggle T4 (free)

### Mục tiêu
Huấn luyện mạng U-Net nhận ảnh suy thoái → dự đoán binary crack mask với IoU > 0.75.

### Tasks

```
[ ] 2.1  Viết src/models/attention_gate.py
         - Class AttentionGate(nn.Module)
         - Input: (x, gating_signal) → output: x * attention_weights
         - Gate: sigmoid(Wg*g + Wx*x + b)

[ ] 2.2  Viết src/models/segmenter.py
         - Class CrackSegmenter(pl.LightningModule)
         - Encoder: ResNet-34 pretrained
         - Decoder: 4 UpBlock với AttentionGate ở mỗi skip connection
         - Head: Conv(1) + Sigmoid
         - Deep supervision: thêm auxiliary heads ở scale /4 và /2

[ ] 2.3  Viết src/losses/segmentation.py
         - class DiceLoss(nn.Module): Dice = 2|A∩B| / (|A|+|B|)
         - class SegmentationLoss: BCE + DiceLoss
         - Lưu ý: input là logits hay probability? Document rõ!

[ ] 2.4  Viết src/data/dataset.py (phần segmentation)
         - class CrackSegDataset(Dataset)
         - __getitem__ trả về: {'image': tensor[3,H,W], 'mask': tensor[1,H,W]}
         - Augmentation: flip, rotate (dùng albumentations)

[ ] 2.5  Viết scripts/train_segmentation.py
         - Load config từ configs/segmenter.yaml
         - Init W&B run: wandb.init(project=cfg.wandb.project)
         - Trainer với: early_stopping, model_checkpoint (monitor IoU)
         - Log: train/loss, val/loss, val/iou, val/f1 mỗi epoch
         - Log ảnh: wandb.log({"examples": [wandb.Image(pred_vis)]})

[ ] 2.6  Chạy training trên Kaggle
         - Upload code + data lên Kaggle Dataset
         - Dùng Accelerator: "gpu", devices: 1
         - Lưu checkpoint mỗi 5 epochs → Google Drive

[ ] 2.7  Evaluate trong notebooks/03_segmentation_eval.ipynb
         - Precision-Recall curve
         - Confusion matrix
         - Visualize: input | pred_mask | gt_mask | overlay
```

### 📚 Skill Matrix — Phase 2

> Những gì bạn cần hiểu để **review code AI sinh ra** ở phase này:

**PyTorch Tensor operations:**
- `torch.cat([a, b], dim=1)` → concat theo channel dimension
- `.permute(0, 2, 3, 1)` → BCHW → BHWC (khi cần numpy visualization)
- `F.interpolate(x, scale_factor=2, mode='bilinear')` → upsample
- `tensor.detach().cpu().numpy()` → chuyển tensor về numpy để visualize

**PyTorch Lightning:**
- `training_step(batch, batch_idx)` → return loss
- `validation_step(batch, batch_idx)` → log metrics
- `configure_optimizers()` → return optimizer (và scheduler nếu cần)
- `self.log("val/iou", iou, on_epoch=True, prog_bar=True)` → log metric

**Khái niệm quan trọng:**
- **Skip connection** trong U-Net là gì và tại sao cần
- **Sigmoid vs Softmax:** segmentation nhị phân dùng Sigmoid (1 output channel)
- **Class imbalance:** tại sao Dice Loss quan trọng khi crack < 5% pixels
- **Attention Gate:** trực giác — giống "spotlight" chỉ sáng vùng crack

**Kiểm tra nhanh:**
```python
# Test forward pass không lỗi
model = CrackSegmenter()
dummy = torch.randn(2, 3, 512, 512)   # batch of 2
out = model(dummy)
assert out.shape == (2, 1, 512, 512)  # phải trả về 1 channel mask
assert out.min() >= 0 and out.max() <= 1  # Sigmoid output ∈ [0,1]
```

---

## Phase 3 — LaMa Inpainting (Module 2)
**Thời gian:** 3–4 tuần | **GPU cần:** Kaggle T4 hoặc RunPod RTX 3090**

### Mục tiêu
Fine-tune LaMa pretrained trên dataset tự tạo. Target: LPIPS < 0.2 trên val set.

### Tasks

```
[ ] 3.1  Download LaMa pretrained checkpoint
         git clone https://github.com/advimman/lama
         # Tải big-lama checkpoint (~416MB)

[ ] 3.2  Viết src/models/ffc.py (nếu không dùng LaMa repo trực tiếp)
         - Class FourierUnit(nn.Module): FFT → Conv → IFFT
         - Class FFC(nn.Module): local branch + global (FourierUnit) branch
         - Class FFCResBlock(nn.Module): 2x FFC + residual

[ ] 3.3  Viết src/models/lama_gan.py
         - Class LamaGenerator(pl.LightningModule): fine-tune LaMa
         - Class PatchDiscriminator(nn.Module): PatchGAN 70×70
         - Tải pretrained weights: model.load_state_dict(torch.load(ckpt))

[ ] 3.4  Viết src/losses/restoration.py
         - class PerceptualLoss: extract VGG-19 features (conv3_3, conv4_3)
         - class AdversarialLoss: hinge loss cho GAN
         - class RestorationLoss: tổng hợp λ1·L1 + λ2·Perceptual + λ3·Adversarial

[ ] 3.5  Viết src/data/dataset.py (phần restoration)
         - class RestorationDataset(Dataset)
         - __getitem__: {'image': tensor[3,H,W], 'mask': tensor[1,H,W],
                          'gt': tensor[3,H,W]}
         - Input lấy từ data/processed/, GT là ảnh sạch gốc

[ ] 3.6  Viết scripts/train_restoration.py
         - Dùng fp16: Trainer(precision="16-mixed")
         - Gradient checkpointing: model.gradient_checkpointing_enable()
         - Log: train/loss_total, train/loss_l1, train/loss_perceptual,
                val/lpips, val/psnr
         - Lưu checkpoint theo val/lpips (monitor="val/lpips", mode="min")

[ ] 3.7  Evaluate trong notebooks/04_restoration_eval.ipynb
         - So sánh: OpenCV baseline vs LaMa fine-tuned
         - Metric table: LPIPS, FID, PSNR, SSIM
         - Visual comparison: 10 ảnh test set
```

### 📚 Skill Matrix — Phase 3

> Những gì bạn cần hiểu để **review code AI sinh ra** ở phase này:

**GAN training (QUAN TRỌNG):**
- GAN cần **2 optimizer**: 1 cho Generator, 1 cho Discriminator
- Trong PyTorch Lightning: `configure_optimizers()` trả về `([opt_G, opt_D], [])`
- Training step luân phiên: step G → step D → step G → ...
- **Mode collapse** là gì và dấu hiệu nhận biết (val loss đột ngột giảm 0)

**Perceptual Loss:**
```python
# Hiểu cách extract intermediate features từ VGG:
vgg = torchvision.models.vgg19(pretrained=True).features
# Lấy features tại layer 16 (conv3_3) và layer 25 (conv4_3)
# So sánh bằng L1/MSE trong feature space, không phải pixel space
```

**Mixed Precision (fp16):**
- `Trainer(precision="16-mixed")` → tự động dùng FP16 cho forward, FP32 cho grad
- Giảm VRAM ~50%, tăng tốc ~1.5x trên T4
- Rủi ro: gradient underflow → Lightning tự xử lý bằng GradScaler

**Fast Fourier Convolutions:**
- Trực giác: CNN 3×3 chỉ nhìn được vùng 3×3 xung quanh
- FFC: biến đổi sang frequency domain → mỗi "pixel" trong freq space ảnh hưởng toàn bộ ảnh
- Kết quả: texture coherence tốt hơn nhiều khi lấp vết nứt dài

**Kiểm tra nhanh:**
```python
# Test LaMa forward pass
model = LamaGenerator()
image = torch.randn(1, 3, 512, 512)   # ảnh suy thoái
mask  = torch.randint(0, 2, (1, 1, 512, 512)).float()
inp = torch.cat([image, mask], dim=1)  # 4 channels
out = model(inp)
assert out.shape == (1, 3, 512, 512)   # RGB output
```

---

## Phase 4 — Tích hợp CodeFormer (Module 3)
**Thời gian:** 3–5 ngày | **GPU cần:** T4 (inference only)

### Tasks

```
[ ] 4.1  Clone CodeFormer repo và tải pretrained weights
         git clone https://github.com/sczhou/CodeFormer
         # Tải: codeformer.pth (~375MB)

[ ] 4.2  Viết face detection wrapper trong src/utils/
         - dùng retina-face: pip install retina-face
         - Hàm detect_faces(img) → list of bounding boxes

[ ] 4.3  Viết scripts/infer.py — full pipeline
         - Load ảnh → resize → normalize
         - Module 1: predict crack mask
         - Module 2: LaMa inpainting
         - Detect faces → if found: Module 3 (CodeFormer)
         - De-normalize → save output
         - Hỗ trợ batch inference (folder input)

[ ] 4.4  Test với 10 ảnh trong data/test/
         - So sánh: before / after với từng module
         - Lưu: comparison grid (4 ảnh: input | mask | inpainted | final)
```

### 📚 Skill Matrix — Phase 4

- **Python subprocess / module import:** cách import code từ repo ngoài không cài pip
- **PIL / torchvision transforms:** chuyển đổi giữa numpy uint8, PIL Image, torch tensor
- **Bounding box crop:** `img[y1:y2, x1:x2]` và resize về (512, 512) cho CodeFormer

---

## Phase 5 — Evaluation, Demo & Báo cáo
**Thời gian:** 1.5 tuần

### Tasks

```
[ ] 5.1  Ablation study (bắt buộc cho báo cáo)
         So sánh 4 cấu hình:
         A. Baseline: OpenCV Navier-Stokes inpainting
         B. Module 1 + OpenCV inpainting (AI mask, traditional inpainting)
         C. Module 1 + Module 2 (AI mask + LaMa)
         D. Module 1 + 2 + 3 (full pipeline)
         Metric: LPIPS, FID, PSNR, SSIM trên test set 200 ảnh

[ ] 5.2  Tạo Gradio demo (gradio==4.x)
         - Input: upload ảnh cũ
         - Output: ảnh phục hồi + crack mask visualization
         - Slider: chọn CodeFormer fidelity weight (0.0–1.0)
         - Nút: "Download result"

[ ] 5.3  Viết README.md
         - GIF demo (trước/sau)
         - Ablation table
         - Quick start (3 lệnh để chạy)
         - Link pretrained checkpoints (Google Drive / HuggingFace)
```

### 📚 Skill Matrix — Phase 5

- **Gradio cơ bản:** `gr.Interface`, `gr.Image`, `gr.Slider`
- **Matplotlib grid:** `plt.subplot`, so sánh nhiều ảnh trong 1 figure
- **Pandas:** tạo bảng kết quả metric, export CSV

---

## Terminal Commands — Cài đặt môi trường

### Kiểm tra môi trường

```bash
# Kiểm tra Python
python --version
python3 --version

# Kiểm tra Conda
conda --version

# Kiểm tra GPU (nếu có NVIDIA GPU)
nvidia-smi

# Kiểm tra PyTorch và GPU
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Tạo và kích hoạt môi trường

```bash
# Tạo môi trường mới tên là old_photo, Python 3.10
conda create -n old_photo python=3.10 -y

# Kích hoạt
conda activate old_photo

# Kiểm tra đang dùng đúng environment
conda info --envs    # dấu * là env đang active
which python         # trên macOS/Linux
where python         # trên Windows
```

### Cài đặt dependencies

```bash
# Cài PyTorch với CUDA (dùng trên máy có GPU NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hoặc nếu không có GPU (CPU only):
pip install torch torchvision torchaudio

# Cài requirements.txt
pip install -r requirements.txt
```

### Nội dung `requirements.txt`

```txt
# Core DL
torch>=2.0.0
torchvision>=0.15.0
pytorch-lightning>=2.0.0

# Image processing
opencv-python>=4.8.0
Pillow>=10.0.0
albumentations>=1.3.0

# Metrics
lpips>=0.1.4
torch-fidelity>=0.3.0
torchmetrics>=1.0.0
scikit-image>=0.21.0

# Config & Experiment tracking
omegaconf>=2.3.0
wandb>=0.15.0

# Face detection
retina-face>=0.0.14

# Data & Utilities
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
pandas>=2.0.0

# Notebook
jupyter>=1.0.0
ipywidgets>=8.0.0

# Demo
gradio>=4.0.0

# Dev
pytest>=7.4.0
```

### Các lệnh hữu ích thường dùng

```bash
# Xem danh sách packages đã cài
pip list

# Cập nhật 1 package
pip install --upgrade wandb

# Đăng nhập W&B (lần đầu)
wandb login

# Chạy build dataset
python scripts/build_dataset.py --config configs/data.yaml

# Chạy training segmenter
python scripts/train_segmentation.py --config configs/segmenter.yaml

# Chạy inference
python scripts/infer.py --input data/test/ --output results/ --config configs/data.yaml

# Xem GPU usage trong khi train
watch -n 1 nvidia-smi   # Linux/macOS

# Deactivate môi trường khi xong
conda deactivate
```

---

## Checklist Trước Khi Push Code Lên GitHub

```
[ ] Không có API key hay password nào trong code
[ ] Không có đường dẫn tuyệt đối (vd: /home/username/...) trong code
[ ] Tất cả hyperparameters đều trong configs/*.yaml, không hard-code
[ ] Đã chạy tests/ thành công: pytest tests/
[ ] Đã update README.md nếu thay đổi cách chạy
[ ] File .gitignore đã bỏ qua: data/, checkpoints/, *.ckpt, wandb/
```
