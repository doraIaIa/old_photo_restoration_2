# ARCHITECTURE.md

# Tài liệu Kiến trúc Hệ thống — Blueprint 2.1

# Old Photo Restoration · SOTA Pipeline

> **Mục đích của file này:** Đây là "bản đồ hệ thống" duy nhất. Mọi AI assistant (Codex, Cursor, Claude) đều phải đọc file này trước khi sinh code. Không được hard-code bất kỳ path hay config nào nằm ngoài `configs/`.

---

## 1. Tổng quan hệ thống (System Overview)

Hệ thống giải quyết bài toán **Mixed Degradation** gồm 2 nhóm:

| Loại                                               | Đặc điểm                          | Module xử lý                  |
| --------------------------------------------------- | ------------------------------------- | ------------------------------- |
| **Unstructured** — noise, blur, color fading | Phân bố đều, toàn ảnh           | Module 2 (LaMa + Restoration)   |
| **Structured** — cracks, scratches, tears    | Cục bộ, có tính chất vật lý 3D | Module 1 → Module 2 phối hợp |

**Triết lý thiết kế:** Divide & Conquer — mỗi module chuyên biệt một nhiệm vụ, kết nối qua interface rõ ràng. Không dùng 1 mạng duy nhất cho tất cả.

---

## 2. Pipeline End-to-End (Data Flow)

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT                                      │
│              Ảnh cũ bất kỳ (RGB, kích thước tùy ý)             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TIỀN XỬ LÝ                                    │
│   • Resize về 512×512 (hoặc bội số của 8)                       │
│   • Normalize pixel values về [0, 1]                            │
│   • Convert sang tensor PyTorch (C, H, W)                       │
│   Output: tensor float32 shape [1, 3, 512, 512]                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODULE 1 — CRACK SEGMENTATION                      │
│                                                                 │
│   Kiến trúc: U-Net + Attention Gate + Deep Supervision          │
│   Encoder  : ResNet-34 (pretrained ImageNet)                    │
│   Decoder  : 4 upsampling blocks + Attention Gate ở mỗi skip   │
│   Supervision: Loss tại scale ×1, ×2, ×4 (deep supervision)    │
│                                                                 │
│   ┌──────────┐    ┌──────────────────┐    ┌──────────────┐     │
│   │ResNet-34 │───▶│Attention Gate ×4 │───▶│ Sigmoid Head │     │
│   │ Encoder  │    │    Decoder       │    │  (1 channel) │     │
│   └──────────┘    └──────────────────┘    └──────┬───────┘     │
│                                                  │             │
│   Loss = BCE + Dice (chống class imbalance)      │             │
│   Metric: IoU, F1-Score, Precision, Recall       │             │
│                                                  │             │
│   Output: Binary Mask [1, 1, 512, 512] ∈ {0, 1} │             │
└──────────────────────────────────────────────────┼─────────────┘
                          │                        │ mask
                          │◀───────────────────────┘
                          │
                          │  concat([image, mask], dim=1)
                          │  → tensor [1, 4, 512, 512]
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│         MODULE 2 — LaMa INPAINTING + GLOBAL RESTORATION        │
│                                                                 │
│   Kiến trúc: LaMa (Large Mask inpainting)                      │
│   Core    : Fast Fourier Convolutions (FFC) + ResBlocks         │
│   FFC     : Biến đổi sang frequency domain → global receptive  │
│             field trong 1 phép tính → xử lý tốt crack dài      │
│                                                                 │
│   ┌─────────────────────────────────────────────┐              │
│   │  [Image ‖ Mask] (4ch) ──▶ FFC Blocks ──▶    │              │
│   │  ResBlocks ──▶ Decoder ──▶ Restored RGB      │              │
│   └─────────────────────────────────────────────┘              │
│                           ║                                     │
│   ┌───────────────────────╨──────────────────────┐             │
│   │         PatchGAN DISCRIMINATOR               │             │
│   │  Phân biệt patch 70×70 thật/giả              │             │
│   └──────────────────────────────────────────────┘             │
│                                                                 │
│   Loss = λ₁·L1(1.0) + λ₂·Perceptual/VGG-19(0.1)               │
│          + λ₃·Adversarial/PatchGAN(0.01)                       │
│   Pretrained: advimman/lama (fine-tune trên data tự tạo)        │
│                                                                 │
│   Output: Restored RGB [1, 3, 512, 512]                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  RetinaFace Detector  │
              │  Có khuôn mặt?        │
              └────────┬──────┬───────┘
                    CÓ │      │ KHÔNG
                       │      │
                       ▼      ▼
┌─────────────────────────┐  ┌─────────────────────┐
│  MODULE 3 — CODEFORMER  │  │  Bỏ qua Module 3    │
│                         │  │  → Thẳng ra output   │
│  VQVAE Codebook lookup  │  └──────────┬──────────┘
│  (~1024 face parts)     │             │
│  Transformer decoder    │             │
│  → Natural face output  │             │
│                         │             │
│  Mode: INFERENCE ONLY   │             │
│  Không cần train        │             │
└────────────┬────────────┘             │
             │                         │
             └────────────┬────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT                                     │
│         Ảnh phục hồi hoàn chỉnh — RGB 512×512                  │
│         De-normalize về [0, 255], save PNG/JPEG                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Interface giữa các Module (Contracts)

> **Quy tắc vàng:** Mọi tensor truyền giữa các module phải được document rõ shape và range. Không được assume.

```python
# Module 1 → Module 2
# Input :  image_tensor  : torch.Tensor [B, 3, H, W], dtype=float32, range=[0,1]
# Output:  mask_tensor   : torch.Tensor [B, 1, H, W], dtype=float32, range={0,1}
# Concat:  lama_input    : torch.cat([image_tensor, mask_tensor], dim=1)
#          → shape [B, 4, H, W]

# Module 2 → Module 3
# Input :  restored_img  : torch.Tensor [B, 3, H, W], dtype=float32, range=[0,1]
# Output:  final_img     : numpy.ndarray [H, W, 3], dtype=uint8, range=[0,255]

# Kích thước H, W phải là bội số của 8 (yêu cầu của LaMa)
# Mặc định: H=W=512
```

---

## 4. Cấu trúc thư mục dự án

```
old_photo_restoration_2/
│
├── ARCHITECTURE.md          ← File này — đọc đầu tiên
├── ROADMAP_AND_TASKS.md     ← Kế hoạch thực thi từng phase
├── SETUP_ENV.md             ← Hướng dẫn cài đặt môi trường
├── requirements.txt         ← Toàn bộ dependencies
├── .gitignore
├── README.md
│
├── configs/                 ← TẤT CẢ hyperparameters ở đây, không hard-code
│   ├── data.yaml            ← Paths, augmentation params, dataset splits
│   ├── segmenter.yaml       ← U-Net config: lr, batch_size, epochs, loss weights
│   ├── generator.yaml       ← LaMa config: lambda L1/perceptual/adversarial
│   └── wandb.yaml           ← W&B project name, entity, tags
│
├── data/
│   ├── crack_bank/          ← Ảnh vết nứt thật đã crop + alpha mask
│   │   ├── raw/             │   ∙ Mỗi crack cần: crack_001.png (RGBA)
│   │   └── processed/       │   ∙ Sau augment: ~10,000 biến thể
│   ├── clean/               ← Ảnh sạch nguồn (chưa suy thoái)
│   │   ├── div2k/           │   ∙ DIV2K: 800 train + 100 val ảnh
│   │   └── ffhq/            │   ∙ FFHQ subset: 10,000 ảnh khuôn mặt
│   ├── processed/           ← Dataset đã qua Degradation Pipeline
│   │   ├── train/
│   │   │   ├── images/      │   ∙ Ảnh suy thoái (input cho model)
│   │   │   └── masks/       │   ∙ Binary mask GT (ground truth)
│   │   └── val/
│   │       ├── images/
│   │       └── masks/
│   └── test/                ← Ảnh cũ thật để demo cuối
│
├── src/                     ← Source code cốt lõi (library, không chạy trực tiếp)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── degradation.py   ← ⭐ QUAN TRỌNG NHẤT: Normal Mapping 3D + Alpha Blend
│   │   ├── dataset.py       ← PyTorch Dataset class cho cả 2 module
│   │   └── transforms.py    ← Albumentations pipeline
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── segmenter.py     ← U-Net + Attention Gate + Deep Supervision
│   │   ├── attention_gate.py← Attention Gate module (dùng lại trong segmenter)
│   │   ├── lama_gan.py      ← LaMa Generator + PatchGAN Discriminator
│   │   └── ffc.py           ← Fast Fourier Convolution layers
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── segmentation.py  ← BCEWithLogitsLoss + DiceLoss
│   │   └── restoration.py   ← L1 + PerceptualLoss(VGG-19) + AdversarialLoss
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py       ← LPIPS, FID wrapper, PSNR, SSIM
│       ├── visualize.py     ← Hàm plot kết quả, log ảnh lên W&B
│       └── checkpoint.py    ← Save/load checkpoint helper
│
├── scripts/                 ← Entry points — chạy từ terminal
│   ├── build_dataset.py     ← Chạy Degradation Pipeline → tạo data/processed/
│   ├── train_segmentation.py← Huấn luyện Module 1
│   ├── train_restoration.py ← Fine-tune Module 2 (LaMa)
│   └── infer.py             ← Inference full pipeline: ảnh cũ → ảnh phục hồi
│
├── notebooks/               ← Jupyter notebooks cho EDA, không dùng để train
│   ├── 01_crack_bank_eda.ipynb     ← Xem thống kê Crack Bank
│   ├── 02_degradation_preview.ipynb← Xem ví dụ output của degradation.py
│   ├── 03_segmentation_eval.ipynb  ← Vẽ Precision-Recall curve, IoU
│   └── 04_restoration_eval.ipynb   ← So sánh LPIPS/FID các cấu hình
│
├── experiments/             ← W&B sync và log files (gitignore phần lớn)
│   └── .gitkeep
│
├── checkpoints/             ← Model weights (gitignore, dùng DVC nếu cần)
│   ├── segmenter/
│   │   └── best_iou.ckpt
│   └── lama_finetuned/
│       └── best_lpips.ckpt
│
└── tests/                   ← Unit tests
    ├── test_degradation.py  ← Test Normal Map output shape, value range
    ├── test_dataset.py      ← Test DataLoader không bị lỗi
    └── test_metrics.py      ← Test LPIPS/FID cho kết quả hợp lệ
```

---

## 5. Quy chuẩn công nghệ (Tech Standards)

### 5.1 Framework

| Thành phần        | Công cụ                          | Ghi chú                                |
| ------------------- | ---------------------------------- | --------------------------------------- |
| Deep Learning       | **PyTorch ≥ 2.0**           | Bắt buộc                              |
| Training loop       | **PyTorch Lightning ≥ 2.0** | Giảm boilerplate, tự động multi-GPU |
| Augmentation        | **Albumentations ≥ 1.3**    | Nhanh hơn torchvision transforms       |
| Config management   | **OmegaConf / YAML**         | Tất cả hyperparams trong `configs/` |
| Experiment tracking | **Weights & Biases (W&B)**   | Bắt buộc — log loss, images, metrics |
| Image processing    | **OpenCV + Pillow**          | OpenCV cho degradation, Pillow cho I/O  |
| Face detection      | **RetinaFace**               | `pip install retina-face`             |

### 5.2 Metrics

| Metric          | Loại        | Thư viện                     | Mục đích                            |
| --------------- | ------------ | ------------------------------ | -------------------------------------- |
| **LPIPS** | Primary ↓   | `pip install lpips`          | Perceptual similarity — metric chính |
| **FID**   | Primary ↓   | `pip install torch-fidelity` | Distribution quality — metric chính  |
| PSNR            | Reference ↑ | `skimage.metrics`            | Chỉ so sánh với baseline            |
| SSIM            | Reference ↑ | `skimage.metrics`            | Chỉ so sánh với baseline            |
| IoU             | Seg ↑       | custom                         | Đánh giá Module 1                   |
| F1-Score        | Seg ↑       | `torchmetrics`               | Đánh giá Module 1                   |

### 5.3 Quy tắc code

```python
# ✅ ĐÚNG — config từ YAML, không hard-code
from omegaconf import OmegaConf
cfg = OmegaConf.load("configs/segmenter.yaml")
lr = cfg.optimizer.lr

# ❌ SAI — hard-code config
lr = 0.0001  # KHÔNG ĐƯỢC LÀM THẾ NÀY

# ✅ ĐÚNG — tensor shape luôn được comment
x = encoder(image)   # [B, 512, H/16, W/16]

# ✅ ĐÚNG — log metrics lên W&B
import wandb
wandb.log({"val/lpips": lpips_score, "val/iou": iou_score}, step=epoch)
```

### 5.4 Checkpoint naming convention

```
checkpoints/
  segmenter/
    epoch=023_iou=0.847.ckpt      ← format: epoch=NNN_metric=X.XXX.ckpt
    best_iou.ckpt                 ← symlink tới best checkpoint
  lama_finetuned/
    epoch=041_lpips=0.143.ckpt
    best_lpips.ckpt
```

---

## 6. Nguồn pretrained models & datasets

```yaml
# Datasets
DIV2K:      https://data.vision.ee.ethz.ch/cvl/DIV2K/
FFHQ:       https://github.com/NVlabs/ffhq-dataset
CrackForest: https://github.com/cuilimeng/CrackForest-dataset

# Pretrained Models
LaMa:       https://github.com/advimman/lama
            # Tải: big-lama checkpoint (416MB)
CodeFormer: https://github.com/sczhou/CodeFormer
ResNet-34:  torchvision.models.resnet34(weights='IMAGENET1K_V1')
VGG-19:     torchvision.models.vgg19(weights='IMAGENET1K_V1')  # frozen
```

---

## 7. Các quyết định kiến trúc quan trọng (ADR)

### ADR-001: Tại sao U-Net thay vì Mask2Former?

- Mask2Former thiết kế cho object instance segmentation (có bounding box)
- Vết nứt là **thin tubular structure** — không có bounding box
- U-Net + Attention Gate + Deep Supervision phù hợp hơn cho thin line segmentation
- Tham số ~25M so với ~47M của Mask2Former → nhẹ hơn, phù hợp T4 GPU

### ADR-002: Tại sao LaMa thay vì Restormer?

- Restormer là **restoration** network (denoise/deblur), không có cơ chế masked region
- LaMa được thiết kế đặc biệt cho **large mask inpainting**
- Fast Fourier Convolutions trong LaMa tự nhiên phù hợp với crack dài

### ADR-003: Tại sao LPIPS + FID là primary metrics?

- Inpainting tạo ra texture hallucinated → sẽ khác GT ở pixel-level
- PSNR/SSIM đo pixel-level → cho điểm thấp dù kết quả tốt về mặt thị giác
- LPIPS đo qua feature space → phản ánh "nhìn có giống nhau không"
- FID đo phân phối thống kê tổng thể → phản ánh chất lượng tập kết quả

### ADR-004: Tại sao Normal Mapping 3D cho Degradation Pipeline?

- `cv2.line` tạo vết nứt phẳng 2D → domain gap lớn với vết nứt thật
- Vết nứt thật có bóng đổ (shadow) và viền sáng (highlight) do tính chất 3D
- Pipeline: heightmap → normal map → Phong illumination → realistic crack
- Kết quả: mask ground truth chính xác 100% + domain gap tối thiểu
