# SETUP_ENV.md
# Cẩm nang Cài đặt Môi trường từ A–Z
# Blueprint 2.1 — Old Photo Restoration

> **Dành cho:** Người mới, chưa từng cài đặt môi trường Python cho Deep Learning.
> **Thời gian ước tính:** 30–60 phút tùy tốc độ mạng.

---

## Bước 0 — Kiểm tra máy hiện tại

Mở **Terminal** (macOS/Linux) hoặc **Command Prompt / PowerShell** (Windows) và chạy từng lệnh sau:

```bash
# Kiểm tra Python đã có chưa
python --version
# hoặc
python3 --version
```

```bash
# Kiểm tra Conda đã có chưa
conda --version
```

```bash
# Kiểm tra GPU NVIDIA (nếu bạn có GPU rời)
nvidia-smi
```

### Đọc kết quả:

| Kết quả bạn thấy | Ý nghĩa | Việc cần làm |
|------------------|---------|--------------|
| `Python 3.10.x` hoặc cao hơn | Python đã có | Kiểm tra tiếp Conda |
| `Python 3.8.x` hoặc thấp hơn | Python quá cũ | Cài Miniconda (sẽ bao gồm Python mới) |
| `command not found` / `'python' is not recognized` | Chưa có Python | Cài Miniconda |
| `conda 23.x.x` hoặc tương tự | Conda đã có | Bỏ qua Bước 1, qua Bước 2 |
| `command not found` | Chưa có Conda | Làm Bước 1 |
| GPU info hiện ra (Driver Version, CUDA Version) | Có GPU NVIDIA | Ghi lại CUDA Version để cài PyTorch đúng |
| `command not found` hoặc không có gì | Không có GPU NVIDIA | Dùng CPU hoặc Kaggle/Colab |

---

## Bước 1 — Cài đặt Miniconda

> **Miniconda** là phiên bản tối giản của Anaconda — nhẹ hơn nhiều (~60MB so với ~3GB), phù hợp cho Deep Learning.

### Tải Miniconda

Truy cập: **https://docs.conda.io/en/latest/miniconda.html**

Chọn đúng file cho hệ điều hành của bạn:

| Hệ điều hành | File cần tải |
|--------------|-------------|
| **Windows** (64-bit) | `Miniconda3-latest-Windows-x86_64.exe` |
| **macOS** Intel (chip cũ) | `Miniconda3-latest-MacOSX-x86_64.pkg` |
| **macOS** Apple Silicon (M1/M2/M3) | `Miniconda3-latest-MacOSX-arm64.pkg` |
| **Linux** (64-bit) | `Miniconda3-latest-Linux-x86_64.sh` |

### Cài đặt

**Windows:**
1. Double-click file `.exe` vừa tải
2. Nhấn Next → Agree → Just Me → Next
3. ⚠️ **Quan trọng:** Tick chọn **"Add Miniconda3 to my PATH environment variable"**
4. Install → Finish
5. Mở **Anaconda Prompt** (tìm trong Start Menu)

**macOS:**
1. Double-click file `.pkg` → làm theo hướng dẫn
2. Mở Terminal mới (Terminal app)

**Linux:**
```bash
# Chạy installer
bash Miniconda3-latest-Linux-x86_64.sh

# Đọc license → gõ 'yes' để đồng ý
# Chọn install location → Enter để dùng mặc định
# "Do you wish the installer to initialize Miniconda3?" → gõ 'yes'

# Áp dụng thay đổi (hoặc đóng terminal và mở lại)
source ~/.bashrc
```

### Kiểm tra cài đặt thành công

```bash
conda --version
# Kết quả mong đợi: conda 23.x.x (hoặc tương tự)

python --version
# Kết quả mong đợi: Python 3.10.x (hoặc 3.11.x)
```

---

## Bước 2 — Tạo Virtual Environment

> **Virtual environment (môi trường ảo)** là một "hộp cát" cô lập — mỗi project có Python và libraries riêng, không ảnh hưởng lẫn nhau.

```bash
# Tạo môi trường mới tên 'old_photo' với Python 3.10
conda create -n old_photo python=3.10 -y

# Quá trình tải mất 1–3 phút tùy mạng
```

```bash
# Kích hoạt môi trường
conda activate old_photo

# Dấu hiệu kích hoạt thành công: tên env xuất hiện trong ngoặc
# (old_photo) username@computer:~$
```

```bash
# Kiểm tra Python trong environment
python --version
# Phải ra: Python 3.10.x

# Kiểm tra pip
pip --version
# Phải ra: pip 23.x ... (old_photo)  ← phải thấy tên env ở đây
```

> 💡 **Lưu ý:** Mỗi lần mở Terminal mới, bạn phải **kích hoạt lại** môi trường:
> ```bash
> conda activate old_photo
> ```

---

## Bước 3 — Clone dự án và cài Dependencies

### Clone repo

```bash
# Di chuyển về thư mục bạn muốn lưu dự án
cd ~/Documents        # macOS/Linux
# hoặc
cd C:\Users\TenBan\Documents   # Windows

# Clone repo (thay bằng URL repo của bạn)
git clone https://github.com/username/old-photo-restoration-v2.git
cd old-photo-restoration-v2
```

### Cài PyTorch

Truy cập **https://pytorch.org/get-started/locally/** để lấy lệnh cài đúng cho máy bạn.

**Nếu có GPU NVIDIA (khuyến nghị):**
```bash
# CUDA 11.8 (xem CUDA version từ nvidia-smi ở Bước 0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Nếu không có GPU (CPU only):**
```bash
pip install torch torchvision torchaudio
```

**Nếu dùng Mac Apple Silicon (M1/M2/M3):**
```bash
pip install torch torchvision torchaudio
# PyTorch tự động dùng Metal Performance Shaders (MPS)
```

### Kiểm tra PyTorch cài đúng

```bash
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
# Mac M1/M2:
print('MPS available:', torch.backends.mps.is_available())
"
```

### Cài toàn bộ requirements

```bash
# Đảm bảo đang trong thư mục dự án và old_photo env đang active
pip install -r requirements.txt

# Quá trình này mất 5–15 phút tùy mạng
# Nếu bị lỗi một package nào đó, xem phần Xử lý lỗi ở cuối file
```

---

## Bước 4 — Cài đặt thêm (Tùy chọn nhưng quan trọng)

### Đăng ký và đăng nhập W&B (Weights & Biases)

```bash
# 1. Tạo tài khoản miễn phí tại: https://wandb.ai/
# 2. Lấy API key tại: https://wandb.ai/settings (mục Danger Zone → API keys)
# 3. Đăng nhập:
wandb login
# Dán API key khi được hỏi → Enter
```

### Cài Jupyter Notebook kernel

```bash
# Đăng ký kernel cho environment này (để dùng trong VS Code / JupyterLab)
pip install ipykernel
python -m ipykernel install --user --name old_photo --display-name "Python (old_photo)"
```

### Tải pretrained models

```bash
# Tạo thư mục chứa pretrained weights
mkdir -p checkpoints/lama_pretrained
mkdir -p checkpoints/codeformer

# LaMa pretrained (big-lama, ~416MB)
# Tải từ: https://github.com/advimman/lama/releases
# Lưu vào: checkpoints/lama_pretrained/big-lama.pt

# CodeFormer pretrained (~375MB)
# Tải từ: https://github.com/sczhou/CodeFormer/releases
# Lưu vào: checkpoints/codeformer/codeformer.pth
```

---

## Bước 5 — Kiểm tra toàn bộ setup

```bash
# Test import tất cả packages quan trọng
python -c "
import torch
import torchvision
import pytorch_lightning as pl
import cv2
import numpy as np
import albumentations as A
import lpips
import wandb
from omegaconf import OmegaConf
print('✅ Tất cả packages đã sẵn sàng!')
print(f'  PyTorch: {torch.__version__}')
print(f'  Lightning: {pl.__version__}')
print(f'  OpenCV: {cv2.__version__}')
print(f'  NumPy: {np.__version__}')
"
```

```bash
# Chạy unit tests để kiểm tra code cốt lõi
pytest tests/ -v

# Kết quả mong đợi: tất cả tests PASSED
```

---

## Bước 6 — Thiết lập cho Kaggle (Train trên cloud miễn phí)

> Kaggle cung cấp **30 giờ GPU T4 miễn phí mỗi tuần** — đủ để train Module 1.

### Upload code lên Kaggle

```bash
# Cài Kaggle API
pip install kaggle

# Tải API token từ: https://www.kaggle.com/settings (Account → API → Create New Token)
# Lưu file kaggle.json vào:
#   macOS/Linux: ~/.kaggle/kaggle.json
#   Windows: C:\Users\TenBan\.kaggle\kaggle.json
```

### Tạo Kaggle Notebook

1. Vào https://www.kaggle.com/code → **New Notebook**
2. Settings → Accelerator: **GPU T4 x2**
3. Settings → Internet: **On** (để pip install)
4. Upload code của bạn qua Kaggle Dataset hoặc dùng Git:

```python
# Trong Kaggle Notebook, cell đầu tiên:
!git clone https://github.com/username/old-photo-restoration-v2.git
%cd old-photo-restoration-v2
!pip install -r requirements.txt -q
```

### Lưu checkpoint tự động lên Google Drive

```python
# Thêm vào đầu training script khi chạy trên Kaggle:
import os
from google.colab import drive  # hoặc dùng Kaggle Secrets

# Hoặc đơn giản hơn: dùng W&B Artifacts để lưu checkpoint
import wandb
wandb.init(project="old-photo-restoration")
# Sau mỗi epoch, checkpoint tự động sync lên W&B cloud
```

---

## Xử lý lỗi thường gặp

### Lỗi: `pip install` bị lỗi một package

```bash
# Thử cài riêng package bị lỗi với verbose mode
pip install ten-package-bi-loi -v

# Nếu lỗi liên quan đến compiler (Windows):
# Cài Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Nếu lỗi 'out of memory' khi cài:
pip install --no-cache-dir -r requirements.txt
```

### Lỗi: `CUDA out of memory` khi train

```bash
# Giảm batch size trong config
# configs/generator.yaml:
#   batch_size: 2  # giảm từ 4 xuống 2

# Hoặc bật mixed precision trong training script:
# Trainer(precision="16-mixed")

# Xóa cache GPU
python -c "import torch; torch.cuda.empty_cache()"
```

### Lỗi: `ModuleNotFoundError: No module named 'src'`

```bash
# Phải chạy script từ thư mục gốc của dự án
cd old-photo-restoration-v2  # đảm bảo đang ở đây
python scripts/train_segmentation.py

# Hoặc thêm vào đầu mỗi script:
# import sys; sys.path.insert(0, '.')
```

### Lỗi: `conda activate` không hoạt động trên Windows

```bash
# Chạy lệnh này một lần để fix:
conda init cmd.exe      # cho Command Prompt
# hoặc
conda init powershell   # cho PowerShell

# Sau đó đóng và mở lại terminal
```

### Quên đang dùng environment nào

```bash
# Xem danh sách tất cả environments, dấu * là env đang active
conda info --envs

# Tên env luôn hiển thị trong ngoặc ở đầu dòng:
# (old_photo) username@computer:~$
```

---

## Cheatsheet — Lệnh dùng hàng ngày

```bash
# === BẮT ĐẦU MỖI NGÀY LÀM VIỆC ===
conda activate old_photo          # kích hoạt môi trường
cd ~/Documents/old-photo-restoration-v2   # vào thư mục dự án

# === KIỂM TRA NHANH ===
nvidia-smi                        # xem GPU còn bao nhiêu VRAM
conda info --envs                 # xem env nào đang active
pip list | grep torch             # kiểm tra PyTorch version

# === CHẠY DỰ ÁN ===
python scripts/build_dataset.py           # tạo dataset
python scripts/train_segmentation.py     # train Module 1
python scripts/train_restoration.py      # fine-tune Module 2
python scripts/infer.py --input test.jpg # inference

# === KẾT THÚC NGÀY LÀM VIỆC ===
conda deactivate                  # tắt môi trường
```
