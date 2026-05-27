"""
src/data/degradation.py
=======================
Degradation Pipeline — Physically-Grounded Crack Blending
Blueprint 2.1 | Phase 1 | Core Contribution

Mục tiêu:
- Nhận ảnh sạch RGB và một ảnh crack RGBA.
- Sinh cặp (degraded_img, crack_mask) với mask ground-truth chính xác.
- Dùng heightmap -> normal map -> Phong illumination để tạo bóng/viền sáng.

Convention:
- Image arrays dùng RGB, dtype uint8, range [0, 255].
- Mask arrays dùng uint8, range {0, 255}.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


Array = np.ndarray


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------
def _ensure_uint8_rgb(img: Array, name: str = "image") -> Array:
    """Validate RGB uint8 image."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(img)!r}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _ensure_uint8_rgba(img: Array, name: str = "crack_rgba") -> Array:
    """Validate RGBA uint8 crack image."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be np.ndarray, got {type(img)!r}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"{name} must have shape (H, W, 4), got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# -----------------------------------------------------------------------------
# Step 1: Heightmap
# -----------------------------------------------------------------------------
def compute_heightmap(crack_gray: Array, blur_sigma: float = 3.0) -> Array:
    """
    Convert grayscale crack image to a normalized heightmap in [0, 1].

    Crack thường tối. Ta invert ảnh để vùng crack có giá trị cao hơn,
    sau đó Gaussian blur để tạo sườn mượt quanh biên crack.

    Args:
        crack_gray: grayscale image, shape (H, W), uint8/float.
        blur_sigma: Gaussian blur sigma.

    Returns:
        heightmap: float32, shape (H, W), range [0, 1].
    """
    if crack_gray.ndim != 2:
        raise ValueError(f"crack_gray must have shape (H, W), got {crack_gray.shape}")

    gray = np.clip(crack_gray, 0, 255).astype(np.float32)
    inverted = 255.0 - gray
    blurred = cv2.GaussianBlur(inverted, (0, 0), sigmaX=float(blur_sigma))

    h_min = float(blurred.min())
    h_max = float(blurred.max())
    if h_max - h_min <= 1e-6:
        return np.zeros_like(blurred, dtype=np.float32)

    heightmap = (blurred - h_min) / (h_max - h_min)
    return heightmap.astype(np.float32)


# -----------------------------------------------------------------------------
# Step 2: Normal map 3D
# -----------------------------------------------------------------------------
def compute_normal_map(heightmap: Array, strength: float = 5.0) -> Array:
    """
    Compute 3D normal map from heightmap.

    Correct formula:
        N = normalize([-dH/dx * strength, -dH/dy * strength, 1])

    Lưu ý: Sobel(luminance) trực tiếp chỉ cho edge/gradient 2D,
    không phải normal map 3D. Normal map phải xuất phát từ heightmap.

    Args:
        heightmap: float array, shape (H, W), normally in [0, 1].
        strength: normal intensity. Larger -> stronger 3D lighting effect.

    Returns:
        normal_map: float32, shape (H, W, 3), each vector has norm ≈ 1.
    """
    if heightmap.ndim != 2:
        raise ValueError(f"heightmap must have shape (H, W), got {heightmap.shape}")

    h = heightmap.astype(np.float32)
    dH_dx = cv2.Sobel(h, cv2.CV_64F, 1, 0, ksize=3)
    dH_dy = cv2.Sobel(h, cv2.CV_64F, 0, 1, ksize=3)

    nx = -dH_dx * float(strength)
    ny = -dH_dy * float(strength)
    nz = np.ones_like(nx)

    normal = np.stack([nx, ny, nz], axis=-1)
    magnitude = np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    normal = normal / magnitude
    return normal.astype(np.float32)


# -----------------------------------------------------------------------------
# Step 3: Phong illumination
# -----------------------------------------------------------------------------
def apply_phong_illumination(
    crack_rgb: Array,
    normal_map: Array,
    light_dir: Optional[Array] = None,
    ambient: float = 0.30,
    diffuse: float = 0.70,
) -> Array:
    """
    Apply simplified Phong diffuse illumination to crack texture.

    I = ambient + diffuse * max(dot(N, L), 0)

    Args:
        crack_rgb: RGB crack image, shape (H, W, 3), uint8.
        normal_map: normal vectors, shape (H, W, 3), float.
        light_dir: 3D light direction. Default: [0.5, 0.5, 1.0].
        ambient: base lighting.
        diffuse: directional lighting amount.

    Returns:
        illuminated RGB crack, uint8, shape (H, W, 3).
    """
    crack_rgb = _ensure_uint8_rgb(crack_rgb, "crack_rgb")
    if normal_map.shape != crack_rgb.shape[:2] + (3,):
        raise ValueError(
            f"normal_map shape must be {crack_rgb.shape[:2] + (3,)}, got {normal_map.shape}"
        )

    if light_dir is None:
        light_dir = np.array([0.5, 0.5, 1.0], dtype=np.float64)
    light = np.asarray(light_dir, dtype=np.float64)
    light = light / max(np.linalg.norm(light), 1e-8)

    dot = np.sum(normal_map.astype(np.float64) * light, axis=-1)
    dot = np.clip(dot, 0.0, 1.0)
    intensity = float(ambient) + float(diffuse) * dot
    intensity = np.clip(intensity, 0.0, 2.0)

    out = (crack_rgb.astype(np.float32) / 255.0) * intensity[:, :, None]
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


# -----------------------------------------------------------------------------
# Step 4: Crack augmentation and placement
# -----------------------------------------------------------------------------
def _rotate_rgba_or_pair(rgb: Array, alpha: Array, angle: float) -> Tuple[Array, Array]:
    """Rotate RGB and alpha around center while preserving canvas size."""
    h, w = rgb.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]

    rgb_rot = cv2.warpAffine(
        rgb,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    alpha_rot = cv2.warpAffine(
        alpha,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rgb_rot, alpha_rot


def augment_crack(
    crack_rgb: Array,
    crack_alpha: Array,
    scale_range: Tuple[float, float] = (0.5, 1.5),
    rotation_range: Tuple[float, float] = (-25.0, 25.0),
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.3,
) -> Tuple[Array, Array]:
    """Random scale, rotate, and flip crack texture and alpha."""
    crack_rgb = _ensure_uint8_rgb(crack_rgb, "crack_rgb")
    if crack_alpha.ndim != 2:
        raise ValueError(f"crack_alpha must have shape (H, W), got {crack_alpha.shape}")
    crack_alpha = np.clip(crack_alpha, 0, 255).astype(np.uint8)

    h, w = crack_rgb.shape[:2]
    scale = random.uniform(*scale_range)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    crack_rgb = cv2.resize(crack_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    crack_alpha = cv2.resize(crack_alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    angle = random.uniform(*rotation_range)
    crack_rgb, crack_alpha = _rotate_rgba_or_pair(crack_rgb, crack_alpha, angle)

    if random.random() < hflip_prob:
        crack_rgb = cv2.flip(crack_rgb, 1)
        crack_alpha = cv2.flip(crack_alpha, 1)
    if random.random() < vflip_prob:
        crack_rgb = cv2.flip(crack_rgb, 0)
        crack_alpha = cv2.flip(crack_alpha, 0)

    return crack_rgb, crack_alpha


# -----------------------------------------------------------------------------
# Step 5: Alpha blending
# -----------------------------------------------------------------------------
def alpha_blend(
    clean_img: Array,
    crack_rgb: Array,
    crack_alpha: Array,
    pos_x: int,
    pos_y: int,
) -> Array:
    """
    Blend crack onto clean image at (pos_x, pos_y).

    Formula:
        out = crack * alpha + clean * (1 - alpha)
    """
    clean_img = _ensure_uint8_rgb(clean_img, "clean_img")
    crack_rgb = _ensure_uint8_rgb(crack_rgb, "crack_rgb")
    crack_alpha = np.clip(crack_alpha, 0, 255).astype(np.uint8)

    result = clean_img.copy().astype(np.float32)
    img_h, img_w = clean_img.shape[:2]
    crack_h, crack_w = crack_rgb.shape[:2]

    x0 = int(pos_x)
    y0 = int(pos_y)
    x1 = min(x0 + crack_w, img_w)
    y1 = min(y0 + crack_h, img_h)

    valid_w = x1 - x0
    valid_h = y1 - y0
    if valid_w <= 0 or valid_h <= 0:
        return clean_img.copy()

    region = result[y0:y1, x0:x1]
    c_rgb = crack_rgb[:valid_h, :valid_w].astype(np.float32)
    alpha = (crack_alpha[:valid_h, :valid_w].astype(np.float32) / 255.0)[:, :, None]

    result[y0:y1, x0:x1] = c_rgb * alpha + region * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Step 6: Global degradation
# -----------------------------------------------------------------------------
def add_global_degradation(
    img: Array,
    noise_sigma_range: Tuple[float, float] = (5.0, 25.0),
    motion_blur_prob: float = 0.50,
    sepia_prob: float = 0.60,
) -> Array:
    """Add unstructured degradation: Gaussian noise, motion blur, sepia fading."""
    img = _ensure_uint8_rgb(img, "img")
    result = img.astype(np.float32)

    # Gaussian noise
    sigma = random.uniform(*noise_sigma_range)
    noise = np.random.normal(0.0, sigma, result.shape)
    result = np.clip(result + noise, 0, 255)

    # Motion blur
    if random.random() < motion_blur_prob:
        kernel_size = random.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size

        matrix = cv2.getRotationMatrix2D(
            (kernel_size / 2.0, kernel_size / 2.0),
            random.uniform(0.0, 180.0),
            1.0,
        )
        kernel = cv2.warpAffine(kernel, matrix, (kernel_size, kernel_size))
        if float(kernel.sum()) > 1e-6:
            kernel = kernel / kernel.sum()
        result = cv2.filter2D(result, -1, kernel)

    # Sepia / color fading. Matrix assumes RGB input.
    if random.random() < sepia_prob:
        sepia = np.array(
            [
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131],
            ],
            dtype=np.float32,
        )
        h, w = result.shape[:2]
        flat = (result / 255.0).reshape(-1, 3)
        result = np.clip(flat @ sepia.T, 0.0, 1.0).reshape(h, w, 3) * 255.0

    return np.clip(result, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
def generate_degraded_pair(
    clean_img: Array,
    crack_rgba: Array,
    normal_strength: float = 5.0,
    light_dir: Optional[Array] = None,
    alpha_threshold: int = 30,
    apply_global_degrade: bool = True,
) -> Tuple[Array, Array]:
    """
    Generate a synthetic old-photo crack pair.

    Args:
        clean_img: clean RGB image, uint8, shape (H, W, 3).
        crack_rgba: crack crop with alpha channel, uint8, shape (h, w, 4).
        normal_strength: strength for normal-map lighting.
        light_dir: optional light direction vector.
        alpha_threshold: alpha threshold to create binary GT mask.
        apply_global_degrade: whether to add noise/blur/sepia after crack blend.

    Returns:
        degraded_img: RGB uint8 image, shape (H, W, 3).
        crack_mask: binary uint8 mask, shape (H, W), values {0, 255}.
    """
    clean_img = _ensure_uint8_rgb(clean_img, "clean_img")
    crack_rgba = _ensure_uint8_rgba(crack_rgba, "crack_rgba")

    crack_rgb = crack_rgba[:, :, :3]
    crack_alpha = crack_rgba[:, :, 3]

    # 1. Heightmap
    crack_gray = cv2.cvtColor(crack_rgb, cv2.COLOR_RGB2GRAY)
    heightmap = compute_heightmap(crack_gray)

    # 2. Normal map
    normal_map = compute_normal_map(heightmap, strength=normal_strength)

    # 3. Physically inspired lighting
    crack_3d = apply_phong_illumination(crack_rgb, normal_map, light_dir)

    # 4. Augment crack
    crack_3d, crack_alpha = augment_crack(crack_3d, crack_alpha)

    img_h, img_w = clean_img.shape[:2]
    crack_h, crack_w = crack_3d.shape[:2]

    # If crack is too large, scale down safely.
    if crack_h >= img_h or crack_w >= img_w:
        scale = min((img_h - 1) / max(crack_h, 1), (img_w - 1) / max(crack_w, 1)) * 0.90
        new_h = max(1, int(crack_h * scale))
        new_w = max(1, int(crack_w * scale))
        crack_3d = cv2.resize(crack_3d, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        crack_alpha = cv2.resize(crack_alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        crack_h, crack_w = new_h, new_w

    pos_x = random.randint(0, max(0, img_w - crack_w))
    pos_y = random.randint(0, max(0, img_h - crack_h))

    # 5. Alpha blend
    degraded = alpha_blend(clean_img, crack_3d, crack_alpha, pos_x, pos_y)

    # 6. Ground-truth mask
    crack_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    valid_h = min(crack_h, img_h - pos_y)
    valid_w = min(crack_w, img_w - pos_x)
    crack_mask[pos_y : pos_y + valid_h, pos_x : pos_x + valid_w] = (
        crack_alpha[:valid_h, :valid_w] > int(alpha_threshold)
    ).astype(np.uint8) * 255

    # 7. Global degradation
    if apply_global_degrade:
        degraded = add_global_degradation(degraded)

    return degraded, crack_mask


# -----------------------------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------------------------
def load_rgb(path: str | Path) -> Array:
    """Load image as RGB uint8."""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_crack_rgba(path: str | Path) -> Array:
    """
    Load crack as RGBA uint8.

    Supported:
    - RGBA PNG: use provided alpha.
    - RGB/BGR image: create alpha from darkness.
    - Grayscale image: create RGB from grayscale and alpha from darkness.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read crack image: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    if img.ndim == 3 and img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alpha = (255 - gray).astype(np.uint8)
        return np.dstack([rgb, alpha]).astype(np.uint8)

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        alpha = (255 - img).astype(np.uint8)
        return np.dstack([rgb, alpha]).astype(np.uint8)

    raise ValueError(f"Unsupported crack image shape: {img.shape} from {path}")


def save_rgb(path: str | Path, img: Array) -> None:
    """Save RGB uint8 image using OpenCV."""
    img = _ensure_uint8_rgb(img, "img")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_mask(path: str | Path, mask: Array) -> None:
    """Save grayscale uint8 mask."""
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {mask.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(mask, 0, 255).astype(np.uint8))


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def _make_demo_clean(h: int = 512, w: int = 512) -> Array:
    """Create a smooth RGB demo image."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    r = 170 + 35 * np.sin(2 * math.pi * y) + 15 * x
    g = 150 + 25 * np.cos(2 * math.pi * x) + 10 * y
    b = 130 + 20 * np.sin(2 * math.pi * (x + y))
    img = np.stack([np.broadcast_to(r, (h, w)), np.broadcast_to(g, (h, w)), b], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _make_demo_crack(size: int = 220) -> Array:
    """Create a synthetic RGBA crack only for self-test."""
    crack = np.zeros((size, size, 4), dtype=np.uint8)
    for t in range(20, size - 20):
        x = int(20 + 0.72 * t + 9 * np.sin(t / 12.0))
        y = int(t)
        if 0 <= x < size and 0 <= y < size:
            for dx in range(-4, 5):
                nx = x + dx
                if 0 <= nx < size:
                    alpha = max(0, 255 - abs(dx) * 55)
                    crack[y, nx] = [35, 25, 18, alpha]
            if t % 37 == 0:
                cv2.line(crack, (x, y), (min(size - 1, x + 30), min(size - 1, y + 15)), (32, 22, 16, 180), 2)
    return crack


def self_test(output_path: str | Path = "degradation_test_output.png") -> None:
    """Run a quick CPU-only validation and save visualization image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random.seed(42)
    np.random.seed(42)

    clean = _make_demo_clean()
    crack = _make_demo_crack()

    gray = cv2.cvtColor(crack[:, :, :3], cv2.COLOR_RGB2GRAY)
    heightmap = compute_heightmap(gray)
    normal = compute_normal_map(heightmap)
    norms = np.linalg.norm(normal, axis=-1)

    assert heightmap.shape == gray.shape
    assert normal.shape == gray.shape + (3,)
    assert np.allclose(norms[heightmap > 0].mean(), 1.0, atol=1e-2)

    degraded, mask = generate_degraded_pair(clean, crack)
    assert degraded.shape == clean.shape
    assert mask.shape == clean.shape[:2]
    assert mask.dtype == np.uint8
    assert int((mask > 0).sum()) > 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clean)
    axes[0].set_title("Clean")
    axes[1].imshow(degraded)
    axes[1].set_title("Degraded with 3D crack")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Ground-truth mask")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    output_path = Path(output_path)
    fig.savefig(str(output_path), dpi=120)
    plt.close(fig)
    print(f"OK: degradation pipeline self-test passed. Saved: {output_path}")


if __name__ == "__main__":
    self_test()
