from __future__ import annotations

import random

import numpy as np

from src.data.degradation import (
    alpha_blend,
    compute_heightmap,
    compute_normal_map,
    generate_degraded_pair,
)


def _make_synthetic_clean(size: int = 64) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = 120
    image[..., 1] = 160
    image[..., 2] = 200
    return image


def _make_synthetic_crack(size: int = 32) -> np.ndarray:
    crack = np.zeros((size, size, 4), dtype=np.uint8)
    crack[..., :3] = 30
    crack[8:24, 14:18, 3] = 255
    crack[10:22, 13:19, :3] = 20
    return crack


def test_compute_heightmap_shape_dtype_and_range() -> None:
    crack_gray = np.zeros((16, 16), dtype=np.uint8)
    crack_gray[4:12, 7:9] = 255

    heightmap = compute_heightmap(crack_gray)

    assert heightmap.shape == crack_gray.shape
    assert heightmap.dtype == np.float32
    assert float(heightmap.min()) >= 0.0
    assert float(heightmap.max()) <= 1.0


def test_compute_normal_map_shape_dtype_and_unit_norm() -> None:
    crack_gray = np.zeros((16, 16), dtype=np.uint8)
    crack_gray[4:12, 7:9] = 255
    heightmap = compute_heightmap(crack_gray)

    normal_map = compute_normal_map(heightmap)
    norms = np.linalg.norm(normal_map, axis=-1)

    assert normal_map.shape == (16, 16, 3)
    assert normal_map.dtype == np.float32
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_alpha_blend_preserves_shape_dtype_and_range() -> None:
    clean = _make_synthetic_clean()
    crack_rgb = np.full((20, 20, 3), 40, dtype=np.uint8)
    crack_alpha = np.zeros((20, 20), dtype=np.uint8)
    crack_alpha[5:15, 5:15] = 180

    blended = alpha_blend(clean, crack_rgb, crack_alpha, pos_x=10, pos_y=12)

    assert blended.shape == clean.shape
    assert blended.dtype == np.uint8
    assert int(blended.min()) >= 0
    assert int(blended.max()) <= 255


def test_generate_degraded_pair_returns_valid_binary_like_mask() -> None:
    random.seed(42)
    np.random.seed(42)

    clean = _make_synthetic_clean()
    crack_rgba = _make_synthetic_crack()

    degraded, mask = generate_degraded_pair(clean, crack_rgba, apply_global_degrade=False)
    unique_values = set(np.unique(mask).tolist())

    assert degraded.shape == clean.shape
    assert degraded.dtype == np.uint8
    assert mask.shape == clean.shape[:2]
    assert mask.dtype == np.uint8
    assert unique_values.issubset({0, 255})
    assert int((mask > 0).sum()) > 0
