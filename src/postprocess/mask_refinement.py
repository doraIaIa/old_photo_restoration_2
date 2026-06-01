from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


VALID_REFINE_MODES = {
    "none",
    "dilate1",
    "dilate2",
    "dilate3",
    "close_dilate1",
    "repair_v1",
    "repair_v2",
    "repair_v3_conservative",
}


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    elif array.ndim != 2:
        raise ValueError(f"Mask phải có shape HxW hoặc HxWx1/HxWx3, nhận được {array.shape}")

    if array.dtype == np.bool_:
        binary = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.floating):
        threshold = 0.5 if float(np.nanmax(array)) <= 1.0 else 127.0
        binary = (array > threshold).astype(np.uint8) * 255
    else:
        binary = (array > 127).astype(np.uint8) * 255
    return np.ascontiguousarray(binary.astype(np.uint8))


def mask_ratio(mask: np.ndarray) -> float:
    binary = ensure_binary_mask(mask)
    return float((binary > 0).mean())


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (ensure_binary_mask(mask) > 0).astype(np.uint8)
    if min_area <= 0:
        return binary * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = np.zeros_like(binary)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= min_area:
            kept[labels == label_index] = 1
    return kept.astype(np.uint8) * 255


def _line_kernels(kernel_len: int) -> list[np.ndarray]:
    if kernel_len < 3:
        kernel_len = 3
    if kernel_len % 2 == 0:
        kernel_len += 1

    horizontal = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
    vertical = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
    diag_down = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
    diag_up = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
    center = kernel_len // 2
    horizontal[center, :] = 1
    vertical[:, center] = 1
    np.fill_diagonal(diag_down, 1)
    np.fill_diagonal(np.fliplr(diag_up), 1)
    return [horizontal, vertical, diag_down, diag_up]


def bridge_line_gaps(mask: np.ndarray, kernel_len: int = 7, iterations: int = 1) -> np.ndarray:
    binary = ensure_binary_mask(mask)
    bridged = binary.copy()
    for kernel in _line_kernels(kernel_len):
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        bridged = np.maximum(bridged, closed)
    return ensure_binary_mask(bridged)


def refine_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode not in VALID_REFINE_MODES:
        raise ValueError(f"mode không hợp lệ: {mode}. Hợp lệ: {sorted(VALID_REFINE_MODES)}")

    binary = ensure_binary_mask(mask)
    ellipse3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    square3 = np.ones((3, 3), dtype=np.uint8)
    square5 = np.ones((5, 5), dtype=np.uint8)

    if mode == "none":
        return binary
    if mode == "dilate1":
        return ensure_binary_mask(cv2.dilate(binary, ellipse3, iterations=1))
    if mode == "dilate2":
        return ensure_binary_mask(cv2.dilate(binary, ellipse3, iterations=2))
    if mode == "dilate3":
        return ensure_binary_mask(cv2.dilate(binary, ellipse3, iterations=3))
    if mode == "close_dilate1":
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, square3, iterations=1)
        return ensure_binary_mask(cv2.dilate(closed, ellipse3, iterations=1))
    if mode == "repair_v1":
        repaired = remove_small_components(binary, min_area=8)
        repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, square3, iterations=1)
        repaired = bridge_line_gaps(repaired, kernel_len=7, iterations=1)
        repaired = cv2.dilate(repaired, ellipse3, iterations=1)
        repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, square3, iterations=1)
        return ensure_binary_mask(repaired)
    if mode == "repair_v2":
        repaired = remove_small_components(binary, min_area=6)
        repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, square3, iterations=1)
        repaired = bridge_line_gaps(repaired, kernel_len=11, iterations=1)
        repaired = cv2.dilate(repaired, ellipse3, iterations=2)
        repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, square5, iterations=1)
        return ensure_binary_mask(repaired)
    if mode == "repair_v3_conservative":
        repaired = remove_small_components(binary, min_area=12)
        repaired = bridge_line_gaps(repaired, kernel_len=5, iterations=1)
        repaired = cv2.dilate(repaired, ellipse3, iterations=1)
        return ensure_binary_mask(repaired)
    raise AssertionError(f"mode chưa được xử lý: {mode}")


def compute_mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_binary = (ensure_binary_mask(pred) > 0).astype(np.uint8)
    gt_binary = (ensure_binary_mask(gt) > 0).astype(np.uint8)
    if pred_binary.shape != gt_binary.shape:
        gt_binary = cv2.resize(gt_binary, (pred_binary.shape[1], pred_binary.shape[0]), interpolation=cv2.INTER_NEAREST)

    tp = float((pred_binary * gt_binary).sum())
    fp = float((pred_binary * (1 - gt_binary)).sum())
    fn = float(((1 - pred_binary) * gt_binary).sum())
    total = float(pred_binary.size)
    eps = 1e-6
    return {
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "f1": (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "pred_ratio": float(pred_binary.mean()),
        "gt_ratio": float(gt_binary.mean()),
        "intersection_ratio": tp / total,
        "missing_ratio": fn / total,
        "extra_ratio": fp / total,
    }


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))


def _save_gray(path: Path, image_gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), ensure_binary_mask(image_gray))


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    image_rgb = np.asarray(image)
    if image_rgb.ndim == 2:
        image_rgb = np.repeat(image_rgb[:, :, None], 3, axis=2)
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    binary = ensure_binary_mask(mask)
    if binary.shape != image_rgb.shape[:2]:
        binary = cv2.resize(binary, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_layer = np.zeros_like(image_rgb)
    color_layer[:, :] = np.array(color, dtype=np.uint8)
    weight = (binary.astype(np.float32) / 255.0)[:, :, None] * alpha
    return np.clip(image_rgb.astype(np.float32) * (1.0 - weight) + color_layer.astype(np.float32) * weight, 0, 255).astype(np.uint8)


def save_mask_diff_visuals(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, output_dir: str | Path, prefix: str) -> None:
    output_path = Path(output_dir)
    pred_binary = ensure_binary_mask(pred)
    gt_binary = ensure_binary_mask(gt)
    if gt_binary.shape != pred_binary.shape:
        gt_binary = cv2.resize(gt_binary, (pred_binary.shape[1], pred_binary.shape[0]), interpolation=cv2.INTER_NEAREST)

    pred_bool = pred_binary > 0
    gt_bool = gt_binary > 0
    overlap = np.zeros((pred_binary.shape[0], pred_binary.shape[1], 3), dtype=np.uint8)
    overlap[pred_bool & gt_bool] = (0, 255, 0)
    overlap[~pred_bool & gt_bool] = (255, 0, 0)
    overlap[pred_bool & ~gt_bool] = (0, 0, 255)

    _save_gray(output_path / f"{prefix}_pred.png", pred_binary)
    _save_gray(output_path / f"{prefix}_gt.png", gt_binary)
    _save_rgb(output_path / f"{prefix}_overlap.png", overlap)
    _save_rgb(output_path / f"{prefix}_overlay_pred.png", _overlay(image, pred_binary, (255, 0, 0)))
    _save_rgb(output_path / f"{prefix}_overlay_gt.png", _overlay(image, gt_binary, (0, 255, 0)))


def summarize_ratios(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "mean": float(array.mean()), "max": float(array.max())}
