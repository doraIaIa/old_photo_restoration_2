from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.data.transforms import get_segmentation_transforms
from src.models.segmenter import CrackSegmenter
from src.postprocess.mask_refinement import VALID_REFINE_MODES, refine_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy demo old photo restoration với nhiều nguồn mask.")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh đầu vào.")
    parser.add_argument("--checkpoint", required=True, help="Đường dẫn checkpoint `best_iou.ckpt`.")
    parser.add_argument("--threshold", type=float, default=0.90, help="Ngưỡng chính cho DL mask.")
    parser.add_argument("--fallback-threshold", type=float, default=0.70, help="Ngưỡng fallback cho DL mask.")
    parser.add_argument("--output-dir", default="outputs/demo/r009_lama", help="Thư mục gốc lưu output demo.")
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda.")
    parser.add_argument("--image-size", type=int, default=512, help="Kích thước resize cho model segmentation.")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "simple_lama", "opencv"],
        help="Backend inpaint: simple_lama, opencv hoặc auto.",
    )
    parser.add_argument(
        "--mask-source",
        default="dl",
        choices=["dl", "cv", "union", "external"],
        help="Nguồn mask cuối cùng dùng để inpaint.",
    )
    parser.add_argument("--external-mask", default="", help="Mask ngoài do người dùng vẽ/sửa thủ công.")
    parser.add_argument("--cv-profile", default="notebook_v7_candidate", help="Profile cho classical CV mask generator.")
    parser.add_argument(
        "--cv-auto-invert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tự động đảo polarity CV mask nếu tỷ lệ trắng quá lớn và mask đảo có vẻ hợp lý hơn.",
    )
    parser.add_argument("--cv-debug", action="store_true", help="Lưu các ảnh debug trung gian của CV pipeline.")
    parser.add_argument("--mask-dilate", type=int, default=0, help="Số lần dilation áp lên final mask sau khi chọn nguồn.")
    parser.add_argument(
        "--mask-refine",
        choices=sorted(VALID_REFINE_MODES),
        default="none",
        help="Refinement bổ sung sau mask-source và --mask-dilate để tạo repair mask.",
    )
    parser.add_argument("--save-prob-mask", action="store_true", help="Lưu `dl_prob_mask.png` nếu có DL inference.")
    parser.add_argument("--save-all-masks", action="store_true", help="Lưu đầy đủ DL/CV/union/external mask nếu khả dụng.")
    parser.add_argument("--no-inpaint", action="store_true", help="Chỉ tạo mask và overlay, không chạy inpaint.")
    parser.add_argument("--config", default="configs/data.yaml", help="Config YAML để lấy fallback runtime settings.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(device_arg: str) -> torch.device:
    requested = device_arg.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng nhưng được yêu cầu qua --device cuda.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"--device không hợp lệ: {device_arg}")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh grayscale: {path}")
    return image


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(ensure_rgb_uint8(image), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Không ghi được ảnh RGB: {path}")


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gray = ensure_gray_uint8(image)
    if not cv2.imwrite(str(path), gray):
        raise IOError(f"Không ghi được ảnh grayscale: {path}")


def save_debug_images(output_dir: Path, debug_images: dict[str, np.ndarray]) -> None:
    for name, image in debug_images.items():
        save_gray(output_dir / f"{name}.png", image)


def threshold_to_tag(value: float) -> str:
    return f"t{value:.2f}".replace(".", "p")


def parse_run_id(checkpoint_path: Path) -> str | None:
    parent = checkpoint_path.parent
    return parent.name or None


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[CrackSegmenter, dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint thiếu `model_state_dict`: {checkpoint_path}")

    model_config = checkpoint.get("model_config") or {}
    model = CrackSegmenter(
        in_channels=int(model_config.get("in_channels", 3)),
        out_channels=int(model_config.get("out_channels", 1)),
        base_channels=int(model_config.get("base_channels", 8)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def build_inference_tensor(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    transform = get_segmentation_transforms(split="val", image_size=image_size, aug_profile="baseline")
    dummy_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    transformed = transform(image=image_rgb, mask=dummy_mask)
    image_tensor = transformed["image"]
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.as_tensor(image_tensor)
    if image_tensor.ndim == 3 and image_tensor.shape[0] != 3 and image_tensor.shape[-1] == 3:
        image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.float()
    if float(image_tensor.max()) > 1.0:
        image_tensor = image_tensor / 255.0
    return image_tensor.unsqueeze(0)


def resize_probability_mask(probability_mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_hw
    return cv2.resize(probability_mask, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def probability_to_uint8(probability_mask: np.ndarray) -> np.ndarray:
    return np.clip(np.round(probability_mask * 255.0), 0, 255).astype(np.uint8)


def binary_mask_from_probability(probability_mask: np.ndarray, threshold: float) -> np.ndarray:
    return (probability_mask >= threshold).astype(np.uint8) * 255


def ensure_gray_uint8(image: np.ndarray | None, fallback_shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    if image is None:
        return np.zeros(fallback_shape, dtype=np.uint8)

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(ensure_rgb_uint8(array), cv2.COLOR_RGB2GRAY)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    elif array.ndim != 2:
        raise ValueError(f"Mask grayscale phải có shape HxW hoặc HxWx1, nhận được {array.shape}")

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
            array = np.clip(np.round(array * scale), 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def ensure_rgb_uint8(image: np.ndarray | None, fallback_shape: tuple[int, int, int] = (512, 512, 3)) -> np.ndarray:
    if image is None:
        return np.full(fallback_shape, 255, dtype=np.uint8)

    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(ensure_gray_uint8(array)[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        gray = ensure_gray_uint8(array[:, :, 0])
        array = np.repeat(gray[:, :, None], 3, axis=2)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Ảnh tile phải có shape HxW hoặc HxWx3, nhận được {array.shape}")

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
            array = np.clip(np.round(array * scale), 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


def make_gray_rgb(gray_image: np.ndarray | None, fallback_shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    gray = ensure_gray_uint8(gray_image, fallback_shape=fallback_shape)
    return np.repeat(gray[:, :, None], 3, axis=2)


def make_placeholder_tile(reference_shape: tuple[int, int, int], label: str, detail: str) -> np.ndarray:
    tile = np.full(reference_shape, 250, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (220, 220, 220), 2)
    cv2.putText(tile, label, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(tile, detail, (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2, cv2.LINE_AA)
    return tile


def letterbox_to_size(image: np.ndarray | None, tile_width: int, tile_height: int, fill: int = 255) -> np.ndarray:
    image_rgb = ensure_rgb_uint8(image, fallback_shape=(tile_height, tile_width, 3))
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Tile có kích thước không hợp lệ: {image_rgb.shape}")

    scale = min(tile_width / width, tile_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=interpolation)

    canvas = np.full((tile_height, tile_width, 3), fill, dtype=np.uint8)
    offset_x = (tile_width - resized_width) // 2
    offset_y = (tile_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def add_label(tile: np.ndarray, label: str, label_height: int = 40) -> np.ndarray:
    image_rgb = ensure_rgb_uint8(tile)
    banner = np.full((label_height, image_rgb.shape[1], 3), 245, dtype=np.uint8)
    baseline_y = min(label_height - 12, 28)
    cv2.putText(banner, label, (12, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    return np.concatenate([banner, image_rgb], axis=0)


def pad_to_width(row: np.ndarray, target_width: int, fill: int = 255) -> np.ndarray:
    row_rgb = ensure_rgb_uint8(row)
    current_width = row_rgb.shape[1]
    if current_width == target_width:
        return row_rgb
    if current_width > target_width:
        raise ValueError(f"target_width={target_width} nhỏ hơn row width={current_width}")

    padded = np.full((row_rgb.shape[0], target_width, 3), fill, dtype=np.uint8)
    padded[:, :current_width] = row_rgb
    return padded


def make_comparison_grid(tiles: list[tuple[str, np.ndarray | None]], max_columns: int = 4) -> np.ndarray:
    if not tiles:
        raise ValueError("Comparison grid cần ít nhất một tile.")

    prepared_images = [ensure_rgb_uint8(image) for _, image in tiles]
    reference_height, reference_width = prepared_images[0].shape[:2]
    tile_width = max(180, min(512, reference_width))
    tile_height = max(180, int(round(reference_height * (tile_width / max(reference_width, 1)))))
    label_height = 40

    labeled_tiles: list[np.ndarray] = []
    for (label, image), fallback in zip(tiles, prepared_images):
        boxed = letterbox_to_size(image if image is not None else fallback, tile_width, tile_height, fill=255)
        labeled_tiles.append(add_label(boxed, label, label_height=label_height))

    rows: list[np.ndarray] = []
    for start in range(0, len(labeled_tiles), max_columns):
        row_tiles = labeled_tiles[start : start + max_columns]
        rows.append(np.concatenate(row_tiles, axis=1))

    target_width = max(row.shape[1] for row in rows)
    padded_rows = [pad_to_width(row, target_width, fill=255) for row in rows]
    return np.concatenate(padded_rows, axis=0)


def make_overlay(image_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = ensure_rgb_uint8(image_rgb).astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255.0
    weight = (ensure_gray_uint8(binary_mask).astype(np.float32) / 255.0)[:, :, None] * alpha
    mixed = overlay * (1.0 - weight) + red * weight
    return np.clip(mixed, 0, 255).astype(np.uint8)


def mask_ratio(mask: np.ndarray | None) -> float | None:
    if mask is None:
        return None
    return float((ensure_gray_uint8(mask) > 0).mean())


def or_masks(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
    if left is None and right is None:
        return None
    if left is None:
        return ensure_gray_uint8(right)
    if right is None:
        return ensure_gray_uint8(left)
    return np.maximum(ensure_gray_uint8(left), ensure_gray_uint8(right))


def dilate_mask(binary_mask: np.ndarray | None, iterations: int) -> np.ndarray | None:
    if binary_mask is None:
        return None
    if iterations <= 0:
        return ensure_gray_uint8(binary_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(ensure_gray_uint8(binary_mask), kernel, iterations=int(iterations))


def remove_small_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = (ensure_gray_uint8(binary_mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label_index] = 1
    return filtered * 255


def is_reasonable_cv_ratio(ratio: float) -> bool:
    return 0.005 <= float(ratio) <= 0.25


def filter_components_by_area_or_span(binary_mask: np.ndarray, min_area: int, min_span: int) -> tuple[np.ndarray, int, int]:
    mask = (ensure_gray_uint8(binary_mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    kept = 0
    total = max(0, num_labels - 1)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if area >= min_area or max(width, height) >= min_span:
            filtered[labels == label_index] = 1
            kept += 1
    return filtered * 255, total, kept


def build_cv_crack_mask(
    image_rgb: np.ndarray,
    profile: str = "notebook_v7_candidate",
    auto_invert: bool = True,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    gray = cv2.cvtColor(ensure_rgb_uint8(image_rgb), cv2.COLOR_RGB2GRAY)

    if profile == "notebook_v7_candidate":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        close_kernel = np.ones((3, 3), dtype=np.uint8)
        dilate_kernel = np.ones((3, 3), dtype=np.uint8)

        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, blackhat_kernel)
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, tophat_kernel)
        edges = cv2.Canny(enhanced, threshold1=60, threshold2=160)

        score = np.maximum(blackhat, (0.65 * tophat).astype(np.uint8))
        score = np.maximum(score, (0.25 * edges).astype(np.uint8))
        score = cv2.GaussianBlur(score, (3, 3), 0)

        pctl = 97.2
        threshold = max(18.0, float(np.percentile(score, pctl)))
        raw_mask = (score >= threshold).astype(np.uint8) * 255
        after_close = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        after_component_filter, num_components, num_components_kept = filter_components_by_area_or_span(
            after_close,
            min_area=22,
            min_span=18,
        )
        final_mask = cv2.dilate(after_component_filter, dilate_kernel, iterations=1)

        area_pct_before_fallback = (mask_ratio(final_mask) or 0.0) * 100.0
        fallback_used = False
        if area_pct_before_fallback < 1.0:
            fallback_used = True
            pctl = 96.0
            threshold = max(14.0, float(np.percentile(score, pctl)))
            raw_mask = (score >= threshold).astype(np.uint8) * 255
            after_close = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
            after_component_filter, num_components, num_components_kept = filter_components_by_area_or_span(
                after_close,
                min_area=22,
                min_span=18,
            )
            final_mask = cv2.dilate(after_component_filter, dilate_kernel, iterations=1)

        final_mask = ensure_gray_uint8(final_mask)
        final_mask = ((final_mask > 127).astype(np.uint8) * 255).astype(np.uint8)
        area_pct_final = (mask_ratio(final_mask) or 0.0) * 100.0

        debug_images = {
            "gray": gray,
            "clahe": enhanced,
            "blackhat_response": blackhat,
            "tophat_response": tophat,
            "canny": edges,
            "score": score,
            "cv_raw_mask": raw_mask,
            "cv_after_close": after_close,
            "cv_after_component_filter": after_component_filter,
            "cv_final_mask": final_mask,
        }
        info = {
            "cv_mask_ratio_before_invert_check": mask_ratio(final_mask),
            "cv_mask_ratio_after_invert_check": mask_ratio(final_mask),
            "cv_auto_inverted": False,
            "cv_warning": None,
            "cv_pctl": float(pctl),
            "cv_threshold": float(threshold),
            "cv_area_pct_before_fallback": float(area_pct_before_fallback),
            "cv_area_pct_final": float(area_pct_final),
            "cv_fallback_used": bool(fallback_used),
            "cv_num_components": int(num_components),
            "cv_num_components_kept": int(num_components_kept),
        }
        return final_mask, info, debug_images

    if profile != "old_photo_crack":
        raise ValueError(f"Chưa hỗ trợ cv profile: {profile}")

    print(
        "WARNING: CV profile 'old_photo_crack' is noisy/deprecated for report demo; "
        "prefer notebook_v7_candidate."
    )

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel_large)
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_medium)
    gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel_small)
    edges = cv2.Canny(enhanced, threshold1=40, threshold2=120)

    response = np.maximum(blackhat, tophat)
    response = np.maximum(response, (gradient.astype(np.float32) * 0.6).astype(np.uint8))

    non_zero_response = response[response > 0]
    percentile_value = float(np.percentile(non_zero_response, 92)) if non_zero_response.size > 0 else 0.0
    response_threshold = max(12.0, percentile_value)
    response_mask = (response >= response_threshold).astype(np.uint8) * 255

    combined_response = np.maximum(response_mask, edges)
    combined = cv2.morphologyEx(combined_response, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small, iterations=1)
    raw_mask = remove_small_components(combined, min_area=32)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    raw_mask = ((raw_mask > 127).astype(np.uint8) * 255).astype(np.uint8)

    ratio_before = mask_ratio(raw_mask) or 0.0
    final_mask = raw_mask.copy()
    auto_inverted = False
    warning: str | None = None

    if auto_invert and ratio_before > 0.50:
        inverted_mask = (255 - raw_mask).astype(np.uint8)
        inverted_ratio = mask_ratio(inverted_mask) or 0.0
        if is_reasonable_cv_ratio(inverted_ratio):
            final_mask = inverted_mask
            auto_inverted = True
        else:
            warning = (
                "CV mask có tỷ lệ trắng quá lớn, đã thử invert nhưng mask đảo vẫn không nằm trong khoảng hợp lý."
            )

    ratio_after = mask_ratio(final_mask) or 0.0
    if warning is None and not is_reasonable_cv_ratio(ratio_after):
        warning = (
            "CV mask ratio sau bước kiểm tra polarity vẫn ngoài khoảng khuyến nghị [0.005, 0.25]."
        )

    debug_images = {
        "gray": gray,
        "clahe": enhanced,
        "blackhat_response": blackhat,
        "tophat_response": tophat,
        "canny": edges,
        "cv_combined_response": combined_response,
        "cv_raw_mask": raw_mask,
        "cv_final_mask": final_mask,
    }
    info = {
        "cv_mask_ratio_before_invert_check": float(ratio_before),
        "cv_mask_ratio_after_invert_check": float(ratio_after),
        "cv_auto_inverted": bool(auto_inverted),
        "cv_warning": warning,
        "cv_pctl": 92.0,
        "cv_threshold": float(response_threshold),
        "cv_area_pct_before_fallback": float(ratio_before * 100.0),
        "cv_area_pct_final": float(ratio_after * 100.0),
        "cv_fallback_used": False,
        "cv_num_components": None,
        "cv_num_components_kept": None,
    }
    return ensure_gray_uint8(final_mask), info, debug_images


def load_external_mask(mask_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    mask = load_grayscale(mask_path)
    target_height, target_width = target_hw
    if mask.shape != (target_height, target_width):
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return ((mask > 127).astype(np.uint8) * 255).astype(np.uint8)


def try_create_simple_lama() -> Any:
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise ImportError(
            "Không import được `simple_lama_inpainting`. Hãy cài package tương ứng, ví dụ `pip install simple-lama-inpainting`."
        ) from exc
    return SimpleLama()


def inpaint_with_simple_lama(image_rgb: np.ndarray, binary_mask: np.ndarray, simple_lama: Any) -> np.ndarray:
    from PIL import Image

    image_pil = Image.fromarray(ensure_rgb_uint8(image_rgb))
    mask_pil = Image.fromarray(ensure_gray_uint8(binary_mask)).convert("L")
    restored = simple_lama(image_pil, mask_pil)
    if not isinstance(restored, Image.Image):
        raise TypeError(f"SimpleLama trả về kiểu không mong đợi: {type(restored)!r}")
    return np.array(restored.convert("RGB"))


def inpaint_with_opencv(image_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(ensure_rgb_uint8(image_rgb), cv2.COLOR_RGB2BGR)
    restored_bgr = cv2.inpaint(image_bgr, ensure_gray_uint8(binary_mask), 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)


def resolve_inpaint_backend(backend_arg: str) -> tuple[str, str | None, Any | None]:
    if backend_arg == "opencv":
        return "opencv", None, None
    if backend_arg == "simple_lama":
        return "simple_lama", None, try_create_simple_lama()
    try:
        return "simple_lama", None, try_create_simple_lama()
    except ImportError as exc:
        return "opencv", str(exc), None


def run_inpaint(image_rgb: np.ndarray, binary_mask: np.ndarray, backend: str, simple_lama: Any | None = None) -> np.ndarray:
    if backend == "simple_lama":
        if simple_lama is None:
            simple_lama = try_create_simple_lama()
        return inpaint_with_simple_lama(image_rgb, binary_mask, simple_lama)
    if backend == "opencv":
        return inpaint_with_opencv(image_rgb, binary_mask)
    raise ValueError(f"Backend inpaint không hợp lệ: {backend}")


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold phải nằm trong [0, 1].")
    if not 0.0 <= args.fallback_threshold <= 1.0:
        raise ValueError("--fallback-threshold phải nằm trong [0, 1].")
    if args.fallback_threshold > args.threshold:
        raise ValueError("--fallback-threshold nên nhỏ hơn hoặc bằng --threshold.")
    if args.image_size <= 0:
        raise ValueError("--image-size phải > 0.")
    if args.mask_dilate < 0:
        raise ValueError("--mask-dilate phải >= 0.")
    if args.mask_source == "external" and not args.external_mask:
        raise ValueError("--mask-source external yêu cầu truyền --external-mask.")

    config = load_config(resolve_path(args.config))
    device = resolve_device(args.device)
    image_path = resolve_path(args.image)
    checkpoint_path = resolve_path(args.checkpoint)
    output_root = resolve_path(args.output_dir)
    external_mask_path = resolve_path(args.external_mask) if args.external_mask else None

    image_rgb = load_rgb(image_path)
    original_height, original_width = image_rgb.shape[:2]
    effective_image_size = int(args.image_size or config.get("build", {}).get("image_size", 512))

    need_dl = args.mask_source in {"dl", "union"} or args.save_prob_mask or args.save_all_masks
    need_cv = args.mask_source in {"cv", "union"} or args.save_all_masks
    need_external = args.mask_source == "external" or external_mask_path is not None

    checkpoint: dict[str, Any] = {}
    checkpoint_model_config: dict[str, Any] = {}
    probability_mask_uint8: np.ndarray | None = None
    dl_mask_primary: np.ndarray | None = None
    dl_mask_fallback: np.ndarray | None = None
    cv_info: dict[str, Any] = {
        "cv_mask_ratio_before_invert_check": None,
        "cv_mask_ratio_after_invert_check": None,
        "cv_auto_inverted": False,
        "cv_warning": None,
        "cv_pctl": None,
        "cv_threshold": None,
        "cv_area_pct_before_fallback": None,
        "cv_area_pct_final": None,
        "cv_fallback_used": False,
        "cv_num_components": None,
        "cv_num_components_kept": None,
    }
    cv_debug_images: dict[str, np.ndarray] = {}

    if need_dl:
        model, checkpoint = load_model(checkpoint_path, device)
        checkpoint_model_config = checkpoint.get("model_config") or {}
        input_tensor = build_inference_tensor(image_rgb, effective_image_size).to(device)
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
        if probabilities.ndim != 2:
            raise ValueError(f"Probability mask phải có 2 chiều, nhận được shape {probabilities.shape}")

        probability_mask = resize_probability_mask(probabilities, (original_height, original_width))
        probability_mask = np.clip(probability_mask, 0.0, 1.0)
        probability_mask_uint8 = probability_to_uint8(probability_mask)
        dl_mask_primary = binary_mask_from_probability(probability_mask, args.threshold)
        dl_mask_fallback = binary_mask_from_probability(probability_mask, args.fallback_threshold)
    else:
        checkpoint = {}
        checkpoint_model_config = {}

    if need_cv:
        cv_mask, cv_info, cv_debug_images = build_cv_crack_mask(
            image_rgb,
            profile=args.cv_profile,
            auto_invert=bool(args.cv_auto_invert),
        )
    else:
        cv_mask = None
    external_mask = load_external_mask(external_mask_path, (original_height, original_width)) if need_external and external_mask_path else None
    union_mask = or_masks(dl_mask_primary, cv_mask) if (dl_mask_primary is not None or cv_mask is not None) else None

    if args.mask_source == "dl":
        if dl_mask_primary is None:
            raise RuntimeError("Mask source `dl` yêu cầu DL inference, nhưng DL mask không khả dụng.")
        final_mask = dl_mask_primary
    elif args.mask_source == "cv":
        if cv_mask is None:
            raise RuntimeError("Mask source `cv` yêu cầu CV mask, nhưng CV mask không khả dụng.")
        final_mask = cv_mask
    elif args.mask_source == "union":
        if union_mask is None:
            raise RuntimeError("Mask source `union` yêu cầu ít nhất một trong DL hoặc CV mask.")
        final_mask = union_mask
    else:
        if external_mask is None:
            raise RuntimeError("Mask source `external` yêu cầu external mask hợp lệ.")
        final_mask = external_mask

    final_mask = dilate_mask(final_mask, args.mask_dilate)
    if final_mask is None:
        raise RuntimeError("Không tạo được final mask.")
    final_mask_before_refine = ensure_gray_uint8(final_mask)
    final_mask_ratio_before_refine = mask_ratio(final_mask_before_refine)
    if args.mask_refine != "none":
        final_mask = refine_mask(final_mask_before_refine, args.mask_refine)
    else:
        final_mask = final_mask_before_refine
    final_mask_ratio_after_refine = mask_ratio(final_mask)

    overlay_final = make_overlay(image_rgb, final_mask)

    sample_output_dir = output_root / image_path.stem / args.mask_source
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    save_rgb(sample_output_dir / "input.png", image_rgb)
    if args.cv_debug and cv_debug_images:
        save_debug_images(sample_output_dir, cv_debug_images)
    if probability_mask_uint8 is not None and (args.save_prob_mask or args.save_all_masks or args.mask_source in {"dl", "union"}):
        save_gray(sample_output_dir / "dl_prob_mask.png", probability_mask_uint8)

    primary_tag = threshold_to_tag(args.threshold)
    fallback_tag = threshold_to_tag(args.fallback_threshold)

    if dl_mask_primary is not None:
        save_gray(sample_output_dir / f"dl_mask_{primary_tag}.png", dl_mask_primary)
    if dl_mask_fallback is not None:
        save_gray(sample_output_dir / f"dl_mask_{fallback_tag}.png", dl_mask_fallback)
    if cv_mask is not None and (args.save_all_masks or args.mask_source in {"cv", "union"}):
        save_gray(sample_output_dir / "cv_mask.png", cv_mask)
    if union_mask is not None and (args.save_all_masks or args.mask_source == "union"):
        save_gray(sample_output_dir / "union_mask.png", union_mask)
    if external_mask is not None:
        save_gray(sample_output_dir / "external_mask.png", external_mask)

    save_gray(sample_output_dir / "final_mask_before_refine.png", final_mask_before_refine)
    if args.mask_refine != "none":
        save_gray(sample_output_dir / "final_mask_refined.png", final_mask)
    save_gray(sample_output_dir / "final_mask.png", final_mask)
    save_rgb(sample_output_dir / "overlay_final.png", overlay_final)

    backend_warning: str | None = None
    inpaint_failed: str | None = None
    actual_backend = "none" if args.no_inpaint else ""
    restored_final: np.ndarray | None = None
    restored_primary: np.ndarray | None = None
    restored_fallback: np.ndarray | None = None
    simple_lama: Any | None = None

    should_save_dl_restored = (
        not args.no_inpaint
        and dl_mask_primary is not None
        and (args.mask_source == "dl" or args.save_all_masks)
    )

    if not args.no_inpaint:
        selected_backend, backend_warning, simple_lama = resolve_inpaint_backend(args.backend)
        actual_backend = selected_backend
        try:
            restored_final = run_inpaint(image_rgb, final_mask, selected_backend, simple_lama)
            save_rgb(sample_output_dir / "restored_final.png", restored_final)

            if should_save_dl_restored and dl_mask_primary is not None:
                restored_primary = run_inpaint(image_rgb, dl_mask_primary, selected_backend, simple_lama)
                save_rgb(sample_output_dir / f"restored_{primary_tag}.png", restored_primary)
            if should_save_dl_restored and dl_mask_fallback is not None:
                restored_fallback = run_inpaint(image_rgb, dl_mask_fallback, selected_backend, simple_lama)
                save_rgb(sample_output_dir / f"restored_{fallback_tag}.png", restored_fallback)
        except Exception as exc:
            if args.backend == "auto" and selected_backend == "simple_lama":
                backend_warning = f"SimpleLama lỗi lúc chạy, fallback sang OpenCV: {exc}"
                actual_backend = "opencv"
                restored_final = run_inpaint(image_rgb, final_mask, "opencv")
                save_rgb(sample_output_dir / "restored_final.png", restored_final)
                if should_save_dl_restored and dl_mask_primary is not None:
                    restored_primary = run_inpaint(image_rgb, dl_mask_primary, "opencv")
                    save_rgb(sample_output_dir / f"restored_{primary_tag}.png", restored_primary)
                if should_save_dl_restored and dl_mask_fallback is not None:
                    restored_fallback = run_inpaint(image_rgb, dl_mask_fallback, "opencv")
                    save_rgb(sample_output_dir / f"restored_{fallback_tag}.png", restored_fallback)
            else:
                inpaint_failed = str(exc)
    else:
        actual_backend = "none"

    placeholder_shape = (min(max(original_height, 256), 768), min(max(original_width, 256), 768), 3)
    restored_final_for_grid = (
        restored_final
        if restored_final is not None
        else make_placeholder_tile(placeholder_shape, "restored final", "not available")
    )

    grid_tiles: list[tuple[str, np.ndarray | None]] = [
        ("input", image_rgb),
        (f"dl mask {args.threshold:.2f}", make_gray_rgb(dl_mask_primary, fallback_shape=(original_height, original_width))),
        ("cv mask", make_gray_rgb(cv_mask, fallback_shape=(original_height, original_width))),
        ("union mask", make_gray_rgb(union_mask, fallback_shape=(original_height, original_width))),
        ("final mask", make_gray_rgb(final_mask, fallback_shape=(original_height, original_width))),
        ("overlay final", overlay_final),
        ("restored final", restored_final_for_grid),
    ]
    comparison_grid = make_comparison_grid(grid_tiles, max_columns=4)
    save_rgb(sample_output_dir / "comparison_grid.png", comparison_grid)

    metadata = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch") if checkpoint else None,
        "checkpoint_metrics": checkpoint.get("metrics") if checkpoint else None,
        "threshold": float(args.threshold),
        "fallback_threshold": float(args.fallback_threshold),
        "mask_source": args.mask_source,
        "cv_profile": args.cv_profile,
        "cv_auto_invert_requested": bool(args.cv_auto_invert),
        "cv_mask_ratio_before_invert_check": cv_info.get("cv_mask_ratio_before_invert_check"),
        "cv_mask_ratio_after_invert_check": cv_info.get("cv_mask_ratio_after_invert_check"),
        "cv_auto_inverted": cv_info.get("cv_auto_inverted"),
        "cv_warning": cv_info.get("cv_warning"),
        "cv_pctl": cv_info.get("cv_pctl"),
        "cv_threshold": cv_info.get("cv_threshold"),
        "cv_area_pct_before_fallback": cv_info.get("cv_area_pct_before_fallback"),
        "cv_area_pct_final": cv_info.get("cv_area_pct_final"),
        "cv_fallback_used": cv_info.get("cv_fallback_used"),
        "cv_num_components": cv_info.get("cv_num_components"),
        "cv_num_components_kept": cv_info.get("cv_num_components_kept"),
        "mask_dilate": int(args.mask_dilate),
        "mask_refine": args.mask_refine,
        "mask_refine_applied": args.mask_refine != "none",
        "final_mask_ratio_before_refine": final_mask_ratio_before_refine,
        "final_mask_ratio_after_refine": final_mask_ratio_after_refine,
        "backend_requested": args.backend,
        "actual_backend": actual_backend,
        "backend_warning": backend_warning,
        "image_size_original": {"width": int(original_width), "height": int(original_height)},
        "image_size_model": int(effective_image_size),
        "dl_mask_ratio": mask_ratio(dl_mask_primary),
        "dl_mask_ratio_primary": mask_ratio(dl_mask_primary),
        "dl_mask_ratio_fallback": mask_ratio(dl_mask_fallback),
        f"dl_mask_ratio_{primary_tag}": mask_ratio(dl_mask_primary),
        f"dl_mask_ratio_{fallback_tag}": mask_ratio(dl_mask_fallback),
        "cv_mask_ratio": mask_ratio(cv_mask),
        "union_mask_ratio": mask_ratio(union_mask),
        "final_mask_ratio": mask_ratio(final_mask),
        "external_mask_path": str(external_mask_path) if external_mask_path else None,
        "inpaint_failed": inpaint_failed,
        "checkpoint_model_config": checkpoint_model_config or None,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    metadata_path = sample_output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"device: {device}")
    print(f"image: {image_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"mask_source: {args.mask_source}")
    print(f"output_dir: {sample_output_dir}")
    print(f"backend_requested: {args.backend}")
    print(f"actual_backend: {actual_backend}")
    if backend_warning:
        print(f"backend_warning: {backend_warning}")
    if inpaint_failed:
        print(f"inpaint_failed: {inpaint_failed}")
    print(f"dl_mask_ratio_{primary_tag}: {mask_ratio(dl_mask_primary)}")
    print(f"dl_mask_ratio_{fallback_tag}: {mask_ratio(dl_mask_fallback)}")
    print(f"cv_mask_ratio_before_invert_check: {cv_info.get('cv_mask_ratio_before_invert_check')}")
    print(f"cv_mask_ratio_after_invert_check: {cv_info.get('cv_mask_ratio_after_invert_check')}")
    print(f"cv_auto_inverted: {cv_info.get('cv_auto_inverted')}")
    print(f"cv_pctl: {cv_info.get('cv_pctl')}")
    print(f"cv_threshold: {cv_info.get('cv_threshold')}")
    print(f"cv_area_pct_before_fallback: {cv_info.get('cv_area_pct_before_fallback')}")
    print(f"cv_area_pct_final: {cv_info.get('cv_area_pct_final')}")
    print(f"cv_fallback_used: {cv_info.get('cv_fallback_used')}")
    print(f"cv_num_components: {cv_info.get('cv_num_components')}")
    print(f"cv_num_components_kept: {cv_info.get('cv_num_components_kept')}")
    if cv_info.get("cv_warning"):
        print(f"cv_warning: {cv_info.get('cv_warning')}")
    print(f"cv_mask_ratio: {mask_ratio(cv_mask)}")
    print(f"union_mask_ratio: {mask_ratio(union_mask)}")
    print(f"mask_refine: {args.mask_refine}")
    print(f"final_mask_ratio_before_refine: {final_mask_ratio_before_refine}")
    print(f"final_mask_ratio_after_refine: {final_mask_ratio_after_refine}")
    print(f"final_mask_ratio: {mask_ratio(final_mask)}")
    print(f"comparison_grid: {sample_output_dir / 'comparison_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
