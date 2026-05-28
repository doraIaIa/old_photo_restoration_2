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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy demo end-to-end cho segmentation và inpainting ảnh cũ.")
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh đầu vào.")
    parser.add_argument("--checkpoint", required=True, help="Đường dẫn checkpoint `best_iou.ckpt`.")
    parser.add_argument("--threshold", type=float, default=0.90, help="Ngưỡng chính để nhị phân hóa mask.")
    parser.add_argument("--fallback-threshold", type=float, default=0.70, help="Ngưỡng fallback để nhị phân hóa mask.")
    parser.add_argument("--output-dir", default="outputs/demo/r009_lama", help="Thư mục gốc để lưu demo output.")
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda.")
    parser.add_argument("--image-size", type=int, default=512, help="Kích thước resize cho segmentation inference.")
    parser.add_argument("--backend", default="auto", choices=["auto", "simple_lama", "opencv"], help="Backend inpaint.")
    parser.add_argument("--save-prob-mask", action="store_true", help="Lưu file `prob_mask.png`.")
    parser.add_argument("--no-inpaint", action="store_true", help="Chỉ chạy segmentation, không inpaint.")
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


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Không ghi được ảnh RGB: {path}")


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Không ghi được ảnh grayscale: {path}")


def threshold_to_tag(value: float) -> str:
    return f"t{value:.2f}".replace(".", "p")


def parse_run_id(checkpoint_path: Path) -> str | None:
    parent = checkpoint_path.parent
    if parent.name:
        return parent.name
    return None


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


def resize_binary_mask(binary_mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_hw
    return cv2.resize(binary_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)


def probability_to_uint8(probability_mask: np.ndarray) -> np.ndarray:
    return np.clip(np.round(probability_mask * 255.0), 0, 255).astype(np.uint8)


def binary_mask_from_probability(probability_mask: np.ndarray, threshold: float) -> np.ndarray:
    return (probability_mask >= threshold).astype(np.uint8) * 255


def make_overlay(image_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255.0
    weight = (binary_mask.astype(np.float32) / 255.0)[:, :, None] * alpha
    mixed = overlay * (1.0 - weight) + red * weight
    return np.clip(mixed, 0, 255).astype(np.uint8)


def make_gray_rgb(gray_image: np.ndarray) -> np.ndarray:
    return np.repeat(gray_image[:, :, None], 3, axis=2)


def make_placeholder_tile(reference_shape: tuple[int, int, int], label: str, detail: str) -> np.ndarray:
    tile = np.full(reference_shape, 24, dtype=np.uint8)
    cv2.putText(tile, label, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(tile, detail, (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return tile


def add_label(image_rgb: np.ndarray, label: str) -> np.ndarray:
    banner_height = 40
    banner = np.full((banner_height, image_rgb.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(banner, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
    return np.concatenate([banner, image_rgb], axis=0)


def make_comparison_grid(tiles: list[tuple[str, np.ndarray]]) -> np.ndarray:
    if len(tiles) != 8:
        raise ValueError(f"Comparison grid cần đúng 8 tile, nhận được {len(tiles)}")
    labeled_tiles = [add_label(image, label) for label, image in tiles]
    first_row = np.concatenate(labeled_tiles[:4], axis=1)
    second_row = np.concatenate(labeled_tiles[4:], axis=1)
    return np.concatenate([first_row, second_row], axis=0)


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

    image_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(binary_mask).convert("L")
    restored = simple_lama(image_pil, mask_pil)
    if not isinstance(restored, Image.Image):
        raise TypeError(f"SimpleLama trả về kiểu không mong đợi: {type(restored)!r}")
    return np.array(restored.convert("RGB"))


def inpaint_with_opencv(image_rgb: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    restored_bgr = cv2.inpaint(image_bgr, binary_mask, 3, cv2.INPAINT_TELEA)
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

    config = load_config(resolve_path(args.config))
    device = resolve_device(args.device)
    image_path = resolve_path(args.image)
    checkpoint_path = resolve_path(args.checkpoint)
    output_root = resolve_path(args.output_dir)

    image_rgb = load_rgb(image_path)
    original_height, original_width = image_rgb.shape[:2]

    model, checkpoint = load_model(checkpoint_path, device)
    checkpoint_model_config = checkpoint.get("model_config") or {}
    effective_image_size = int(args.image_size or config.get("build", {}).get("image_size", 512))

    input_tensor = build_inference_tensor(image_rgb, effective_image_size).to(device)
    logits = model(input_tensor)
    probabilities = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
    if probabilities.ndim != 2:
        raise ValueError(f"Probability mask phải có 2 chiều, nhận được shape {probabilities.shape}")

    probability_mask = resize_probability_mask(probabilities, (original_height, original_width))
    probability_mask = np.clip(probability_mask, 0.0, 1.0)
    probability_mask_uint8 = probability_to_uint8(probability_mask)

    primary_mask = binary_mask_from_probability(probability_mask, args.threshold)
    fallback_mask = binary_mask_from_probability(probability_mask, args.fallback_threshold)

    primary_overlay = make_overlay(image_rgb, primary_mask)
    fallback_overlay = make_overlay(image_rgb, fallback_mask)

    sample_output_dir = output_root / image_path.stem
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    save_rgb(sample_output_dir / "input.png", image_rgb)
    if args.save_prob_mask:
        save_gray(sample_output_dir / "prob_mask.png", probability_mask_uint8)

    primary_tag = threshold_to_tag(args.threshold)
    fallback_tag = threshold_to_tag(args.fallback_threshold)

    save_gray(sample_output_dir / f"mask_{primary_tag}.png", primary_mask)
    save_rgb(sample_output_dir / f"overlay_{primary_tag}.png", primary_overlay)
    save_gray(sample_output_dir / f"mask_{fallback_tag}.png", fallback_mask)
    save_rgb(sample_output_dir / f"overlay_{fallback_tag}.png", fallback_overlay)

    backend_warning: str | None = None
    inpaint_failed: str | None = None
    actual_backend = "none" if args.no_inpaint else ""
    restored_primary: np.ndarray | None = None
    restored_fallback: np.ndarray | None = None

    simple_lama: Any | None = None

    if not args.no_inpaint:
        selected_backend, backend_warning, simple_lama = resolve_inpaint_backend(args.backend)
        actual_backend = selected_backend
        try:
            restored_primary = run_inpaint(image_rgb, primary_mask, selected_backend, simple_lama)
            restored_fallback = run_inpaint(image_rgb, fallback_mask, selected_backend, simple_lama)
            save_rgb(sample_output_dir / f"restored_{primary_tag}.png", restored_primary)
            save_rgb(sample_output_dir / f"restored_{fallback_tag}.png", restored_fallback)
        except Exception as exc:
            if args.backend == "auto" and selected_backend == "simple_lama":
                backend_warning = f"SimpleLama lỗi lúc chạy, fallback sang OpenCV: {exc}"
                actual_backend = "opencv"
                restored_primary = run_inpaint(image_rgb, primary_mask, "opencv")
                restored_fallback = run_inpaint(image_rgb, fallback_mask, "opencv")
                save_rgb(sample_output_dir / f"restored_{primary_tag}.png", restored_primary)
                save_rgb(sample_output_dir / f"restored_{fallback_tag}.png", restored_fallback)
            else:
                inpaint_failed = str(exc)
    else:
        actual_backend = "none"

    if restored_primary is None:
        restored_primary = make_placeholder_tile(image_rgb.shape, f"restored {primary_tag}", "inpaint unavailable")
    if restored_fallback is None:
        restored_fallback = make_placeholder_tile(image_rgb.shape, f"restored {fallback_tag}", "inpaint unavailable")

    comparison_grid = make_comparison_grid(
        [
            ("input", image_rgb),
            ("prob mask", make_gray_rgb(probability_mask_uint8)),
            (f"mask {primary_tag}", make_gray_rgb(primary_mask)),
            (f"overlay {primary_tag}", primary_overlay),
            (f"restored {primary_tag}", restored_primary),
            (f"mask {fallback_tag}", make_gray_rgb(fallback_mask)),
            (f"overlay {fallback_tag}", fallback_overlay),
            (f"restored {fallback_tag}", restored_fallback),
        ]
    )
    save_rgb(sample_output_dir / "comparison_grid.png", comparison_grid)

    metadata = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "run_id": parse_run_id(checkpoint_path),
        "threshold": float(args.threshold),
        "fallback_threshold": float(args.fallback_threshold),
        "requested_backend": args.backend,
        "actual_backend": actual_backend,
        "backend_warning": backend_warning,
        "no_inpaint": bool(args.no_inpaint),
        "inpaint_failed": inpaint_failed,
        "image_size": int(effective_image_size),
        "device": str(device),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "mask_ratio_t0p90": float((primary_mask > 0).mean()),
        "mask_ratio_t0p70": float((fallback_mask > 0).mean()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "checkpoint_model_config": checkpoint_model_config,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    metadata_path = sample_output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"device: {device}")
    print(f"image: {image_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"output_dir: {sample_output_dir}")
    print(f"requested_backend: {args.backend}")
    print(f"actual_backend: {actual_backend}")
    if backend_warning:
        print(f"backend_warning: {backend_warning}")
    if inpaint_failed:
        print(f"inpaint_failed: {inpaint_failed}")
    print(f"mask_ratio_{primary_tag}: {(primary_mask > 0).mean():.6f}")
    print(f"mask_ratio_{fallback_tag}: {(fallback_mask > 0).mean():.6f}")
    print(f"comparison_grid: {sample_output_dir / 'comparison_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
