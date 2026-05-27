from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CrackSegDataset
from src.data.transforms import get_segmentation_transforms
from src.models.segmenter import CrackSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export panel dự đoán segmentation để kiểm tra trực quan.")
    parser.add_argument("--run-id", required=True, help="Run ID chứa checkpoint `best_iou.ckpt`.")
    parser.add_argument("--dataset-id", required=True, help="Dataset ID dùng để export prediction.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Split cần export.")
    parser.add_argument("--num-samples", type=int, default=24, help="Số sample tối đa cần export.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng nhị phân hóa xác suất dự đoán.")
    parser.add_argument("--device", default="auto", help="auto, cpu hoặc cuda.")
    parser.add_argument("--config", default="configs/data.yaml", help="Đường dẫn config YAML.")
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


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    return mask


def load_model(checkpoint_path: Path, device: torch.device) -> CrackSegmenter:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint thiếu `model_state_dict`: {checkpoint_path}")

    model = CrackSegmenter().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_overlay(image: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    alpha = (pred_mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
    overlay = overlay * (1.0 - alpha) + red * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_panel(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    if image.shape[:2] != gt_mask.shape[:2] or image.shape[:2] != pred_mask.shape[:2]:
        raise ValueError(
            f"Shape mismatch khi tạo panel: image={image.shape}, gt_mask={gt_mask.shape}, pred_mask={pred_mask.shape}"
        )
    gt_rgb = np.repeat(gt_mask[:, :, None], 3, axis=2)
    pred_rgb = np.repeat(pred_mask[:, :, None], 3, axis=2)
    overlay = make_overlay(image, pred_mask)
    return np.concatenate([image, gt_rgb, pred_rgb, overlay], axis=1)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold phải nằm trong [0, 1].")

    config = load_config(PROJECT_ROOT / args.config)
    device = resolve_device(args.device)
    dataset_root = PROJECT_ROOT / config["processed"]["root"] / args.dataset_id
    if not dataset_root.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset root: {dataset_root}")

    checkpoint_path = PROJECT_ROOT / config["checkpoints"]["root"] / "segmenter" / args.run_id / "best_iou.ckpt"
    model = load_model(checkpoint_path, device)

    image_size = int(config["build"]["image_size"])
    dataset = CrackSegDataset(
        dataset_root=dataset_root,
        split=args.split,
        transform=get_segmentation_transforms(split="val", image_size=image_size),
        return_paths=True,
    )

    output_root = PROJECT_ROOT / config["outputs"]["debug_root"] / "segmentation" / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)

    num_samples = min(args.num_samples, len(dataset))
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"dataset_root: {dataset_root}")
    print(f"output_root: {output_root}")
    print(f"num_samples: {num_samples}")

    for index in range(num_samples):
        sample = dataset[index]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        pred_mask = (probabilities >= args.threshold).float()

        pred_mask_np = (pred_mask.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
        gt_mask = load_mask(Path(sample["mask_path"]))
        image = load_rgb(Path(sample["image_path"]))

        if logits.shape[-2:] != gt_mask.shape or image.shape[:2] != gt_mask.shape:
            raise ValueError(
                "Shape mismatch giữa logits, input image và ground-truth mask: "
                f"logits={tuple(logits.shape)}, image={image.shape}, mask={gt_mask.shape}"
            )

        panel = make_panel(image, gt_mask, pred_mask_np)
        save_rgb(output_root / f"pred_{index + 1:06d}.png", panel)

    print(f"prediction_export_root: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
