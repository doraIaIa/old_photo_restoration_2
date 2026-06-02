from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from scripts.evaluate_real_segmentation import find_image_path, load_mask, load_model, load_rgb, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo comparison grid giữa hai segmentation checkpoints.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--label-a", default="r011")
    parser.add_argument("--label-b", default="r012")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks_repair_manual")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold-a", type=float, default=0.5)
    parser.add_argument("--threshold-b", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def read_split_ids(path: Path) -> list[str]:
    ids = [line.strip().lstrip("\ufeff") for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"Split rỗng: {path}")
    return ids


def to_tensor(image_rgb: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0)
    return tensor.to(device)


@torch.no_grad()
def predict_mask(model: torch.nn.Module, image_rgb: np.ndarray, image_size: int, threshold: float, device: torch.device) -> np.ndarray:
    logits = model(to_tensor(image_rgb, image_size, device))
    probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return (probability >= threshold).astype(np.uint8) * 255


def overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    canvas = image_rgb.astype(np.float32).copy()
    color_layer = np.zeros_like(canvas)
    color_layer[:, :, 0] = color[0]
    color_layer[:, :, 1] = color[1]
    color_layer[:, :, 2] = color[2]
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
    return np.clip(canvas * (1.0 - alpha) + color_layer * alpha, 0, 255).astype(np.uint8)


def draw_label(image_rgb: np.ndarray, text: str) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image_bgr, (0, 0), (image_bgr.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(image_bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def make_grid(image_rgb: np.ndarray, gt_mask: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, label_a: str, label_b: str) -> np.ndarray:
    image_size = image_rgb.shape[0]
    gt_overlay = overlay(image_rgb, gt_mask, (255, 0, 0))
    overlay_a = overlay(image_rgb, mask_a, (255, 0, 0))
    overlay_b = overlay(image_rgb, mask_b, (0, 255, 0))
    panels = [
        draw_label(image_rgb, "image"),
        draw_label(gt_overlay, "manual GT"),
        draw_label(overlay_a, label_a),
        draw_label(overlay_b, label_b),
    ]
    return np.concatenate([cv2.resize(panel, (image_size, image_size), interpolation=cv2.INTER_LINEAR) for panel in panels], axis=1)


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    split_file = Path(args.split_file).resolve()
    checkpoint_a = Path(args.checkpoint_a).resolve()
    checkpoint_b = Path(args.checkpoint_b).resolve()
    output_dir = Path(args.output_dir).resolve()
    device = resolve_device(args.device)

    model_a = load_model(checkpoint_a, device)
    model_b = load_model(checkpoint_b, device)
    ids = read_split_ids(split_file)
    rows: list[dict[str, Any]] = []
    for sample_id in ids:
        image_path = find_image_path(data_root / args.image_dir, sample_id)
        mask_path = data_root / args.mask_dir / f"{sample_id}_mask.png"
        image_rgb = cv2.resize(load_rgb(image_path), (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        gt_mask = cv2.resize(load_mask(mask_path), (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
        mask_a = predict_mask(model_a, image_rgb, args.image_size, args.threshold_a, device)
        mask_b = predict_mask(model_b, image_rgb, args.image_size, args.threshold_b, device)
        grid = make_grid(image_rgb, gt_mask, mask_a, mask_b, args.label_a, args.label_b)
        output_path = output_dir / f"{sample_id}_{args.label_a}_vs_{args.label_b}.png"
        save_rgb(output_path, grid)
        rows.append(
            {
                "id": sample_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "grid_path": str(output_path),
                "threshold_a": args.threshold_a,
                "threshold_b": args.threshold_b,
            }
        )

    summary = {
        "data_root": str(data_root),
        "split_file": str(split_file),
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "threshold_a": args.threshold_a,
        "threshold_b": args.threshold_b,
        "sample_count": len(rows),
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sample_count: {len(rows)}")
    print(f"output_dir: {output_dir}")
    print(f"summary_json: {output_dir / 'comparison_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
