#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

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

from scripts.train_r013_finetune import load_model, resolve_device, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sinh mask/overlay demo thật cho r011 và r013.")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--r011-checkpoint", required=True)
    parser.add_argument("--r013-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--r011-threshold", type=float, required=True)
    parser.add_argument("--r013-threshold", type=float, default=0.50)
    parser.add_argument("--r013-sensitive-threshold", type=float, default=0.40)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def save_gray(path: Path, image_gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.clip(image_gray, 0, 255).astype(np.uint8)):
        raise RuntimeError(f"Không ghi được mask: {path}")


def image_tensor(image_rgb: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(tensor, (2, 0, 1))).float().unsqueeze(0)
    return tensor.to(device)


def overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image_rgb.astype(np.float32).copy()
    red = np.zeros_like(result)
    red[:, :, 0] = 255
    alpha = (mask > 0).astype(np.float32)[:, :, None] * 0.45
    return np.clip(result * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)


@torch.no_grad()
def predict_mask(model: torch.nn.Module, image_rgb: np.ndarray, image_size: int, threshold: float, device: torch.device) -> np.ndarray:
    logits = model(image_tensor(image_rgb, image_size, device))
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    prob = cv2.resize(prob, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (prob >= threshold).astype(np.uint8) * 255


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    r011, _ = load_model(resolve_path(args.r011_checkpoint), device)
    r013, _ = load_model(resolve_path(args.r013_checkpoint), device)
    r011.eval()
    r013.eval()
    output_root = resolve_path(args.output_dir)
    rows = []
    for image_text in args.images:
        image_path = resolve_path(image_text)
        image_rgb = read_rgb(image_path)
        sample_dir = output_root / image_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, sample_dir / image_path.name)
        save_rgb(sample_dir / "input.png", image_rgb)
        r011_mask = predict_mask(r011, image_rgb, args.image_size, args.r011_threshold, device)
        r013_mask = predict_mask(r013, image_rgb, args.image_size, args.r013_threshold, device)
        r013_sensitive = predict_mask(r013, image_rgb, args.image_size, args.r013_sensitive_threshold, device)
        save_gray(sample_dir / "r011_mask.png", r011_mask)
        save_gray(sample_dir / "r013_mask_t050.png", r013_mask)
        save_gray(sample_dir / "r013_mask_sensitive.png", r013_sensitive)
        save_rgb(sample_dir / "r011_overlay.png", overlay(image_rgb, r011_mask))
        save_rgb(sample_dir / "r013_overlay_t050.png", overlay(image_rgb, r013_mask))
        save_rgb(sample_dir / "r013_overlay_sensitive.png", overlay(image_rgb, r013_sensitive))
        rows.append(
            {
                "image": str(image_path),
                "output_dir": str(sample_dir),
                "r011_threshold": args.r011_threshold,
                "r013_threshold": args.r013_threshold,
                "r013_sensitive_threshold": args.r013_sensitive_threshold,
                "r011_mask_ratio": float(np.count_nonzero(r011_mask) / r011_mask.size),
                "r013_mask_ratio": float(np.count_nonzero(r013_mask) / r013_mask.size),
                "r013_sensitive_mask_ratio": float(np.count_nonzero(r013_sensitive) / r013_sensitive.size),
            }
        )
    with (output_root / "demo_mask_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"device": str(device), "rows": rows}, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"device": str(device), "rows": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
