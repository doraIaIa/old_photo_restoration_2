from __future__ import annotations

import random
from typing import Callable

import cv2
import numpy as np
import torch


def _to_image_tensor(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    image = np.clip(image, 0, 255).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).float()


def _to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {mask.shape}")
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask[None, :, :]).float()


class _FallbackSegmentationTransform:
    def __init__(self, split: str, image_size: int) -> None:
        self.split = split
        self.image_size = int(image_size)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict[str, torch.Tensor]:
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.split == "train":
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.flip(image, axis=1))
                mask = np.ascontiguousarray(np.flip(mask, axis=1))
            if random.random() < 0.1:
                image = np.ascontiguousarray(np.flip(image, axis=0))
                mask = np.ascontiguousarray(np.flip(mask, axis=0))
            if random.random() < 0.3:
                k = random.choice((1, 2, 3))
                image = np.ascontiguousarray(np.rot90(image, k=k))
                mask = np.ascontiguousarray(np.rot90(mask, k=k))

        return {
            "image": _to_image_tensor(image),
            "mask": _to_mask_tensor(mask),
        }


def _build_albumentations_transform(split: str, image_size: int) -> Callable:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transforms: list = [A.Resize(image_size, image_size)]
    if split == "train":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.3),
            ]
        )
    transforms.extend(
        [
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(transpose_mask=True),
        ]
    )
    return A.Compose(transforms)


def get_segmentation_transforms(split: str, image_size: int = 512) -> Callable:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    try:
        return _build_albumentations_transform(split=split, image_size=int(image_size))
    except Exception:
        return _FallbackSegmentationTransform(split=split, image_size=int(image_size))
