from __future__ import annotations

import random
from typing import Callable

import cv2
import numpy as np
import torch


VALID_AUG_PROFILES = {"baseline", "strong"}


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


def _apply_brightness_contrast(image: np.ndarray, brightness_limit: float = 0.12, contrast_limit: float = 0.12) -> np.ndarray:
    alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)
    beta = random.uniform(-brightness_limit, brightness_limit) * 255.0
    adjusted = image.astype(np.float32) * alpha + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _rotate_image_and_mask(image: np.ndarray, mask: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_image, rotated_mask


def _elastic_distort(image: np.ndarray, mask: np.ndarray, alpha: float = 18.0, sigma: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    displacement_x = np.random.randn(height, width).astype(np.float32)
    displacement_y = np.random.randn(height, width).astype(np.float32)
    displacement_x = cv2.GaussianBlur(displacement_x, (0, 0), sigmaX=sigma, sigmaY=sigma) * alpha
    displacement_y = cv2.GaussianBlur(displacement_y, (0, 0), sigmaX=sigma, sigmaY=sigma) * alpha

    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + displacement_x
    map_y = grid_y + displacement_y

    warped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_mask = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_image, warped_mask


class _FallbackSegmentationTransform:
    def __init__(self, split: str, image_size: int, aug_profile: str) -> None:
        self.split = split
        self.image_size = int(image_size)
        self.aug_profile = aug_profile

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

            if self.aug_profile == "baseline":
                if random.random() < 0.3:
                    k = random.choice((1, 2, 3))
                    image = np.ascontiguousarray(np.rot90(image, k=k))
                    mask = np.ascontiguousarray(np.rot90(mask, k=k))
            elif self.aug_profile == "strong":
                if random.random() < 0.7:
                    angle = random.uniform(-45.0, 45.0)
                    image, mask = _rotate_image_and_mask(image, mask, angle)
                if random.random() < 0.4:
                    image = _apply_brightness_contrast(image)
                if random.random() < 0.15:
                    image, mask = _elastic_distort(image, mask)

        return {
            "image": _to_image_tensor(image),
            "mask": _to_mask_tensor(mask),
        }


def _build_albumentations_transform(split: str, image_size: int, aug_profile: str) -> Callable:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transforms: list = [A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)]
    if split == "train":
        if aug_profile == "baseline":
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.1),
                    A.RandomRotate90(p=0.3),
                ]
            )
        elif aug_profile == "strong":
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.Affine(
                        rotate=(-45, 45),
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        mode=cv2.BORDER_REFLECT_101,
                        p=0.7,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.12,
                        contrast_limit=0.12,
                        p=0.4,
                    ),
                    A.ElasticTransform(
                        alpha=18.0,
                        sigma=5.0,
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.15,
                    ),
                ]
            )
    transforms.extend(
        [
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(transpose_mask=True),
        ]
    )
    return A.Compose(transforms)


def get_segmentation_transforms(split: str, image_size: int = 512, aug_profile: str = "baseline") -> Callable:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if aug_profile not in VALID_AUG_PROFILES:
        raise ValueError(f"aug_profile must be one of {sorted(VALID_AUG_PROFILES)}, got {aug_profile!r}")

    effective_profile = aug_profile if split == "train" else "baseline"
    try:
        return _build_albumentations_transform(
            split=split,
            image_size=int(image_size),
            aug_profile=effective_profile,
        )
    except Exception:
        return _FallbackSegmentationTransform(
            split=split,
            image_size=int(image_size),
            aug_profile=effective_profile,
        )
