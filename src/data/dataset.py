from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không thể đọc ảnh RGB: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không thể đọc mask: {path}")
    return mask


def _default_to_tensor(image: np.ndarray, mask: np.ndarray) -> dict[str, torch.Tensor]:
    image_tensor = torch.from_numpy(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))).float()
    mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)[None, :, :]).float()
    return {"image": image_tensor, "mask": mask_tensor}


class CrackSegDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        transform: Any = None,
        return_paths: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.return_paths = return_paths

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset_root không tồn tại: {self.dataset_root}")
        if self.split not in {"train", "val"}:
            raise ValueError(f"split phải là 'train' hoặc 'val', nhận được {self.split!r}")

        self.split_root = self.dataset_root / self.split
        self.images_dir = self.split_root / "images"
        self.masks_dir = self.split_root / "masks"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Thiếu thư mục images: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Thiếu thư mục masks: {self.masks_dir}")

        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError(f"Không tìm thấy sample nào trong split {self.split!r} tại {self.split_root}")

    def _load_samples(self) -> list[dict[str, Path]]:
        manifest_path = self.dataset_root / "manifest.csv"
        if manifest_path.exists():
            return self._load_samples_from_manifest(manifest_path)
        return self._load_samples_from_directories()

    def _load_samples_from_manifest(self, manifest_path: Path) -> list[dict[str, Path]]:
        samples: list[dict[str, Path]] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("split") != self.split:
                    continue
                image_path = self.dataset_root / row["degraded_path"]
                mask_path = self.dataset_root / row["mask_path"]
                if not image_path.exists():
                    raise FileNotFoundError(f"Thiếu ảnh theo manifest: {image_path}")
                if not mask_path.exists():
                    raise FileNotFoundError(f"Thiếu mask theo manifest: {mask_path}")
                samples.append(
                    {
                        "sample_id": row["sample_id"],
                        "image_path": image_path,
                        "mask_path": mask_path,
                    }
                )
        return samples

    def _load_samples_from_directories(self) -> list[dict[str, Path]]:
        image_paths = sorted(path for path in self.images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        mask_paths = sorted(path for path in self.masks_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

        if not image_paths:
            raise ValueError(f"Không có ảnh nào trong {self.images_dir}")
        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Số lượng images ({len(image_paths)}) và masks ({len(mask_paths)}) không khớp trong split {self.split}"
            )

        mask_by_stem = {path.stem: path for path in mask_paths}
        samples: list[dict[str, Path]] = []
        for image_path in image_paths:
            mask_path = mask_by_stem.get(image_path.stem)
            if mask_path is None:
                raise FileNotFoundError(f"Không tìm thấy mask tương ứng cho ảnh: {image_path.name}")
            samples.append(
                {
                    "sample_id": image_path.stem,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = _load_rgb(sample["image_path"])
        mask = _load_mask(sample["mask_path"])

        if self.transform is None:
            transformed = _default_to_tensor(image, mask)
        else:
            transformed = self.transform(image=image, mask=mask)

        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]

        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.as_tensor(image_tensor)
        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.as_tensor(mask_tensor)

        if image_tensor.ndim == 3 and image_tensor.shape[0] != 3 and image_tensor.shape[-1] == 3:
            image_tensor = image_tensor.permute(2, 0, 1)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1 and mask_tensor.shape[-1] == 1:
            mask_tensor = mask_tensor.permute(2, 0, 1)

        image_tensor = image_tensor.float()
        mask_tensor = mask_tensor.float()

        if float(image_tensor.max()) > 1.0:
            image_tensor = image_tensor / 255.0
        if float(mask_tensor.max()) > 1.0:
            mask_tensor = (mask_tensor > 127).float()
        else:
            mask_tensor = (mask_tensor > 0.5).float()

        output: dict[str, Any] = {
            "image": image_tensor,
            "mask": mask_tensor,
        }
        if self.return_paths:
            output["image_path"] = str(sample["image_path"])
            output["mask_path"] = str(sample["mask_path"])
        return output
