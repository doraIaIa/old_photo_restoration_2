from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.models.segmenter import CrackSegmenter


IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation checkpoint trên real old-photo split.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not thresholds:
        raise ValueError("--thresholds không được rỗng.")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold phải nằm trong [0, 1], nhận được {threshold}")
    return thresholds


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


def load_torch_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint phải là dict, nhận được: {type(checkpoint)!r}")
    return checkpoint


def load_model(checkpoint_path: Path, device: torch.device) -> CrackSegmenter:
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
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
    return model


def find_image_path(images_dir: Path, sample_id: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{sample_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Không tìm thấy ảnh cho id={sample_id} trong {images_dir}")


def read_split_ids(split_file: Path) -> list[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Không tìm thấy split file: {split_file}")
    ids = [line.strip().lstrip("\ufeff") for line in split_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"Split file rỗng: {split_file}")
    return ids


def load_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh RGB: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    return mask


def to_image_tensor(image: np.ndarray) -> torch.Tensor:
    image = np.clip(image, 0, 255).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).float()


def to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
    mask = (mask > 127).astype(np.float32)
    return torch.from_numpy(mask[None, :, :]).float()


class RealOldPhotoEvalDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split_file: str | Path,
        image_size: int,
        image_dir: str = "images",
        mask_dir: str = "masks",
    ) -> None:
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.image_size = int(image_size)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images_dir = self.data_root / image_dir
        self.masks_dir = self.data_root / mask_dir

        if self.image_size <= 0:
            raise ValueError("image_size phải > 0.")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy images/: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy masks/: {self.masks_dir}")

        self.samples: list[dict[str, Any]] = []
        for sample_id in read_split_ids(self.split_file):
            image_path = find_image_path(self.images_dir, sample_id)
            mask_path = self.masks_dir / f"{sample_id}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Không tìm thấy mask cho id={sample_id}: {mask_path}")
            self.samples.append({"id": sample_id, "image_path": image_path, "mask_path": mask_path})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = load_rgb(sample["image_path"])
        mask = load_mask(sample["mask_path"])
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return {
            "id": sample["id"],
            "image": to_image_tensor(image),
            "mask": to_mask_tensor(mask),
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
        }


def update_counts(
    counts: dict[float, dict[str, float]],
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    thresholds: list[float],
) -> None:
    targets = (targets >= 0.5).float()
    for threshold in thresholds:
        predictions = (probabilities >= threshold).float()
        true_positive = float((predictions * targets).sum().detach().cpu())
        false_positive = float((predictions * (1.0 - targets)).sum().detach().cpu())
        false_negative = float(((1.0 - predictions) * targets).sum().detach().cpu())
        counts[threshold]["tp"] += true_positive
        counts[threshold]["fp"] += false_positive
        counts[threshold]["fn"] += false_negative


def metrics_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-6) -> dict[str, float]:
    iou = (tp + eps) / (tp + fp + fn + eps)
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


def write_threshold_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "iou", "f1", "precision", "recall", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def make_overlay(image_rgb: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 0] = 255
    alpha = (pred_mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
    overlay = overlay * (1.0 - alpha) + red_layer * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


@torch.no_grad()
def collect_probabilities(
    model: CrackSegmenter,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
) -> tuple[dict[float, dict[str, float]], dict[str, np.ndarray]]:
    counts = {threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for threshold in thresholds}
    probabilities_by_id: dict[str, np.ndarray] = {}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.sigmoid(logits)
        update_counts(counts, probabilities, masks, thresholds)

        batch_ids = list(batch["id"])
        for item_index, sample_id in enumerate(batch_ids):
            probabilities_by_id[str(sample_id)] = probabilities[item_index, 0].detach().cpu().numpy()

    return counts, probabilities_by_id


def export_predictions(
    dataset: RealOldPhotoEvalDataset,
    probabilities_by_id: dict[str, np.ndarray],
    output_dir: Path,
    threshold: float,
    image_size: int,
) -> None:
    overlays_dir = output_dir / "overlays"
    pred_masks_dir = output_dir / "pred_masks"
    threshold_label = f"{threshold:.2f}".replace(".", "p")

    for sample in dataset.samples:
        sample_id = str(sample["id"])
        probability = probabilities_by_id[sample_id]
        pred_mask = (probability >= threshold).astype(np.uint8) * 255

        image = load_rgb(sample["image_path"])
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        overlay = make_overlay(image, pred_mask)

        save_rgb(overlays_dir / f"{sample_id}_overlay.png", overlay)
        save_mask(pred_masks_dir / f"{sample_id}_pred_t{threshold_label}.png", pred_mask)


def main() -> int:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    device = resolve_device(args.device)

    data_root = Path(args.data_root).resolve()
    split_file = Path(args.split_file).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path, device)
    dataset = RealOldPhotoEvalDataset(
        data_root,
        split_file,
        image_size=args.image_size,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    counts, probabilities_by_id = collect_probabilities(model, loader, device, thresholds)

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_counts = counts[threshold]
        metrics = metrics_from_counts(
            tp=threshold_counts["tp"],
            fp=threshold_counts["fp"],
            fn=threshold_counts["fn"],
        )
        rows.append(
            {
                "threshold": f"{threshold:.2f}",
                "iou": f"{metrics['iou']:.6f}",
                "f1": f"{metrics['f1']:.6f}",
                "precision": f"{metrics['precision']:.6f}",
                "recall": f"{metrics['recall']:.6f}",
                "tp": f"{threshold_counts['tp']:.0f}",
                "fp": f"{threshold_counts['fp']:.0f}",
                "fn": f"{threshold_counts['fn']:.0f}",
            }
        )

    best_row = max(rows, key=lambda row: float(row["iou"]))
    best_threshold = float(best_row["threshold"])
    write_threshold_csv(output_dir / "threshold_sweep.csv", rows)
    export_predictions(dataset, probabilities_by_id, output_dir, best_threshold, args.image_size)

    summary = {
        "data_root": str(data_root),
        "image_dir": args.image_dir,
        "mask_dir": args.mask_dir,
        "split_file": str(split_file),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "image_size": args.image_size,
        "sample_count": len(dataset),
        "thresholds": thresholds,
        "best_threshold_by_iou": best_threshold,
        "best_metrics": {
            "iou": float(best_row["iou"]),
            "f1": float(best_row["f1"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
        },
        "threshold_sweep_csv": str(output_dir / "threshold_sweep.csv"),
        "overlays_dir": str(output_dir / "overlays"),
        "pred_masks_dir": str(output_dir / "pred_masks"),
    }
    dump_json(output_dir / "summary.json", summary)

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda_device: {torch.cuda.get_device_name(device)}")
    print(f"sample_count: {len(dataset)}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"threshold_sweep_csv: {output_dir / 'threshold_sweep.csv'}")
    print(f"summary_json: {output_dir / 'summary.json'}")
    print(
        "best_by_iou: "
        f"threshold={best_row['threshold']} "
        f"iou={best_row['iou']} "
        f"f1={best_row['f1']} "
        f"precision={best_row['precision']} "
        f"recall={best_row['recall']}"
    )
    print(f"overlays: {output_dir / 'overlays'}")
    print(f"pred_masks: {output_dir / 'pred_masks'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
