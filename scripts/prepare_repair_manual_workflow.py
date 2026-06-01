from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chuẩn bị workflow annotate masks_repair_manual cho r012.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--candidate-count", type=int, default=15)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def find_image(images_dir: Path, sample_id: str) -> Path | None:
    for ext in (".jpg", ".png", ".jpeg"):
        p = images_dir / f"{sample_id}{ext}"
        if p.exists():
            return p
    return None


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return mask


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def overlay(image: np.ndarray, mask: np.ndarray | None, color=(255, 0, 0)) -> np.ndarray:
    image_rgb = image.copy().astype(np.float32)
    if mask is None:
        return image.astype(np.uint8)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    layer = np.zeros_like(image_rgb)
    layer[:, :] = np.array(color, dtype=np.float32)
    alpha = ((mask > 127).astype(np.float32) * 0.45)[:, :, None]
    return np.clip(image_rgb * (1 - alpha) + layer * alpha, 0, 255).astype(np.uint8)


def choose_candidates(data_root: Path, count: int) -> list[str]:
    masks_dir = data_root / "masks"
    ratios: list[tuple[str, float]] = []
    for idx in range(1, 61):
        sample_id = f"{idx:03d}"
        mask_path = masks_dir / f"{sample_id}_mask.png"
        mask = load_mask(mask_path) if mask_path.exists() else None
        ratio = float((mask > 127).mean()) if mask is not None else 0.0
        ratios.append((sample_id, ratio))
    ratios.sort(key=lambda item: item[1], reverse=True)
    if all(ratio == 0.0 for _, ratio in ratios):
        return [f"{idx:03d}" for idx in range(1, min(count, 15) + 1)]
    return [sample_id for sample_id, _ in ratios[:count]]


def write_guide(path: Path, data_root: Path) -> None:
    manual_dir = data_root / "masks_repair_manual"
    content = f"""# Repair Manual Annotation Guide

## Mask convention

- Background = `0`
- Repair region = `255`

## Annotate

- Phủ lõi nứt.
- Phủ viền xám/trắng quanh nứt.
- Nối đoạn crack bị đứt nếu là cùng một vết.
- Phủ rách viền ảnh.
- Không phủ chi tiết quan trọng nếu không hư.
- Mask nên là repair region cho LaMa, không phải chỉ centerline.

## Filename

- `001_mask.png`
- `002_mask.png`

## Folder

`{manual_dir}`

Không train r012 cho đến khi folder này có đủ manual repair masks đã audit pass.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    repair_v1_dir = data_root / "masks_repair_repair_v1"
    manual_dir = data_root / "masks_repair_manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = choose_candidates(data_root, max(1, args.candidate_count))
    rows = []
    for sample_id in candidates:
        image_path = find_image(images_dir, sample_id)
        if image_path is None:
            continue
        image = load_rgb(image_path)
        thin_mask = load_mask(masks_dir / f"{sample_id}_mask.png")
        repair_v1_mask = load_mask(repair_v1_dir / f"{sample_id}_mask.png")
        candidate_dir = output_dir / "candidates" / sample_id
        save_rgb(candidate_dir / f"{sample_id}_image_preview.png", image)
        save_rgb(candidate_dir / f"{sample_id}_thin_mask_overlay.png", overlay(image, thin_mask, (255, 0, 0)))
        save_rgb(candidate_dir / f"{sample_id}_repair_v1_overlay.png", overlay(image, repair_v1_mask, (0, 255, 0)))
        rows.append(
            {
                "id": sample_id,
                "image_path": str(image_path),
                "manual_target_path": str(manual_dir / f"{sample_id}_mask.png"),
                "image_preview": str(candidate_dir / f"{sample_id}_image_preview.png"),
                "thin_mask_overlay": str(candidate_dir / f"{sample_id}_thin_mask_overlay.png"),
                "repair_v1_overlay": str(candidate_dir / f"{sample_id}_repair_v1_overlay.png"),
            }
        )

    guide_path = output_dir / "REPAIR_MANUAL_ANNOTATION_GUIDE.md"
    write_guide(guide_path, data_root)
    with (output_dir / "candidate_index.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["id", "image_path", "manual_target_path", "image_preview", "thin_mask_overlay", "repair_v1_overlay"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "data_root": str(data_root),
        "manual_mask_dir": str(manual_dir),
        "manual_mask_dir_exists": manual_dir.exists(),
        "manual_masks_existing": len(list(manual_dir.glob("*_mask.png"))),
        "candidate_ids": [row["id"] for row in rows],
        "annotation_guide": str(guide_path),
        "r012_training": "skipped because masks_repair_manual is not ready",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manual_mask_dir: {manual_dir}")
    print(f"candidate_ids: {','.join(summary['candidate_ids'])}")
    print(f"annotation_guide: {guide_path}")
    print("r012 training: skipped because masks_repair_manual is not ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
