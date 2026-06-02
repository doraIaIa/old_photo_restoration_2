from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo overlay QC cho manual repair masks.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ids", default="", help="Danh sách ID cách nhau bằng dấu phẩy. Mặc định dùng toàn bộ masks.")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks_repair_manual")
    parser.add_argument("--output-dir", default="manual_qc_overlay")
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_ids(raw: str, masks_dir: Path) -> list[str]:
    if raw.strip():
        return [item.strip().zfill(3) for item in raw.split(",") if item.strip()]
    ids = []
    for path in sorted(masks_dir.glob("*_mask.png")):
        stem = path.stem
        ids.append(stem[:-5] if stem.endswith("_mask") else stem)
    return ids


def find_image(images_dir: Path, sample_id: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{sample_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    overlay = image_bgr.copy()
    positive = mask > 127
    red_layer = np.zeros_like(image_bgr)
    red_layer[:, :, 2] = 255
    overlay[positive] = cv2.addWeighted(image_bgr[positive], 1.0 - alpha, red_layer[positive], alpha, 0)
    return overlay


def process_one(sample_id: str, data_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    images_dir = data_root / args.image_dir
    masks_dir = data_root / args.mask_dir
    output_dir = data_root / args.output_dir
    image_path = find_image(images_dir, sample_id)
    mask_path = masks_dir / f"{sample_id}_mask.png"
    row: dict[str, Any] = {
        "id": sample_id,
        "image_path": str(image_path) if image_path else "",
        "mask_path": str(mask_path),
        "output_path": "",
        "status": "pending",
    }
    if image_path is None:
        row["status"] = "missing_image"
        return row
    if not mask_path.exists():
        row["status"] = "missing_mask"
        return row

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None:
        row["status"] = "unreadable_image"
        return row
    if mask is None:
        row["status"] = "unreadable_mask"
        return row
    image_height, image_width = image_bgr.shape[:2]
    mask_height, mask_width = mask.shape[:2]
    row["image_size"] = {"width": int(image_width), "height": int(image_height)}
    row["mask_size"] = {"width": int(mask_width), "height": int(mask_height)}
    if (image_width, image_height) != (mask_width, mask_height):
        row["status"] = "size_mismatch"
        return row

    overlay = make_overlay(image_bgr, mask, args.alpha)
    output_path = output_dir / f"{sample_id}_overlay.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Không ghi được overlay: {output_path}")
    row["status"] = "created"
    row["output_path"] = str(output_path)
    return row


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha phải nằm trong [0, 1].")
    data_root = resolve_path(args.data_root)
    masks_dir = data_root / args.mask_dir
    if not data_root.exists():
        raise FileNotFoundError(f"Không tìm thấy data-root: {data_root}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy mask-dir: {masks_dir}")
    ids = parse_ids(args.ids, masks_dir)
    rows = [process_one(sample_id, data_root, args) for sample_id in ids]
    summary = {
        "data_root": str(data_root),
        "ids": ids,
        "overlay_created": sum(1 for row in rows if row["status"] == "created"),
        "skipped_size_mismatch": [row["id"] for row in rows if row["status"] == "size_mismatch"],
        "missing_images": [row["id"] for row in rows if row["status"] == "missing_image"],
        "rows": rows,
    }
    summary_path = data_root / args.output_dir / "overlay_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(f"id={row['id']} status={row['status']} output={row.get('output_path', '')}")
    print(f"overlay_created: {summary['overlay_created']}")
    print(f"skipped_size_mismatch: {summary['skipped_size_mismatch']}")
    print(f"missing_images: {summary['missing_images']}")
    print(f"summary_json: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
