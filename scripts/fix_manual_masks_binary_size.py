from __future__ import annotations

import argparse
import json
import shutil
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
    parser = argparse.ArgumentParser(description="Fix manual repair masks về đúng binary 0/255 và đúng size ảnh gốc.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ids", default="", help="Danh sách ID cách nhau bằng dấu phẩy, ví dụ 003,005,007.")
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè masks_repair_manual sau khi backup bản cũ.")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks_repair_manual")
    parser.add_argument("--preview-dir", default="masks_repair_manual_fixed_preview")
    parser.add_argument("--backup-dir", default="masks_repair_manual_backup_before_fix")
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


def read_mask_any(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def fix_one(sample_id: str, data_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    images_dir = data_root / args.image_dir
    masks_dir = data_root / args.mask_dir
    preview_dir = data_root / args.preview_dir
    backup_dir = data_root / args.backup_dir
    image_path = find_image(images_dir, sample_id)
    mask_path = masks_dir / f"{sample_id}_mask.png"
    row: dict[str, Any] = {
        "id": sample_id,
        "image_path": str(image_path) if image_path else "",
        "mask_path": str(mask_path),
        "status": "pending",
        "output_path": "",
        "backup_path": "",
    }
    if image_path is None:
        row["status"] = "missing_image"
        return row
    if not mask_path.exists():
        row["status"] = "missing_mask"
        return row

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        row["status"] = "unreadable_image"
        return row
    mask = read_mask_any(mask_path)
    image_height, image_width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]
    row["original_image_size"] = {"width": int(image_width), "height": int(image_height)}
    row["original_mask_size"] = {"width": int(mask_width), "height": int(mask_height)}

    if (mask_width, mask_height) != (image_width, image_height):
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    fixed = np.where(mask >= args.threshold, 255, 0).astype(np.uint8)
    positive_ratio = float(np.count_nonzero(fixed == 255) / fixed.size)
    output_path = (masks_dir if args.overwrite else preview_dir) / f"{sample_id}_mask.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        backup_path = backup_dir / f"{sample_id}_mask.png"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(mask_path, backup_path)
        row["backup_path"] = str(backup_path)

    if not cv2.imwrite(str(output_path), fixed):
        raise RuntimeError(f"Không ghi được fixed mask: {output_path}")

    row["status"] = "fixed"
    row["fixed_mask_size"] = {"width": int(fixed.shape[1]), "height": int(fixed.shape[0])}
    row["unique_values_after_fix"] = sorted(int(value) for value in np.unique(fixed))
    row["positive_ratio"] = positive_ratio
    row["output_path"] = str(output_path)
    return row


def main() -> int:
    args = parse_args()
    if not 0 <= args.threshold <= 255:
        raise ValueError("--threshold phải nằm trong [0, 255].")
    data_root = resolve_path(args.data_root)
    masks_dir = data_root / args.mask_dir
    if not data_root.exists():
        raise FileNotFoundError(f"Không tìm thấy data-root: {data_root}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy mask-dir: {masks_dir}")

    ids = parse_ids(args.ids, masks_dir)
    rows = [fix_one(sample_id, data_root, args) for sample_id in ids]
    summary = {
        "data_root": str(data_root),
        "ids": ids,
        "threshold": args.threshold,
        "overwrite": bool(args.overwrite),
        "fixed_count": sum(1 for row in rows if row["status"] == "fixed"),
        "rows": rows,
    }
    summary_path = data_root / (args.backup_dir if args.overwrite else args.preview_dir) / "fix_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"id={row['id']} status={row['status']} "
            f"image_size={row.get('original_image_size')} mask_size={row.get('original_mask_size')} "
            f"fixed_size={row.get('fixed_mask_size')} unique={row.get('unique_values_after_fix')} "
            f"positive_ratio={row.get('positive_ratio')} output={row.get('output_path')}"
        )
    print(f"fixed_count: {summary['fixed_count']}")
    print(f"summary_json: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
