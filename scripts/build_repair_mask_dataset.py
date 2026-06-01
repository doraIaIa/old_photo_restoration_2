from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.postprocess.mask_refinement import ensure_binary_mask, mask_ratio, refine_mask, summarize_ratios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo repair-mask dataset từ mask thật ngoài repo.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--modes", default="dilate1,dilate2,close_dilate1,repair_v1,repair_v2,repair_v3_conservative")
    parser.add_argument("--make-overlays", action="store_true")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument("--audit-output-dir", default="outputs/report_assets/repair_mask_r011_suite/repair_dataset_audit")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def find_image(images_dir: Path, sample_id: str) -> Path | None:
    for extension in (".jpg", ".png", ".jpeg"):
        candidate = images_dir / f"{sample_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def load_gray(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    return mask


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_gray(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), ensure_binary_mask(mask))


def save_overlay(path: Path, image_rgb: np.ndarray, mask: np.ndarray) -> None:
    binary = ensure_binary_mask(mask)
    if binary.shape != image_rgb.shape[:2]:
        binary = cv2.resize(binary, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = image_rgb.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    alpha = (binary.astype(np.float32) / 255.0)[:, :, None] * 0.45
    overlay = np.clip(overlay * (1.0 - alpha) + red * alpha, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root)
    audit_output_dir = resolve_path(args.audit_output_dir)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    images_dir = data_root / args.image_dir
    masks_dir = data_root / args.mask_dir
    if not images_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy image dir: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy mask dir: {masks_dir}")

    rows: list[dict[str, Any]] = []
    mode_rows = {mode: [] for mode in modes}
    ids = [f"{index:03d}" for index in range(1, 61)]

    for sample_id in ids:
        row: dict[str, Any] = {"id": sample_id, "missing": False, "unreadable": False, "size_mismatch": False}
        image_path = find_image(images_dir, sample_id)
        mask_path = masks_dir / f"{sample_id}_mask.png"
        if image_path is None or not mask_path.exists():
            row["missing"] = True
            rows.append(row)
            continue
        try:
            image = load_rgb(image_path)
            original_mask = ensure_binary_mask(load_gray(mask_path))
        except Exception:
            row["unreadable"] = True
            rows.append(row)
            continue
        row["size_mismatch"] = image.shape[:2] != original_mask.shape[:2]
        original_ratio = mask_ratio(original_mask)
        row["original_mask_ratio"] = f"{original_ratio:.8f}"

        for mode in modes:
            repaired = refine_mask(original_mask, mode)
            target_dir = output_root / f"masks_repair_{mode}"
            target_path = target_dir / f"{sample_id}_mask.png"
            save_gray(target_path, repaired)
            repaired_ratio = mask_ratio(repaired)
            warning = ""
            if repaired_ratio > 0.30:
                warning = "ratio_gt_0.30"
            elif repaired_ratio > 0.20:
                warning = "ratio_gt_0.20"
            row[f"{mode}_ratio"] = f"{repaired_ratio:.8f}"
            row[f"{mode}_delta_ratio"] = f"{(repaired_ratio - original_ratio):.8f}"
            row[f"{mode}_warning"] = warning
            mode_rows[mode].append({"id": sample_id, "ratio": repaired_ratio, "warning": warning})
            if args.make_overlays and image.shape[:2] == repaired.shape[:2]:
                save_overlay(audit_output_dir / "overlays" / mode / f"{sample_id}_overlay.png", image, repaired)
        rows.append(row)

    fieldnames = ["id", "missing", "unreadable", "size_mismatch", "original_mask_ratio"]
    for mode in modes:
        fieldnames.extend([f"{mode}_ratio", f"{mode}_delta_ratio", f"{mode}_warning"])
    write_csv(audit_output_dir / "repair_mask_audit.csv", rows, fieldnames)

    summary = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "modes": modes,
        "num_expected": len(ids),
        "num_rows": len(rows),
        "mode_ratio_summary": {
            mode: {
                **summarize_ratios(mode_rows[mode], "ratio"),
                "num_warning_gt_0.20": sum(1 for item in mode_rows[mode] if item["warning"] == "ratio_gt_0.20"),
                "num_warning_gt_0.30": sum(1 for item in mode_rows[mode] if item["warning"] == "ratio_gt_0.30"),
            }
            for mode in modes
        },
    }
    (audit_output_dir / "repair_mask_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (audit_output_dir / "repair_mask_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"repair_mask_audit: {audit_output_dir / 'repair_mask_audit.csv'}")
    print(f"repair_mask_summary: {audit_output_dir / 'repair_mask_summary.json'}")
    for mode, payload in summary["mode_ratio_summary"].items():
        print(f"{mode}: min={payload['min']:.6f} mean={payload['mean']:.6f} max={payload['max']:.6f} gt030={payload['num_warning_gt_0.30']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
