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
IMAGE_DIR = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3"
SOURCE_MASK = PROJECT_ROOT / "data" / "demo_masks" / "real_manual_3" / "demo3_mask.png"
TARGET_MASK = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3" / "manual_masks" / "demo3_mask.png"
BACKUP_DIR = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3" / "manual_masks_backup_before_fix"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_pipeline_candidate" / "oracle_mask_fix_preview"
AUDIT_JSON = PROJECT_ROOT / "outputs" / "final_pipeline_candidate" / "oracle_demo3_audit_after_fix.json"
AUDIT_MD = PROJECT_ROOT / "outputs" / "final_pipeline_candidate" / "oracle_demo3_audit_after_fix.md"

VARIANT_OFFSETS = {
    "pad_bottom_right": (0, 0),
    "pad_top_left": (1, 1),
    "pad_right_top": (0, 1),
    "pad_left_bottom": (1, 0),
}
VALID_APPLY = set(VARIANT_OFFSETS) | {"nearest_resize"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sửa size oracle mask demo3 bằng preview an toàn.")
    parser.add_argument("--apply", choices=sorted(VALID_APPLY), default="")
    return parser.parse_args()


def find_demo3_image() -> Path:
    for extension in [".png", ".jpg", ".jpeg"]:
        path = IMAGE_DIR / f"demo3{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Không tìm thấy demo3 trong {IMAGE_DIR}")


def read_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return image


def read_mask_raw(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Không đọc được mask: {path}")
    return mask


def mask_to_binary(mask_raw: np.ndarray) -> tuple[np.ndarray, bool]:
    has_alpha = bool(mask_raw.ndim == 3 and mask_raw.shape[2] == 4)
    if mask_raw.ndim == 3:
        if mask_raw.shape[2] == 4:
            bgr = mask_raw[:, :, :3]
            alpha = mask_raw[:, :, 3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = np.where(alpha > 0, gray, 0).astype(np.uint8)
        else:
            gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
    elif mask_raw.ndim == 2:
        gray = mask_raw
    else:
        raise ValueError(f"Mask shape không hợp lệ: {mask_raw.shape}")
    binary = (gray > 127).astype(np.uint8) * 255
    return binary.astype(np.uint8), has_alpha


def make_canvas_variant(binary_mask: np.ndarray, target_hw: tuple[int, int], offset_xy: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    offset_x, offset_y = offset_xy
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    src_h, src_w = binary_mask.shape[:2]
    paste_w = min(src_w, target_w - offset_x)
    paste_h = min(src_h, target_h - offset_y)
    if paste_w <= 0 or paste_h <= 0:
        raise ValueError(f"Offset không hợp lệ: {offset_xy}")
    canvas[offset_y : offset_y + paste_h, offset_x : offset_x + paste_w] = binary_mask[:paste_h, :paste_w]
    return canvas


def make_nearest_resize(binary_mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    return cv2.resize(binary_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    red = np.zeros_like(image_rgb)
    red[:, :, 0] = 255
    weight = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
    overlay = image_rgb.astype(np.float32) * (1.0 - weight) + red.astype(np.float32) * weight
    return np.clip(overlay, 0, 255).astype(np.uint8)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def write_gray(path: Path, image_gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_gray.astype(np.uint8)):
        raise RuntimeError(f"Không ghi được mask: {path}")


def metadata_for_mask(name: str, mask: np.ndarray, image_shape: tuple[int, int], correction_type: str, offset_xy: tuple[int, int] | None) -> dict[str, Any]:
    unique = [int(value) for value in np.unique(mask)]
    return {
        "variant": name,
        "correction_type": correction_type,
        "offset_xy": list(offset_xy) if offset_xy is not None else None,
        "image_size_hw": [int(image_shape[0]), int(image_shape[1])],
        "fixed_size_hw": [int(mask.shape[0]), int(mask.shape[1])],
        "fixed_size_wh": [int(mask.shape[1]), int(mask.shape[0])],
        "unique_values": unique,
        "unique_count": len(unique),
        "binary": set(unique).issubset({0, 255}),
        "positive_ratio": float((mask > 0).mean()),
        "empty_mask": bool((mask > 0).sum() == 0),
    }


def fit_tile(image_rgb: np.ndarray, label: str, size: tuple[int, int] = (360, 270)) -> np.ndarray:
    tile_w, tile_h = size
    canvas = np.full((tile_h, tile_w, 3), 246, dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    scale = min(tile_w / max(width, 1), (tile_h - 34) / max(height, 1))
    resized = cv2.resize(image_rgb, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    x0 = (tile_w - resized.shape[1]) // 2
    y0 = 34 + (tile_h - 34 - resized.shape[0]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (tile_w, 32), (25, 25, 25), thickness=-1)
    cv2.putText(canvas, label[:40], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def build_contact_sheet(overlays: dict[str, np.ndarray]) -> Path:
    tiles = [fit_tile(image, name) for name, image in overlays.items()]
    first_row = np.concatenate(tiles[:3], axis=1)
    second_row = np.concatenate(tiles[3:], axis=1)
    if second_row.shape[1] < first_row.shape[1]:
        pad = np.full((second_row.shape[0], first_row.shape[1] - second_row.shape[1], 3), 246, dtype=np.uint8)
        second_row = np.concatenate([second_row, pad], axis=1)
    output_path = OUTPUT_DIR / "demo3_oracle_mask_fix_contact.png"
    write_rgb(output_path, np.concatenate([first_row, second_row], axis=0))
    return output_path


def audit_target() -> dict[str, Any]:
    image_path = find_demo3_image()
    image = read_image_bgr(image_path)
    result: dict[str, Any] = {
        "target_exists": TARGET_MASK.exists(),
        "target_mask_path": str(TARGET_MASK),
        "image_path": str(image_path),
        "image_size_hw": [int(image.shape[0]), int(image.shape[1])],
        "image_size_wh": [int(image.shape[1]), int(image.shape[0])],
    }
    if not TARGET_MASK.exists():
        result.update({"pass": False, "error": "target_missing"})
        return result
    raw = read_mask_raw(TARGET_MASK)
    binary, has_alpha = mask_to_binary(raw)
    unique = [int(value) for value in np.unique(binary)]
    same_size = binary.shape[:2] == image.shape[:2]
    binary_ok = set(unique).issubset({0, 255})
    positive_ratio = float((binary > 0).mean())
    result.update(
        {
            "raw_shape": list(raw.shape),
            "has_alpha": has_alpha,
            "mask_size_hw": [int(binary.shape[0]), int(binary.shape[1])],
            "mask_size_wh": [int(binary.shape[1]), int(binary.shape[0])],
            "same_size": same_size,
            "unique_values": unique,
            "binary": binary_ok,
            "positive_ratio": positive_ratio,
            "likely_empty": positive_ratio <= 0.0,
            "pass": bool(same_size and binary_ok and positive_ratio > 0.0),
            "error": "" if same_size and binary_ok and positive_ratio > 0.0 else "audit_failed",
        }
    )
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Oracle Demo3 Audit After Fix", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in result.items())
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def apply_variant(variant_name: str, variant_path: Path) -> dict[str, Any]:
    TARGET_MASK.parent.mkdir(parents=True, exist_ok=True)
    if TARGET_MASK.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP_DIR / f"demo3_mask_backup_{len(list(BACKUP_DIR.glob('demo3_mask_backup_*.png'))) + 1:03d}.png"
        shutil.copy2(TARGET_MASK, backup_path)
    shutil.copy2(variant_path, TARGET_MASK)
    audit = audit_target()
    audit["applied_variant"] = variant_name
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Oracle Demo3 Audit After Fix", ""]
    lines.extend(f"- {key}: `{value}`" for key, value in audit.items())
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    args = parse_args()
    image_path = find_demo3_image()
    image = read_image_bgr(image_path)
    if not SOURCE_MASK.exists():
        raise FileNotFoundError(f"Không tìm thấy source mask: {SOURCE_MASK}")
    raw_mask = read_mask_raw(SOURCE_MASK)
    binary_mask, has_alpha = mask_to_binary(raw_mask)
    image_hw = image.shape[:2]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    variants: dict[str, np.ndarray] = {}
    for name, offset_xy in VARIANT_OFFSETS.items():
        variants[name] = make_canvas_variant(binary_mask, image_hw, offset_xy)
    variants["nearest_resize"] = make_nearest_resize(binary_mask, image_hw)

    overlays: dict[str, np.ndarray] = {}
    variant_paths: dict[str, Path] = {}
    metadata_rows: list[dict[str, Any]] = []
    for name, fixed_mask in variants.items():
        mask_path = OUTPUT_DIR / f"{name}_demo3_mask.png"
        overlay_path = OUTPUT_DIR / f"{name}_overlay.png"
        metadata_path = OUTPUT_DIR / f"{name}_metadata.json"
        write_gray(mask_path, fixed_mask)
        overlay = make_overlay(image, fixed_mask)
        write_rgb(overlay_path, overlay)
        correction_type = "nearest_resize" if name == "nearest_resize" else "canvas_pad"
        metadata = metadata_for_mask(name, fixed_mask, image_hw, correction_type, VARIANT_OFFSETS.get(name))
        metadata.update(
            {
                "image_path": str(image_path),
                "source_mask": str(SOURCE_MASK),
                "source_raw_shape": list(raw_mask.shape),
                "source_size_hw": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
                "source_size_wh": [int(binary_mask.shape[1]), int(binary_mask.shape[0])],
                "source_has_alpha": has_alpha,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        overlays[name] = overlay
        variant_paths[name] = mask_path
        metadata_rows.append(metadata)

    contact_path = build_contact_sheet(overlays)
    summary_lines = [
        "# Oracle Mask Fix Preview",
        "",
        f"- Image path: `{image_path}`",
        f"- Image size W x H: `{image.shape[1]} x {image.shape[0]}`",
        f"- Image size H x W: `{image.shape[0]} x {image.shape[1]}`",
        f"- Source mask: `{SOURCE_MASK}`",
        f"- Source mask raw shape: `{list(raw_mask.shape)}`",
        f"- Source mask size W x H: `{binary_mask.shape[1]} x {binary_mask.shape[0]}`",
        f"- Source mask size H x W: `{binary_mask.shape[0]} x {binary_mask.shape[1]}`",
        f"- Source has alpha: `{has_alpha}`",
        f"- Contact sheet: `{contact_path}`",
        "- Suggested default candidate if alignment is uncertain: `pad_bottom_right`.",
        "- Not applied unless `--apply` is provided.",
        "",
        "## Variants",
        "",
    ]
    for metadata in metadata_rows:
        summary_lines.append(
            f"- `{metadata['variant']}`: size_hw={metadata['fixed_size_hw']}, "
            f"positive_ratio={metadata['positive_ratio']}, offset={metadata['offset_xy']}"
        )

    applied = None
    if args.apply:
        applied = apply_variant(args.apply, variant_paths[args.apply])
        summary_lines.extend(
            [
                "",
                "## Apply Result",
                "",
                f"- Applied variant: `{args.apply}`",
                f"- Target mask: `{TARGET_MASK}`",
                f"- Audit pass: `{applied.get('pass')}`",
                f"- Audit JSON: `{AUDIT_JSON}`",
                f"- Audit MD: `{AUDIT_MD}`",
            ]
        )

    (OUTPUT_DIR / "ORACLE_MASK_FIX_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"preview_folder: {OUTPUT_DIR}")
    print(f"contact_sheet: {contact_path}")
    print("suggested_variant: pad_bottom_right")
    print(f"applied: {args.apply or 'none'}")
    if applied is not None:
        print(f"audit_pass: {applied.get('pass')}")
        print(f"target_mask: {TARGET_MASK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
