from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "codeformer_pipeline_test"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "codeformer_activation"
GRID_DIR = OUTPUT_ROOT / "face_comparison_grids"
INDEX_CSV = OUTPUT_ROOT / "face_comparison_index.csv"
SUMMARY_MD = OUTPUT_ROOT / "FACE_COMPARISON_SUMMARY.md"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_rgb(path: Path) -> np.ndarray | None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def fit_to_tile(image_rgb: np.ndarray, size: tuple[int, int] = (420, 320)) -> np.ndarray:
    tile_w, tile_h = size
    canvas = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    scale = min(tile_w / max(width, 1), tile_h / max(height, 1))
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (tile_w - new_w) // 2
    y0 = (tile_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def add_label(tile: np.ndarray, label: str) -> np.ndarray:
    labeled = tile.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 34), (20, 20, 20), thickness=-1)
    cv2.putText(labeled, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def diff_image(before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    if before_rgb.shape[:2] != after_rgb.shape[:2]:
        after_rgb = cv2.resize(after_rgb, (before_rgb.shape[1], before_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = cv2.absdiff(before_rgb, after_rgb)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    heat = cv2.applyColorMap(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def detect_face_box(image_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return None
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    pad = int(max(w, h) * 0.35)
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    x1 = min(image_rgb.shape[1], int(x + w) + pad)
    y1 = min(image_rgb.shape[0], int(y + h) + pad)
    return x0, y0, x1, y1


def crop_box(image_rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return image_rgb[y0:y1, x0:x1]


def build_grid(original: np.ndarray, before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, bool]:
    tiles = [
        add_label(fit_to_tile(original), "original"),
        add_label(fit_to_tile(before), "before CodeFormer"),
        add_label(fit_to_tile(after), "after CodeFormer"),
        add_label(fit_to_tile(diff_image(before, after)), "difference heatmap"),
    ]
    face_box = detect_face_box(after)
    if face_box is not None:
        tiles.extend(
            [
                add_label(fit_to_tile(crop_box(before, face_box)), "face crop before"),
                add_label(fit_to_tile(crop_box(after, face_box)), "face crop after"),
            ]
        )
    first_row = np.concatenate(tiles[:3], axis=1)
    if len(tiles) >= 6:
        second_row = np.concatenate(tiles[3:6], axis=1)
        return np.concatenate([first_row, second_row], axis=0), True
    fallback = np.concatenate([tiles[0], tiles[1], tiles[2]], axis=1)
    return fallback, False


def find_case_dirs() -> list[Path]:
    if not INPUT_ROOT.exists():
        return []
    return sorted(path.parent for path in INPUT_ROOT.glob("demo*/auto_r011_union_refined/metadata.json"))


def main() -> int:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case_dir in find_case_dirs():
        demo_id = case_dir.parent.name
        metadata_path = case_dir / "metadata.json"
        metadata = load_json(metadata_path)
        original_path = case_dir / "input.png"
        if not original_path.exists():
            original_path = Path(str(metadata.get("image_path", "")))
        before_path = case_dir / "restored_before_face.png"
        after_path = case_dir / "restored_final.png"
        comparison_path = case_dir / "comparison_grid.png"

        original = read_rgb(original_path)
        before = read_rgb(before_path)
        after = read_rgb(after_path)
        ok = original is not None and before is not None and after is not None
        output_path = GRID_DIR / f"{demo_id}_face_comparison.png"
        face_crop_detected = False
        if ok:
            grid, face_crop_detected = build_grid(original, before, after)
            write_rgb(output_path, grid)

        rows.append(
            {
                "demo_id": demo_id,
                "ok": ok,
                "face_restoration_applied": metadata.get("face_restoration_applied"),
                "face_backend": metadata.get("face_backend"),
                "inpainting_backend_actual": metadata.get("inpainting_backend_actual"),
                "codeformer_fidelity": metadata.get("codeformer_fidelity"),
                "restored_before_face_exists": before_path.exists(),
                "restored_final_exists": after_path.exists(),
                "comparison_grid_exists": comparison_path.exists(),
                "face_crop_detected": face_crop_detected,
                "metadata_path": str(metadata_path),
                "output_grid": str(output_path) if ok else "",
            }
        )

    fields = [
        "demo_id",
        "ok",
        "face_restoration_applied",
        "face_backend",
        "inpainting_backend_actual",
        "codeformer_fidelity",
        "restored_before_face_exists",
        "restored_final_exists",
        "comparison_grid_exists",
        "face_crop_detected",
        "metadata_path",
        "output_grid",
    ]
    with INDEX_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for row in rows if row["ok"])
    lines = [
        "# CodeFormer Face Comparison Summary",
        "",
        f"- Input root: `{INPUT_ROOT}`",
        f"- Output grids: `{GRID_DIR}`",
        f"- Cases found: `{len(rows)}`",
        f"- Grids written: `{passed}`",
        "- Quality note: needs human visual review; metadata only proves CodeFormer ran.",
        "",
        "| demo_id | ok | face_backend | fidelity | face_crop_detected | grid |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['demo_id']} | {row['ok']} | {row['face_backend']} | {row['codeformer_fidelity']} | "
            f"{row['face_crop_detected']} | `{row['output_grid']}` |"
        )
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"grids: {GRID_DIR}")
    print(f"index: {INDEX_CSV}")
    print(f"summary: {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
