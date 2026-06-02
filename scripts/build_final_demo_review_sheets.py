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
BENCHMARK_ROOT = PROJECT_ROOT / "outputs" / "final_demo_benchmark"
REVIEW_DIR = BENCHMARK_ROOT / "review_sheets"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo contact sheets cho final demo benchmark.")
    parser.add_argument("--benchmark-root", default=str(BENCHMARK_ROOT))
    return parser.parse_args()


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


def fit_tile(image_rgb: np.ndarray | None, label: str, size: tuple[int, int] = (360, 260)) -> np.ndarray:
    tile_w, tile_h = size
    canvas = np.full((tile_h, tile_w, 3), 244, dtype=np.uint8)
    if image_rgb is not None:
        height, width = image_rgb.shape[:2]
        scale = min(tile_w / max(width, 1), (tile_h - 34) / max(height, 1))
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x0 = (tile_w - new_w) // 2
        y0 = 34 + (tile_h - 34 - new_h) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    cv2.rectangle(canvas, (0, 0), (tile_w, 32), (25, 25, 25), thickness=-1)
    cv2.putText(canvas, label[:42], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def metadata_to_label(metadata: dict[str, Any]) -> str:
    mode = metadata.get("pipeline_mode", metadata.get("mask_mode", "unknown"))
    backend = metadata.get("inpainting_backend_actual", metadata.get("actual_backend", ""))
    face_backend = metadata.get("face_backend", metadata.get("face_restoration_backend", ""))
    fidelity = metadata.get("codeformer_fidelity", "")
    mask_ratio = metadata.get("final_mask_ratio", "")
    return f"{mode} | {backend} | face={face_backend} | fid={fidelity} | mask={mask_ratio}"


def build_sheet(case_dir: Path, output_path: Path) -> dict[str, Any]:
    metadata_path = case_dir / "metadata.json"
    metadata = load_json(metadata_path)
    original = read_rgb(case_dir / "input.png")
    mask = read_rgb(case_dir / "final_mask.png")
    overlay = read_rgb(case_dir / "overlay_final.png")
    before_face = read_rgb(case_dir / "restored_before_face.png")
    restored_final = read_rgb(case_dir / "restored_final.png")
    label_text = metadata_to_label(metadata)
    label_tile = np.full((72, 360 * 5, 3), 252, dtype=np.uint8)
    cv2.putText(label_tile, label_text[:150], (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    tiles = [
        fit_tile(original, "original"),
        fit_tile(mask, "mask"),
        fit_tile(overlay, "overlay"),
        fit_tile(before_face, "before face"),
        fit_tile(restored_final, "restored final"),
    ]
    sheet = np.concatenate([label_tile, np.concatenate(tiles, axis=1)], axis=0)
    write_rgb(output_path, sheet)
    return {
        "case_dir": str(case_dir),
        "review_sheet": str(output_path),
        "metadata_path": str(metadata_path),
        "mode": metadata.get("pipeline_mode", metadata.get("mask_mode", "")),
        "backend": metadata.get("inpainting_backend_actual", metadata.get("actual_backend", "")),
        "face_backend": metadata.get("face_backend", metadata.get("face_restoration_backend", "")),
        "face_restoration_applied": metadata.get("face_restoration_applied", ""),
        "codeformer_fidelity": metadata.get("codeformer_fidelity", ""),
        "final_mask_ratio": metadata.get("final_mask_ratio", ""),
    }


def find_case_dirs(benchmark_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for metadata_path in benchmark_root.glob("cases/**/metadata.json"):
        if "_run_demo_raw" in metadata_path.parts:
            continue
        candidates.append(metadata_path.parent)
    for metadata_path in (benchmark_root / "mask_refinement_sweep").glob("**/metadata.json"):
        if "_run_demo_raw" in metadata_path.parts:
            continue
        candidates.append(metadata_path.parent)
    return sorted(set(candidates))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_review_note(benchmark_root: Path, rows: list[dict[str, Any]]) -> None:
    diagnosis_path = benchmark_root / "BOTTLENECK_DIAGNOSIS.md"
    existing = diagnosis_path.read_text(encoding="utf-8") if diagnosis_path.exists() else "# Final Demo Bottleneck Diagnosis\n"
    if "## Review Sheets" in existing:
        return
    lines = [
        existing.rstrip(),
        "",
        "## Review Sheets",
        "",
        f"- Review sheet folder: `{benchmark_root / 'review_sheets'}`",
        f"- Review sheets generated: `{len(rows)}`",
        "- Human visual review is required before choosing final demo config.",
        "",
    ]
    diagnosis_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    review_dir = benchmark_root / "review_sheets"
    review_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case_dir in find_case_dirs(benchmark_root):
        relative = case_dir.relative_to(benchmark_root)
        safe_name = "__".join(relative.parts)
        output_path = review_dir / f"{safe_name}.png"
        rows.append(build_sheet(case_dir, output_path))
    write_csv(benchmark_root / "review_sheet_index.csv", rows)
    append_review_note(benchmark_root, rows)
    print(f"review_sheets: {review_dir}")
    print(f"review_sheet_index: {benchmark_root / 'review_sheet_index.csv'}")
    print(f"count: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
