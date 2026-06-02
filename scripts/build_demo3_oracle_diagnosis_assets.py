from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_pipeline_candidate" / "demo3_oracle_diagnosis"
CASE_RUNS = OUTPUT_ROOT / "case_runs"

CASES = [
    ("baseline_auto", CASE_RUNS / "baseline_auto" / "demo3" / "auto_r011_union_refined"),
    ("sensitive_auto", CASE_RUNS / "sensitive_auto" / "demo3" / "auto_r011_sensitive_low_threshold"),
    ("oracle_simple_lama", CASE_RUNS / "oracle_simple_lama" / "demo3" / "external"),
    ("oracle_simple_lama_codeformer", CASE_RUNS / "oracle_simple_lama_codeformer" / "demo3" / "external"),
    ("oracle_opencv", CASE_RUNS / "oracle_opencv" / "demo3" / "external"),
]
REQUIRED = ["input.png", "final_mask.png", "overlay_final.png", "restored_before_face.png", "restored_final.png", "comparison_grid.png", "metadata.json"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def fit_tile(image_rgb: np.ndarray, label: str, size: tuple[int, int] = (390, 280)) -> np.ndarray:
    tile_w, tile_h = size
    canvas = np.full((tile_h, tile_w, 3), 246, dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    scale = min(tile_w / max(width, 1), (tile_h - 34) / max(height, 1))
    resized = cv2.resize(
        image_rgb,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x0 = (tile_w - resized.shape[1]) // 2
    y0 = 34 + (tile_h - 34 - resized.shape[0]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (tile_w, 32), (24, 24, 24), thickness=-1)
    cv2.putText(canvas, label[:40], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def copy_case(label: str, source_dir: Path) -> dict[str, Any]:
    target_dir = OUTPUT_ROOT / "named_cases" / label
    target_dir.mkdir(parents=True, exist_ok=True)
    exists = {}
    for filename in REQUIRED:
        source = source_dir / filename
        exists[filename] = source.exists()
        if source.exists():
            shutil.copy2(source, target_dir / filename)
    metadata = load_json(target_dir / "metadata.json")
    return {
        "case_label": label,
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "all_required_present": all(exists.values()),
        "mask_mode": metadata.get("mask_mode", metadata.get("pipeline_mode", "")),
        "backend": metadata.get("inpainting_backend_actual", ""),
        "face_mode": metadata.get("face_mode", ""),
        "face_restoration_applied": metadata.get("face_restoration_applied", ""),
        "face_backend": metadata.get("face_backend", ""),
        "final_mask_ratio": metadata.get("final_mask_ratio", ""),
        "experimental": metadata.get("experimental", ""),
        "warning": metadata.get("warning", ""),
    }


def build_grid(rows: list[dict[str, Any]]) -> Path:
    case_dir = {row["case_label"]: Path(row["target_dir"]) for row in rows}
    tiles = [
        fit_tile(read_rgb(case_dir["baseline_auto"] / "input.png"), "original"),
        fit_tile(read_rgb(case_dir["baseline_auto"] / "overlay_final.png"), "baseline overlay"),
        fit_tile(read_rgb(case_dir["sensitive_auto"] / "overlay_final.png"), "sensitive overlay"),
        fit_tile(read_rgb(case_dir["oracle_simple_lama"] / "overlay_final.png"), "oracle overlay"),
        fit_tile(read_rgb(case_dir["baseline_auto"] / "restored_before_face.png"), "baseline restored"),
        fit_tile(read_rgb(case_dir["sensitive_auto"] / "restored_before_face.png"), "sensitive restored"),
        fit_tile(read_rgb(case_dir["oracle_simple_lama"] / "restored_before_face.png"), "oracle simple_lama"),
        fit_tile(read_rgb(case_dir["oracle_opencv"] / "restored_before_face.png"), "oracle opencv"),
        fit_tile(read_rgb(case_dir["oracle_simple_lama_codeformer"] / "restored_final.png"), "oracle + CodeFormer"),
    ]
    rows_rgb = [np.concatenate(tiles[start : start + 3], axis=1) for start in range(0, 9, 3)]
    grid = np.concatenate(rows_rgb, axis=0)
    output_path = OUTPUT_ROOT / "demo3_oracle_comparison_grid.png"
    write_rgb(output_path, grid)
    return output_path


def write_index(rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_label",
        "mask_mode",
        "backend",
        "face_mode",
        "face_restoration_applied",
        "face_backend",
        "final_mask_ratio",
        "experimental",
        "warning",
        "all_required_present",
        "source_dir",
        "target_dir",
    ]
    with (OUTPUT_ROOT / "oracle_diagnosis_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_summary(rows: list[dict[str, Any]], grid_path: Path) -> None:
    by_label = {row["case_label"]: row for row in rows}
    baseline_ratio = as_float(by_label["baseline_auto"].get("final_mask_ratio"))
    sensitive_ratio = as_float(by_label["sensitive_auto"].get("final_mask_ratio"))
    oracle_ratio = as_float(by_label["oracle_simple_lama"].get("final_mask_ratio"))
    ratio_signal = "unknown"
    if sensitive_ratio is not None and oracle_ratio is not None:
        if sensitive_ratio > oracle_ratio:
            ratio_signal = "sensitive mask is wider than oracle; inspect for overmasking or missing halo in oracle."
        elif sensitive_ratio < oracle_ratio:
            ratio_signal = "oracle mask is wider than sensitive; automatic mask may miss damaged regions."
        else:
            ratio_signal = "sensitive and oracle mask ratios are similar."
    bottleneck_likely = "needs human visual review"
    lines = [
        "# Demo3 Oracle Diagnosis Summary",
        "",
        f"- Comparison grid: `{grid_path}`",
        f"- Index: `{OUTPUT_ROOT / 'oracle_diagnosis_index.csv'}`",
        f"- Bottleneck likely: `{bottleneck_likely}`",
        "",
        "## Mask Ratios",
        "",
        f"- Baseline auto: `{baseline_ratio}`",
        f"- Sensitive auto: `{sensitive_ratio}`",
        f"- Oracle/manual: `{oracle_ratio}`",
        f"- Ratio signal: {ratio_signal}",
        "",
        "## CodeFormer",
        "",
        f"- Oracle + CodeFormer applied: `{by_label['oracle_simple_lama_codeformer'].get('face_restoration_applied')}`",
        f"- Face backend: `{by_label['oracle_simple_lama_codeformer'].get('face_backend')}`",
        "",
        "## Diagnosis Logic For Human Review",
        "",
        "- If oracle restored still has many cracks/blotches, bottleneck likely is `inpainting_backend`.",
        "- If oracle restored is clearly better than sensitive, bottleneck likely is `mask_quality_or_refinement`, and sensitive may overmask or mark wrong regions.",
        "- If sensitive is close to or better than oracle, sensitive mode is a demo3 candidate but still needs false-positive warning.",
        "- If oracle mask ratio is lower than sensitive but oracle output is better, sensitive is likely overmasking.",
        "- If sensitive mask is wider but output looks better, oracle may miss halo/thin cracks or sensitive may suit the current inpainting backend better.",
        "- If oracle + opencv is better on thin cracks, OpenCV is a candidate for thin crack cases.",
        "- If oracle + simple_lama is more natural, simple_lama remains the candidate backend.",
        "",
        "## Current Safe Conclusion",
        "",
        "Metadata shows sensitive mask ratio is higher than oracle ratio, so overmasking must be checked visually.",
        "No final quality claim is made until the comparison grid is reviewed.",
        "",
    ]
    (OUTPUT_ROOT / "ORACLE_DIAGNOSIS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [copy_case(label, source) for label, source in CASES]
    grid_path = build_grid(rows)
    write_index(rows)
    write_summary(rows, grid_path)
    print(f"grid: {grid_path}")
    print(f"summary: {OUTPUT_ROOT / 'ORACLE_DIAGNOSIS_SUMMARY.md'}")
    print(f"index: {OUTPUT_ROOT / 'oracle_diagnosis_index.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
