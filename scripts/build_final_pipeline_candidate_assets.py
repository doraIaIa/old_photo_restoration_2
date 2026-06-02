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
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_pipeline_candidate"
DEMO3_ROOT = OUTPUT_ROOT / "demo3_sensitive_mode"
CASE_RUNS = DEMO3_ROOT / "case_runs"
MANUAL_MASK_DIR = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3" / "manual_masks"

CASES = [
    ("baseline_r011_union_simple_lama_off", CASE_RUNS / "baseline_simple_lama" / "demo3" / "auto_r011_union_refined"),
    ("sensitive_r011_simple_lama_off", CASE_RUNS / "sensitive_simple_lama_off" / "demo3" / "auto_r011_sensitive_low_threshold"),
    ("sensitive_r011_simple_lama_codeformer", CASE_RUNS / "sensitive_simple_lama_codeformer" / "demo3" / "auto_r011_sensitive_low_threshold"),
    ("sensitive_r011_opencv_off", CASE_RUNS / "sensitive_opencv_off" / "demo3" / "auto_r011_sensitive_low_threshold"),
]

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


def fit_tile(image_rgb: np.ndarray, label: str, size: tuple[int, int] = (430, 310)) -> np.ndarray:
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
    cv2.rectangle(canvas, (0, 0), (tile_w, 32), (22, 22, 22), thickness=-1)
    cv2.putText(canvas, label[:42], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def copy_required_case_files(label: str, case_dir: Path) -> dict[str, Any]:
    target = DEMO3_ROOT / "named_cases" / label
    target.mkdir(parents=True, exist_ok=True)
    required = ["input.png", "final_mask.png", "overlay_final.png", "restored_before_face.png", "restored_final.png", "comparison_grid.png", "metadata.json"]
    copied = {}
    for filename in required:
        source = case_dir / filename
        copied[filename] = source.exists()
        if source.exists():
            shutil.copy2(source, target / filename)
    metadata = load_json(target / "metadata.json")
    return {
        "case_label": label,
        "source_dir": str(case_dir),
        "target_dir": str(target),
        "all_required_present": all(copied.values()),
        "mask_mode": metadata.get("mask_mode", metadata.get("pipeline_mode", "")),
        "backend": metadata.get("inpainting_backend_actual", ""),
        "face_mode": metadata.get("face_mode", ""),
        "face_restoration_applied": metadata.get("face_restoration_applied", ""),
        "face_backend": metadata.get("face_backend", ""),
        "final_mask_ratio": metadata.get("final_mask_ratio", ""),
        "experimental": metadata.get("experimental", ""),
        "warning": metadata.get("warning", ""),
        **{f"{name}_exists": copied[name] for name in required},
    }


def build_grid(rows: list[dict[str, Any]]) -> Path:
    by_label = {row["case_label"]: Path(row["target_dir"]) for row in rows}
    baseline = by_label["baseline_r011_union_simple_lama_off"]
    sensitive = by_label["sensitive_r011_simple_lama_off"]
    codeformer = by_label["sensitive_r011_simple_lama_codeformer"]
    tiles = [
        fit_tile(read_rgb(baseline / "input.png"), "original"),
        fit_tile(read_rgb(baseline / "overlay_final.png"), "baseline overlay"),
        fit_tile(read_rgb(sensitive / "overlay_final.png"), "sensitive overlay"),
        fit_tile(read_rgb(baseline / "restored_before_face.png"), "baseline restored"),
        fit_tile(read_rgb(sensitive / "restored_before_face.png"), "sensitive restored"),
        fit_tile(read_rgb(codeformer / "restored_final.png"), "sensitive + CodeFormer"),
    ]
    grid = np.concatenate([np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)], axis=0)
    output_path = OUTPUT_ROOT / "demo3_final_sensitive_comparison.png"
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
    with (OUTPUT_ROOT / "candidate_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_oracle_instructions() -> None:
    instruction_path = OUTPUT_ROOT / "ORACLE_MASK_INSTRUCTIONS.md"
    if MANUAL_MASK_DIR.exists() and (MANUAL_MASK_DIR / "demo3_mask.png").exists():
        return
    lines = [
        "# Oracle Mask Instructions",
        "",
        "Chưa tìm thấy `data\\demo_inputs\\real_manual_3\\manual_masks\\demo3_mask.png`.",
        "",
        "Để chạy oracle/manual-mask diagnosis, tạo file:",
        "",
        "`data\\demo_inputs\\real_manual_3\\manual_masks\\demo3_mask.png`",
        "",
        "Yêu cầu mask:",
        "",
        "- Cùng size với ảnh gốc `demo3`.",
        "- Binary 0/255.",
        "- Nền giấy không hư: `0`.",
        "- Vết nứt/xước/gấp/halo/vùng giấy hư cần xóa: `255`.",
        "- Đánh rõ các vết nứt dài/mỏng bị model bỏ sót.",
        "- Không đánh nhầm mép mặt, súng, áo, tóc hoặc chi tiết thật cần giữ.",
        "",
        "Sau khi có mask, rerun final pipeline candidate để so sánh:",
        "",
        "- external/manual mask + simple_lama + face off",
        "- external/manual mask + simple_lama + codeformer_if_available",
        "",
    ]
    instruction_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(rows: list[dict[str, Any]], grid_path: Path) -> None:
    baseline = next(row for row in rows if row["case_label"] == "baseline_r011_union_simple_lama_off")
    sensitive = next(row for row in rows if row["case_label"] == "sensitive_r011_simple_lama_off")
    codeformer = next(row for row in rows if row["case_label"] == "sensitive_r011_simple_lama_codeformer")
    oracle_available = MANUAL_MASK_DIR.exists() and (MANUAL_MASK_DIR / "demo3_mask.png").exists()
    lines = [
        "# Final Pipeline Candidate Summary",
        "",
        f"- Comparison grid: `{grid_path}`",
        f"- Candidate index: `{OUTPUT_ROOT / 'candidate_index.csv'}`",
        f"- Oracle demo3 mask available: `{oracle_available}`",
        "",
        "## Candidate Cases",
        "",
        f"- Baseline: `{baseline['mask_mode']}` + `{baseline['backend']}` + face off, mask_ratio `{baseline['final_mask_ratio']}`",
        f"- Sensitive: `{sensitive['mask_mode']}` + `{sensitive['backend']}` + face off, mask_ratio `{sensitive['final_mask_ratio']}`",
        f"- Sensitive + CodeFormer: applied `{codeformer['face_restoration_applied']}`, backend `{codeformer['face_backend']}`",
        "",
        "## Recommendation Status",
        "",
        "`auto_r011_sensitive_low_threshold` is an experimental demo3 candidate only.",
        "It should not become the project-wide default until broader visual review confirms acceptable false positives.",
        "",
        "## Warning",
        "",
        "Recall-sensitive thresholding may increase false positives on other old photos.",
        "Do not claim this mode is generally best.",
        "",
    ]
    (OUTPUT_ROOT / "FINAL_CANDIDATE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [copy_required_case_files(label, case_dir) for label, case_dir in CASES]
    grid_path = build_grid(rows)
    write_index(rows)
    write_oracle_instructions()
    write_summary(rows, grid_path)
    print(f"summary: {OUTPUT_ROOT / 'FINAL_CANDIDATE_SUMMARY.md'}")
    print(f"grid: {grid_path}")
    print(f"index: {OUTPUT_ROOT / 'candidate_index.csv'}")
    print(f"oracle_instructions: {OUTPUT_ROOT / 'ORACLE_MASK_INSTRUCTIONS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
