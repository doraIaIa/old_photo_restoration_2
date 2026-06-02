from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3"
DEFAULT_ORACLE_MASK = DEFAULT_IMAGE_DIR / "manual_masks" / "demo3_mask.png"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "blueprint21_completion" / "inpainting_baseline_benchmark"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    mode: str
    backend: str
    face_mode: str = "off"
    codeformer_fidelity: float | None = None
    external_mask: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy benchmark backend inpainting cho demo3.")
    parser.add_argument("--image", default="", help="Ảnh demo3. Mặc định tự tìm trong data/demo_inputs/real_manual_3.")
    parser.add_argument("--oracle-mask", default=str(DEFAULT_ORACLE_MASK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def find_demo3_image() -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = DEFAULT_IMAGE_DIR / f"demo3{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Không tìm thấy demo3 trong {DEFAULT_IMAGE_DIR}")


def build_cases(oracle_mask: Path) -> list[BenchmarkCase]:
    return [
        BenchmarkCase("baseline_auto_simple_lama", "auto_r011_union_refined", "simple_lama"),
        BenchmarkCase("sensitive_auto_simple_lama", "auto_r011_sensitive_low_threshold", "simple_lama"),
        BenchmarkCase("oracle_simple_lama", "external", "simple_lama", external_mask=oracle_mask),
        BenchmarkCase("baseline_auto_opencv", "auto_r011_union_refined", "opencv"),
        BenchmarkCase("sensitive_auto_opencv", "auto_r011_sensitive_low_threshold", "opencv"),
        BenchmarkCase("oracle_opencv", "external", "opencv", external_mask=oracle_mask),
        BenchmarkCase(
            "sensitive_auto_simple_lama_codeformer",
            "auto_r011_sensitive_low_threshold",
            "simple_lama",
            face_mode="codeformer_if_available",
            codeformer_fidelity=0.7,
        ),
        BenchmarkCase(
            "oracle_simple_lama_codeformer",
            "external",
            "simple_lama",
            face_mode="codeformer_if_available",
            codeformer_fidelity=0.7,
            external_mask=oracle_mask,
        ),
    ]


def run_case(case: BenchmarkCase, image: Path, output_root: Path, device: str) -> Path:
    case_output_root = output_root / "case_runs" / case.label
    command = [
        sys.executable,
        "scripts\\run_restoration_pipeline.py",
        "--image",
        str(image),
        "--output-dir",
        str(case_output_root),
        "--mode",
        case.mode,
        "--backend",
        case.backend,
        "--device",
        device,
    ]
    if case.face_mode == "off":
        command.append("--skip-face-restoration")
    else:
        command.extend(["--face-mode", case.face_mode])
    if case.codeformer_fidelity is not None:
        command.extend(["--codeformer-fidelity", f"{case.codeformer_fidelity:.2f}"])
    if case.external_mask is not None:
        command.extend(["--external-mask", str(case.external_mask)])

    print("run:", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Case {case.label} lỗi với exit code {result.returncode}")
    return case_output_root / image.stem / case.mode


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def fit_tile(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = image_rgb.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    resized = cv2.resize(image_rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def add_label(tile: np.ndarray, label: str) -> np.ndarray:
    output = tile.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (20, 20, 20), thickness=-1)
    cv2.putText(output, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def build_grid(rows: list[dict[str, Any]], output_path: Path) -> None:
    wanted = [
        ("input.png", "Original"),
        ("baseline_auto_simple_lama", "Baseline SL"),
        ("sensitive_auto_simple_lama", "Sensitive SL"),
        ("oracle_simple_lama", "Oracle SL"),
        ("baseline_auto_opencv", "Baseline OpenCV"),
        ("sensitive_auto_opencv", "Sensitive OpenCV"),
        ("oracle_opencv", "Oracle OpenCV"),
        ("sensitive_auto_simple_lama_codeformer", "Sensitive SL+CF"),
        ("oracle_simple_lama_codeformer", "Oracle SL+CF"),
    ]
    row_by_label = {row["case_label"]: row for row in rows}
    tiles: list[np.ndarray] = []
    tile_size = (360, 360)
    for key, label in wanted:
        if key == "input.png":
            first = rows[0] if rows else {}
            image_path = Path(first.get("input_path", ""))
            tile = fit_tile(read_rgb(image_path), tile_size) if image_path.exists() else np.full((360, 360, 3), 235, np.uint8)
        else:
            row = row_by_label.get(key, {})
            image_path = Path(row.get("restored_final_path", ""))
            tile = fit_tile(read_rgb(image_path), tile_size) if image_path.exists() else np.full((360, 360, 3), 235, np.uint8)
        tiles.append(add_label(tile, label))
    grid = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def collect_row(case: BenchmarkCase, final_dir: Path) -> dict[str, Any]:
    metadata_path = final_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "case_label": case.label,
        "mode": case.mode,
        "backend_requested": case.backend,
        "backend_actual": metadata.get("inpainting_backend_actual", metadata.get("actual_backend", "")),
        "face_mode": metadata.get("face_mode", case.face_mode),
        "face_restoration_applied": metadata.get("face_restoration_applied", False),
        "mask_ratio": metadata.get("final_mask_ratio", ""),
        "input_path": str(final_dir / "input.png"),
        "final_mask_path": str(final_dir / "final_mask.png"),
        "overlay_path": str(final_dir / "overlay_final.png"),
        "restored_before_face_path": str(final_dir / "restored_before_face.png"),
        "restored_final_path": str(final_dir / "restored_final.png"),
        "comparison_grid_path": str(final_dir / "comparison_grid.png"),
        "metadata_path": str(metadata_path),
    }


def write_index(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    ratio_by_label = {row["case_label"]: row.get("mask_ratio", "") for row in rows}
    lines = [
        "# Inpainting Baseline Benchmark",
        "",
        "Benchmark này khóa input ở demo3 để so sánh backend inpainting với mask baseline, mask sensitive và oracle/manual mask.",
        "",
        "## Mask ratio",
        "",
        f"- baseline auto: {ratio_by_label.get('baseline_auto_simple_lama', '')}",
        f"- sensitive auto: {ratio_by_label.get('sensitive_auto_simple_lama', '')}",
        f"- oracle/manual: {ratio_by_label.get('oracle_simple_lama', '')}",
        "",
        "## Case đã chạy",
        "",
        "| case | mode | backend | face | mask_ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_label']} | {row['mode']} | {row['backend_actual'] or row['backend_requested']} | "
            f"{row['face_mode']} | {row['mask_ratio']} |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis thận trọng",
            "",
            "- Oracle/manual mask là upper-bound chẩn đoán cho demo3, không phải mode tự động.",
            "- Nếu oracle vẫn còn nứt hoặc bệt vùng mảnh, bottleneck còn lại nghiêng về backend inpainting.",
            "- Nếu sensitive rộng hơn oracle nhưng visual tốt hơn, có thể oracle còn thiếu halo/vết nứt mảnh hoặc inpainting cần vùng mask rộng hơn.",
            "- Kết luận cuối vẫn cần human visual review; script này không tự chấm chất lượng ảnh.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    image = resolve_path(args.image) if args.image else find_demo3_image()
    oracle_mask = resolve_path(args.oracle_mask)
    output_root = resolve_path(args.output_dir)
    if not oracle_mask.exists():
        raise FileNotFoundError(f"Không tìm thấy oracle mask: {oracle_mask}")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case in build_cases(oracle_mask):
        final_dir = run_case(case, image, output_root, args.device)
        rows.append(collect_row(case, final_dir))

    write_index(rows, output_root / "benchmark_index.csv")
    write_summary(rows, output_root / "INPAINTING_BASELINE_SUMMARY.md")
    build_grid(rows, output_root / "inpainting_baseline_comparison_grid.png")
    print(f"benchmark_index: {output_root / 'benchmark_index.csv'}")
    print(f"summary: {output_root / 'INPAINTING_BASELINE_SUMMARY.md'}")
    print(f"comparison_grid: {output_root / 'inpainting_baseline_comparison_grid.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
