from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "blueprint21_final_validation" / "official_lama_validation"
REQUIRED_FILES = [
    "input.png",
    "final_mask.png",
    "overlay_final.png",
    "restored_before_face.png",
    "restored_final.png",
    "comparison_grid.png",
    "metadata.json",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class ValidationCase:
    label: str
    demo_id: str
    mode: str
    backend: str
    face_mode: str
    fidelity: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy final validation cho official_lama trên demo1/demo2/demo3.")
    parser.add_argument("--input-root", default=str(INPUT_ROOT))
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def find_demo_image(input_root: Path, demo_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = input_root / f"{demo_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Không tìm thấy {demo_id}.png/.jpg/.jpeg trong {input_root}")


def build_cases() -> list[ValidationCase]:
    return [
        ValidationCase("demo1_r011_union_official_off", "demo1", "auto_r011_union_refined", "official_lama", "off"),
        ValidationCase(
            "demo1_r011_union_official_codeformer",
            "demo1",
            "auto_r011_union_refined",
            "official_lama",
            "codeformer_if_available",
            0.7,
        ),
        ValidationCase(
            "demo1_r011_union_simple_codeformer",
            "demo1",
            "auto_r011_union_refined",
            "simple_lama",
            "codeformer_if_available",
            0.7,
        ),
        ValidationCase("demo2_r011_union_official_off", "demo2", "auto_r011_union_refined", "official_lama", "off"),
        ValidationCase(
            "demo2_r011_union_official_codeformer",
            "demo2",
            "auto_r011_union_refined",
            "official_lama",
            "codeformer_if_available",
            0.7,
        ),
        ValidationCase(
            "demo2_r011_union_simple_codeformer",
            "demo2",
            "auto_r011_union_refined",
            "simple_lama",
            "codeformer_if_available",
            0.7,
        ),
        ValidationCase(
            "demo3_sensitive_official_off",
            "demo3",
            "auto_r011_sensitive_low_threshold",
            "official_lama",
            "off",
        ),
        ValidationCase(
            "demo3_sensitive_official_codeformer",
            "demo3",
            "auto_r011_sensitive_low_threshold",
            "official_lama",
            "codeformer_if_available",
            0.7,
        ),
        ValidationCase(
            "demo3_sensitive_simple_codeformer",
            "demo3",
            "auto_r011_sensitive_low_threshold",
            "simple_lama",
            "codeformer_if_available",
            0.7,
        ),
    ]


def run_case(case: ValidationCase, image_path: Path, output_root: Path) -> tuple[int, str, str, Path]:
    case_root = output_root / "case_runs" / case.label
    command = [
        sys.executable,
        "scripts\\run_restoration_pipeline.py",
        "--input",
        str(image_path),
        "--mode",
        case.mode,
        "--inpaint-backend",
        case.backend,
        "--face-mode",
        case.face_mode,
        "--output",
        str(case_root),
    ]
    if case.fidelity is not None:
        command.extend(["--codeformer-fidelity", f"{case.fidelity:.3f}"])
    print("run:", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.returncode, result.stdout, result.stderr, case_root / image_path.stem / case.mode


def read_metadata(final_dir: Path) -> dict[str, Any]:
    metadata_path = final_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def collect_row(case: ValidationCase, final_dir: Path, returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    metadata = read_metadata(final_dir)
    required_present = all((final_dir / filename).exists() for filename in REQUIRED_FILES)
    return {
        "case_label": case.label,
        "demo_id": case.demo_id,
        "mode": case.mode,
        "backend_requested": case.backend,
        "face_mode_requested": case.face_mode,
        "returncode": returncode,
        "pass": bool(returncode == 0 and required_present),
        "final_dir": str(final_dir),
        "inpainting_backend_requested": metadata.get("inpainting_backend_requested", metadata.get("backend_requested", "")),
        "inpainting_backend_actual": metadata.get("inpainting_backend_actual", metadata.get("actual_backend", "")),
        "official_lama_reason": metadata.get("official_lama_reason", ""),
        "face_restoration_applied": metadata.get("face_restoration_applied", ""),
        "face_backend": metadata.get("face_backend", ""),
        "processing_time_sec": metadata.get("processing_time_sec", ""),
        "final_mask_ratio": metadata.get("final_mask_ratio", ""),
        "required_present": required_present,
        "stdout_tail": stdout[-500:],
        "stderr_tail": stderr[-500:],
    }


def write_index(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fit_tile(image_rgb: np.ndarray, size: tuple[int, int] = (360, 360)) -> np.ndarray:
    target_width, target_height = size
    height, width = image_rgb.shape[:2]
    scale = min(target_width / max(width, 1), target_height / max(height, 1))
    resized = cv2.resize(
        image_rgb,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((target_height, target_width, 3), 245, dtype=np.uint8)
    y = (target_height - resized.shape[0]) // 2
    x = (target_width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def label_tile(tile: np.ndarray, label: str) -> np.ndarray:
    output = tile.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(output, label[:42], (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def build_contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    tiles: list[np.ndarray] = []
    for row in rows:
        final_dir = Path(str(row["final_dir"]))
        image_path = final_dir / "restored_final.png"
        if not image_path.exists():
            image_path = final_dir / "overlay_final.png"
        label = f"{row['demo_id']} | {row['inpainting_backend_actual']} | face={row['face_backend'] or row['face_mode_requested']}"
        tiles.append(label_tile(fit_tile(read_rgb(image_path)), label))
    if not tiles:
        return
    while len(tiles) % 3:
        tiles.append(label_tile(np.full_like(tiles[0], 245), "missing"))
    grid = np.vstack([np.hstack(tiles[index : index + 3]) for index in range(0, len(tiles), 3)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    passed = [row for row in rows if row["pass"]]
    failed = [row for row in rows if not row["pass"]]
    lines = [
        "# Official LaMa Final Validation Summary",
        "",
        f"- total cases: `{len(rows)}`",
        f"- pass count: `{len(passed)}`",
        f"- fail count: `{len(failed)}`",
        "- warning: needs human visual review before changing project defaults.",
        "",
        "## Cases",
        "",
        "| case | requested | actual | official_reason | face_applied | face_backend | runtime | pass |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_label']} | {row['inpainting_backend_requested']} | {row['inpainting_backend_actual']} | "
            f"{row['official_lama_reason']} | {row['face_restoration_applied']} | {row['face_backend']} | "
            f"{row['processing_time_sec']} | {row['pass']} |"
        )
    if failed:
        lines.extend(["", "## Failed cases", ""])
        for row in failed:
            lines.append(f"- `{row['case_label']}`: returncode `{row['returncode']}`, required_present `{row['required_present']}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in build_cases():
        image_path = find_demo_image(input_root, case.demo_id)
        returncode, stdout, stderr, final_dir = run_case(case, image_path, output_root)
        rows.append(collect_row(case, final_dir, returncode, stdout, stderr))
    write_index(rows, output_root / "validation_index.csv")
    write_summary(rows, output_root / "OFFICIAL_LAMA_VALIDATION_SUMMARY.md")
    build_contact_sheet(rows, output_root / "official_lama_validation_contact_sheet.png")
    pass_count = sum(1 for row in rows if row["pass"])
    print(f"validation: {pass_count}/{len(rows)} pass")
    print(output_root / "validation_index.csv")
    print(output_root / "official_lama_validation_contact_sheet.png")
    return 0 if pass_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
