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
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "batch_review"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
REVIEW_COLUMNS = [
    "image_id",
    "config_name",
    "mask_mode",
    "backend_requested",
    "backend_actual",
    "fallback_applied",
    "mask_ratio",
    "face_restoration",
    "codeformer_fidelity",
    "status",
    "mask_error",
    "inpainting_error",
    "codeformer_error",
    "overall_quality_1_5",
    "notes",
]
DATASET_COLUMNS = [
    "image_id",
    "filename",
    "split",
    "source",
    "has_manual_mask",
    "mask_path",
    "has_face",
    "damage_level",
    "damage_type",
    "hard_negative_type",
    "notes",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class BatchCase:
    config_name: str
    mask_mode: str
    backend: str
    face_restoration: str
    codeformer_fidelity: float


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy batch review scaffold cho demo data hoặc data mới sau này.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--configs",
        default="r011_default",
        help="Danh sách config cách nhau bằng dấu phẩy. Giá trị hiện dùng làm nhãn review.",
    )
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--backend", default="official_lama", choices=["official_lama", "simple_lama", "opencv"])
    parser.add_argument(
        "--mask-modes",
        default="auto_r011_union_refined",
        help="Danh sách mode pipeline cách nhau bằng dấu phẩy.",
    )
    parser.add_argument("--face-restoration", default="off", choices=["on", "off"])
    parser.add_argument("--codeformer-fidelity", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--write-template", action="store_true", help="Chỉ ghi CSV template, không chạy pipeline.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_templates(output_dir: Path) -> None:
    review_template = {column: "" for column in REVIEW_COLUMNS}
    dataset_template = {column: "" for column in DATASET_COLUMNS}
    write_csv(output_dir / "review_sheet_template.csv", REVIEW_COLUMNS, [review_template])
    write_csv(output_dir / "dataset_index_template.csv", DATASET_COLUMNS, [dataset_template])


def find_images(input_dir: Path, max_images: int) -> list[Path]:
    if not input_dir.exists():
        return []
    images = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return images[:max_images] if max_images > 0 else images


def build_cases(args: argparse.Namespace) -> list[BatchCase]:
    configs = parse_csv_list(args.configs)
    mask_modes = parse_csv_list(args.mask_modes)
    cases: list[BatchCase] = []
    for config in configs:
        for mask_mode in mask_modes:
            cases.append(
                BatchCase(
                    config_name=config,
                    mask_mode=mask_mode,
                    backend=args.backend,
                    face_restoration=args.face_restoration,
                    codeformer_fidelity=float(args.codeformer_fidelity),
                )
            )
    return cases


def run_pipeline(image_path: Path, case: BatchCase, output_dir: Path, device: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts\\run_restoration_pipeline.py",
        "--image",
        str(image_path),
        "--mode",
        case.mask_mode,
        "--backend",
        case.backend,
        "--output-dir",
        str(output_dir / "case_runs" / case.config_name),
        "--device",
        device,
    ]
    if case.face_restoration == "on":
        command.extend(["--face-mode", "codeformer_if_available", "--codeformer-fidelity", f"{case.codeformer_fidelity:.2f}"])
    else:
        command.append("--skip-face-restoration")

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    final_dir = output_dir / "case_runs" / case.config_name / image_path.stem / case.mask_mode
    metadata_path = final_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1200:],
        "stderr_tail": completed.stderr[-1200:],
        "final_dir": final_dir,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "command": command,
    }


def review_row(image_path: Path, case: BatchCase, result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata", {})
    ok = result.get("returncode") == 0 and Path(result["metadata_path"]).exists()
    return {
        "image_id": image_path.stem,
        "config_name": case.config_name,
        "mask_mode": case.mask_mode,
        "backend_requested": case.backend,
        "backend_actual": metadata.get("inpainting_backend_actual", metadata.get("actual_backend", "")),
        "fallback_applied": metadata.get("fallback_applied", ""),
        "mask_ratio": metadata.get("final_mask_ratio", ""),
        "face_restoration": case.face_restoration,
        "codeformer_fidelity": case.codeformer_fidelity if case.face_restoration == "on" else "",
        "status": "done" if ok else "failed",
        "mask_error": "",
        "inpainting_error": "",
        "codeformer_error": "",
        "overall_quality_1_5": "",
        "notes": "" if ok else (result.get("stderr_tail") or result.get("stdout_tail") or "pipeline failed")[-300:],
    }


def dataset_row(image_path: Path) -> dict[str, Any]:
    return {
        "image_id": image_path.stem,
        "filename": image_path.name,
        "split": "demo",
        "source": "existing_demo",
        "has_manual_mask": "",
        "mask_path": "",
        "has_face": "",
        "damage_level": "",
        "damage_type": "",
        "hard_negative_type": "",
        "notes": "",
    }


def read_rgb_or_blank(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        tile = np.full((320, 320, 3), 240, dtype=np.uint8)
        cv2.putText(tile, label[:30], (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
        return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fit_tile(image_rgb: np.ndarray, size: tuple[int, int] = (320, 320)) -> np.ndarray:
    target_width, target_height = size
    height, width = image_rgb.shape[:2]
    scale = min(target_width / max(width, 1), target_height / max(height, 1))
    resized = cv2.resize(image_rgb, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width, 3), 245, dtype=np.uint8)
    y = (target_height - resized.shape[0]) // 2
    x = (target_width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def add_label(tile_rgb: np.ndarray, label: str) -> np.ndarray:
    tile = tile_rgb.copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(tile, label[:42], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def build_comparison_grid(rows: list[dict[str, Any]], output_path: Path) -> None:
    tiles: list[np.ndarray] = []
    for row in rows:
        if row["status"] != "done":
            continue
        final_dir = Path(str(row.get("output_dir", ""))) if row.get("output_dir") else None
        if final_dir is None:
            continue
        image_path = final_dir / "restored_final.png"
        label = f"{row['image_id']} | {row['config_name']} | {row['backend_actual']}"
        tiles.append(add_label(fit_tile(read_rgb_or_blank(image_path, label)), label))
    if not tiles:
        return
    while len(tiles) % 3:
        tiles.append(add_label(np.full_like(tiles[0], 245), ""))
    grid = np.vstack([np.hstack(tiles[index : index + 3]) for index in range(0, len(tiles), 3)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_templates(output_dir)
    if args.write_template:
        print(f"Đã ghi template vào {output_dir}")
        return 0

    images = find_images(input_dir, args.max_images)
    if not images:
        print(
            f"Không tìm thấy ảnh trong {input_dir}. Đã ghi review_sheet_template.csv và dataset_index_template.csv.",
            file=sys.stderr,
        )
        return 2

    rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for image_path in images:
        index_rows.append(dataset_row(image_path))
        for case in build_cases(args):
            result = run_pipeline(image_path, case, output_dir, args.device)
            row = review_row(image_path, case, result)
            row["output_dir"] = str(result["final_dir"])
            row["metadata_json"] = str(result["metadata_path"]) if result["metadata_path"].exists() else ""
            rows.append(row)
            if result["metadata_path"].exists():
                per_image_dir = output_dir / "metadata" / image_path.stem / case.config_name / case.mask_mode
                per_image_dir.mkdir(parents=True, exist_ok=True)
                (per_image_dir / "metadata.json").write_text(
                    json.dumps(result["metadata"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    batch_columns = DATASET_COLUMNS
    review_columns = REVIEW_COLUMNS + ["output_dir", "metadata_json"]
    write_csv(output_dir / "batch_index.csv", batch_columns, index_rows)
    write_csv(output_dir / "review_sheet.csv", review_columns, rows)
    build_comparison_grid(rows, output_dir / "comparison_grid.png")
    print(f"batch_index: {output_dir / 'batch_index.csv'}")
    print(f"review_sheet: {output_dir / 'review_sheet.csv'}")
    print(f"comparison_grid: {output_dir / 'comparison_grid.png'}")
    failed = [row for row in rows if row["status"] != "done"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
