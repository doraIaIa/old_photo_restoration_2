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
BENCHMARK_ROOT = PROJECT_ROOT / "outputs" / "final_demo_benchmark"
SHORTLIST_ROOT = BENCHMARK_ROOT / "shortlist_review"
INDEX_PATH = BENCHMARK_ROOT / "benchmark_index.csv"

DEMOS = ["demo1", "demo2", "demo3"]
VARIANTS = [
    ("auto_r011_union_refined", "simple_lama", "r011 union + simple_lama"),
    ("auto_r011_union_refined", "opencv", "r011 union + opencv"),
    ("auto_r012_union_refined", "simple_lama", "r012 union + simple_lama"),
    ("auto_r012_union_refined", "opencv", "r012 union + opencv"),
]
REQUIRED_FILES = [
    "final_mask.png",
    "overlay_final.png",
    "restored_before_face.png",
    "comparison_grid.png",
    "metadata.json",
]

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
    cv2.rectangle(canvas, (0, 0), (tile_w, 32), (22, 22, 22), thickness=-1)
    cv2.putText(canvas, label[:40], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def load_benchmark_rows() -> list[dict[str, str]]:
    with INDEX_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], demo_id: str, mode: str, backend: str) -> dict[str, str] | None:
    for row in rows:
        if (
            row.get("benchmark_group") == "main"
            and row.get("demo_id") == demo_id
            and row.get("mode") == mode
            and row.get("backend") == backend
            and row.get("face_mode") == "off"
        ):
            return row
    return None


def copy_case_files(source_dir: Path, target_dir: Path) -> dict[str, bool]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, bool] = {}
    for filename in REQUIRED_FILES:
        source = source_dir / filename
        copied[filename] = source.exists()
        if source.exists():
            shutil.copy2(source, target_dir / filename)
    return copied


def make_contact_sheet(demo_id: str, case_targets: list[tuple[str, Path]]) -> Path:
    column_tiles = []
    for label, case_dir in case_targets:
        overlay = read_rgb(case_dir / "overlay_final.png")
        restored = read_rgb(case_dir / "restored_before_face.png")
        header = np.full((52, 360, 3), 252, dtype=np.uint8)
        cv2.putText(header, label[:34], (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        column = np.concatenate(
            [
                header,
                fit_tile(overlay, "mask overlay"),
                fit_tile(restored, "restored before face"),
            ],
            axis=0,
        )
        column_tiles.append(column)
    sheet = np.concatenate(column_tiles, axis=1)
    output_path = SHORTLIST_ROOT / "contact_sheets" / f"{demo_id}_shortlist_contact.png"
    write_rgb(output_path, sheet)
    return output_path


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_guide(rows: list[dict[str, Any]], missing: list[str], sheet_paths: list[Path]) -> None:
    lines = [
        "# Shortlist Review Guide",
        "",
        "Mục tiêu: xem 12 case quan trọng nhất trước khi chọn final demo config.",
        "",
        "## Xem trước",
        "",
    ]
    lines.extend(f"- `{path}`" for path in sheet_paths)
    lines.extend(
        [
            "",
            "Không dùng CodeFormer trong shortlist này. Các case đều có `face_mode=off` để tập trung so sánh mask và inpainting.",
            "",
            "## Manual Scoring Table",
            "",
            "| demo | variant | crack_removed (Good/Acceptable/Bad) | artifacts (Low/Medium/High) | content_preserved (Good/Acceptable/Bad) | overall_choice (rank 1-4) | notes |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for demo_id in DEMOS:
        for _, _, label in VARIANTS:
            lines.append(f"| {demo_id} | {label} |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Case Index",
            "",
            "| demo | variant | copied | case_dir |",
            "|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(f"| {row['demo_id']} | {row['variant_label']} | {row['all_required_files_present']} | `{row['shortlist_case_dir']}` |")
    lines.extend(["", "## Missing Cases", ""])
    if missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("- Không thiếu case nào trong 12 case shortlist.")
    lines.extend(["", "Final config: `not selected yet`; cần human visual review.", ""])
    (SHORTLIST_ROOT / "SHORTLIST_REVIEW_GUIDE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    SHORTLIST_ROOT.mkdir(parents=True, exist_ok=True)
    rows = load_benchmark_rows()
    index_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    sheet_paths: list[Path] = []

    for demo_id in DEMOS:
        case_targets: list[tuple[str, Path]] = []
        for mode, backend, label in VARIANTS:
            row = find_row(rows, demo_id, mode, backend)
            if row is None:
                missing.append(f"{demo_id} {mode} {backend}")
                continue
            source_dir = Path(row["case_dir"])
            target_dir = SHORTLIST_ROOT / "cases" / demo_id / f"{mode}__{backend}"
            copied = copy_case_files(source_dir, target_dir)
            metadata = load_metadata(target_dir / "metadata.json")
            all_present = all(copied.values())
            index_rows.append(
                {
                    "demo_id": demo_id,
                    "mask_mode": mode,
                    "backend": backend,
                    "face_mode": "off",
                    "variant_label": label,
                    "source_case_dir": str(source_dir),
                    "shortlist_case_dir": str(target_dir),
                    "all_required_files_present": all_present,
                    "final_mask_present": copied["final_mask.png"],
                    "overlay_present": copied["overlay_final.png"],
                    "restored_before_face_present": copied["restored_before_face.png"],
                    "comparison_grid_present": copied["comparison_grid.png"],
                    "metadata_present": copied["metadata.json"],
                    "final_mask_ratio": metadata.get("final_mask_ratio", ""),
                    "inpainting_backend_actual": metadata.get("inpainting_backend_actual", metadata.get("actual_backend", "")),
                }
            )
            if all_present:
                case_targets.append((label, target_dir))
            else:
                missing.append(f"{demo_id} {mode} {backend}: thiếu file required")
        if len(case_targets) == 4:
            sheet_paths.append(make_contact_sheet(demo_id, case_targets))
        else:
            missing.append(f"{demo_id}: không đủ 4 variant để tạo contact sheet")

    fields = [
        "demo_id",
        "mask_mode",
        "backend",
        "face_mode",
        "variant_label",
        "source_case_dir",
        "shortlist_case_dir",
        "all_required_files_present",
        "final_mask_present",
        "overlay_present",
        "restored_before_face_present",
        "comparison_grid_present",
        "metadata_present",
        "final_mask_ratio",
        "inpainting_backend_actual",
    ]
    with (SHORTLIST_ROOT / "shortlist_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)
    write_guide(index_rows, missing, sheet_paths)
    print(f"shortlist_root: {SHORTLIST_ROOT}")
    print(f"cases: {len(index_rows)}")
    print(f"contact_sheets: {len(sheet_paths)}")
    print(f"missing: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
