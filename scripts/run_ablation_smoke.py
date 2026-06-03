from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ablation_smoke"
DEFAULT_MANUAL_MASK_DIR = DEFAULT_INPUT_DIR / "manual_masks"
RESULT_COLUMNS = [
    "image_id",
    "case_id",
    "mask_mode",
    "backend_actual",
    "face_restoration",
    "codeformer_fidelity",
    "fallback_applied",
    "mask_ratio",
    "runtime_seconds",
    "reviewer_score_1_5",
    "notes",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class AblationCase:
    case_id: str
    mask_mode: str
    backend: str
    face_restoration: bool = False
    codeformer_fidelity: float | None = None
    requires_manual_mask: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy ablation smoke trên demo input sẵn có, có skip rõ khi thiếu mask.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--manual-mask-dir", default=str(DEFAULT_MANUAL_MASK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--backend", default="official_lama", choices=["official_lama", "simple_lama", "opencv"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--write-template", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_template(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ablation_results_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow({column: "" for column in RESULT_COLUMNS})


def find_images(input_dir: Path, max_images: int) -> list[Path]:
    if not input_dir.exists():
        return []
    images = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return images[:max_images] if max_images > 0 else images


def manual_mask_for(image_path: Path, manual_mask_dir: Path) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = manual_mask_dir / f"{image_path.stem}_mask{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_cases(default_backend: str) -> list[AblationCase]:
    return [
        AblationCase("A_opencv_fallback_baseline", "auto_r011_union_refined", "opencv"),
        AblationCase("B_r011_default_official_face_off", "auto_r011_union_refined", default_backend),
        AblationCase("C_r011_sensitive_official_face_off", "auto_r011_sensitive_low_threshold", default_backend),
        AblationCase(
            "D_r011_default_official_codeformer_0p7",
            "auto_r011_union_refined",
            default_backend,
            face_restoration=True,
            codeformer_fidelity=0.7,
        ),
        AblationCase("E_oracle_mask_official", "external", default_backend, requires_manual_mask=True),
    ]


def run_case(
    image_path: Path,
    case: AblationCase,
    output_dir: Path,
    device: str,
    external_mask: Path | None = None,
) -> tuple[dict[str, Any], float]:
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
        str(output_dir / "case_runs" / case.case_id),
        "--device",
        device,
    ]
    if case.face_restoration:
        command.extend(["--face-mode", "codeformer_if_available"])
        if case.codeformer_fidelity is not None:
            command.extend(["--codeformer-fidelity", f"{case.codeformer_fidelity:.2f}"])
    else:
        command.append("--skip-face-restoration")
    if external_mask is not None:
        command.extend(["--external-mask", str(external_mask)])
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    runtime = time.perf_counter() - started
    final_dir = output_dir / "case_runs" / case.case_id / image_path.stem / case.mask_mode
    metadata_path = final_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["_returncode"] = completed.returncode
    metadata["_stderr_tail"] = completed.stderr[-400:]
    return metadata, runtime


def result_row(
    image_path: Path,
    case: AblationCase,
    metadata: dict[str, Any],
    runtime: float | None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "image_id": image_path.stem,
        "case_id": case.case_id,
        "mask_mode": case.mask_mode,
        "backend_actual": metadata.get("inpainting_backend_actual", metadata.get("actual_backend", "")),
        "face_restoration": "on" if case.face_restoration else "off",
        "codeformer_fidelity": case.codeformer_fidelity if case.codeformer_fidelity is not None else "",
        "fallback_applied": metadata.get("fallback_applied", ""),
        "mask_ratio": metadata.get("final_mask_ratio", ""),
        "runtime_seconds": f"{runtime:.3f}" if runtime is not None else "",
        "reviewer_score_1_5": "",
        "notes": notes or metadata.get("_stderr_tail", ""),
    }


def write_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ablation_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    manual_mask_dir = resolve_path(args.manual_mask_dir)
    output_dir = resolve_path(args.output_dir)
    write_template(output_dir)
    if args.write_template:
        print(f"Đã ghi ablation_results_template.csv vào {output_dir}")
        return 0

    images = find_images(input_dir, args.max_images)
    if not images:
        print(f"Không tìm thấy ảnh demo trong {input_dir}. Chỉ ghi template.", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for image_path in images:
        for case in build_cases(args.backend):
            external_mask = manual_mask_for(image_path, manual_mask_dir) if case.requires_manual_mask else None
            if case.requires_manual_mask and external_mask is None:
                rows.append(result_row(image_path, case, {}, None, f"skip: thiếu manual mask cho {image_path.name}"))
                continue
            metadata, runtime = run_case(image_path, case, output_dir, args.device, external_mask=external_mask)
            note = "" if metadata.get("_returncode") == 0 else f"failed: returncode={metadata.get('_returncode')}"
            rows.append(result_row(image_path, case, metadata, runtime, note))

    write_results(output_dir, rows)
    print(f"ablation_results: {output_dir / 'ablation_results.csv'}")
    print("Script không tự kết luận case tốt nhất; cần review trực quan thật.")
    failed = [row for row in rows if str(row["notes"]).startswith("failed")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
