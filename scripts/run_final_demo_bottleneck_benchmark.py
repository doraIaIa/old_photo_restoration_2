from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_INPUT_ROOT = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_demo_benchmark"
R012_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r012-manual-repair-ft-s42" / "best_iou.ckpt"

BASE_MODES = [
    "auto_r011",
    "auto_r011_refined",
    "auto_r011_union_refined",
    "auto_r012",
    "auto_r012_refined",
    "auto_r012_union_refined",
]
BACKENDS = ["simple_lama", "opencv"]
FACE_MODES = ["off", "codeformer_if_available"]
SWEEP_THRESHOLDS = [0.1, 0.2, 0.3, 0.5]
SWEEP_DILATIONS = [3, 5, 7]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class Case:
    demo_id: str
    image_path: Path
    mode: str
    backend: str
    face_mode: str
    output_dir: Path
    threshold: float = 0.70
    mask_dilate: int = 0
    external_mask: Path | None = None
    benchmark_group: str = "main"

    @property
    def case_id(self) -> str:
        threshold_tag = f"t{self.threshold:.2f}".replace(".", "p")
        face_tag = "face_codeformer" if self.face_mode == "codeformer_if_available" else "face_off"
        dilate_tag = f"d{self.mask_dilate}"
        return f"{self.demo_id}__{self.mode}__{self.backend}__{face_tag}__{threshold_tag}__{dilate_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy benchmark bottleneck cho final demo.")
    parser.add_argument("--input-root", default=str(DEMO_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--codeformer-fidelity", type=float, default=0.7)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-mask-sweep", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def find_demo_images(input_root: Path) -> list[Path]:
    images = []
    for name in ["demo1", "demo2", "demo3"]:
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = input_root / f"{name}{ext}"
            if candidate.exists():
                images.append(candidate)
                break
    return images


def find_manual_mask(input_root: Path, demo_id: str) -> Path | None:
    candidates = [
        input_root / f"{demo_id}_mask.png",
        input_root / f"{demo_id}_manual_mask.png",
        input_root / "masks" / f"{demo_id}_mask.png",
        input_root / "manual_masks" / f"{demo_id}_mask.png",
        input_root / "external_masks" / f"{demo_id}_mask.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_cases(input_root: Path, output_root: Path, include_mask_sweep: bool) -> tuple[list[Case], dict[str, Path]]:
    images = find_demo_images(input_root)
    manual_masks = {image.stem: find_manual_mask(input_root, image.stem) for image in images}
    cases: list[Case] = []
    modes = [mode for mode in BASE_MODES if not mode.startswith("auto_r012") or R012_CHECKPOINT.exists()]
    for image_path in images:
        demo_id = image_path.stem
        for mode in modes:
            for backend in BACKENDS:
                for face_mode in FACE_MODES:
                    cases.append(
                        Case(
                            demo_id=demo_id,
                            image_path=image_path,
                            mode=mode,
                            backend=backend,
                            face_mode=face_mode,
                            output_dir=output_root / "cases" / demo_id / mode / backend / face_mode,
                        )
                    )
        manual_mask = manual_masks.get(demo_id)
        if manual_mask is not None:
            for backend in BACKENDS:
                for face_mode in FACE_MODES:
                    cases.append(
                        Case(
                            demo_id=demo_id,
                            image_path=image_path,
                            mode="external",
                            backend=backend,
                            face_mode=face_mode,
                            external_mask=manual_mask,
                            output_dir=output_root / "cases" / demo_id / "external_manual" / backend / face_mode,
                            benchmark_group="oracle",
                        )
                    )
        if include_mask_sweep:
            for threshold in SWEEP_THRESHOLDS:
                for dilation in SWEEP_DILATIONS:
                    cases.append(
                        Case(
                            demo_id=demo_id,
                            image_path=image_path,
                            mode="auto_r011_union_refined",
                            backend="simple_lama",
                            face_mode="off",
                            threshold=threshold,
                            mask_dilate=dilation,
                            output_dir=output_root / "mask_refinement_sweep" / demo_id / f"t{threshold:.1f}".replace(".", "p") / f"d{dilation}",
                            benchmark_group="mask_refinement_sweep",
                        )
                    )
    existing_manual = {demo_id: mask for demo_id, mask in manual_masks.items() if mask is not None}
    return cases, existing_manual


def final_case_dir(case: Case) -> Path:
    return case.output_dir / case.demo_id / case.mode


def command_for_case(case: Case, codeformer_fidelity: float) -> list[str]:
    fallback_threshold = min(0.40, case.threshold)
    command = [
        sys.executable,
        "scripts\\run_restoration_pipeline.py",
        "--image",
        str(case.image_path),
        "--mode",
        case.mode,
        "--backend",
        case.backend,
        "--face-mode",
        case.face_mode,
        "--codeformer-fidelity",
        f"{codeformer_fidelity:.3f}",
        "--threshold",
        f"{case.threshold:.2f}",
        "--fallback-threshold",
        f"{fallback_threshold:.2f}",
        "--mask-dilate",
        str(case.mask_dilate),
        "--output-dir",
        str(case.output_dir),
    ]
    if case.external_mask is not None:
        command.extend(["--external-mask", str(case.external_mask)])
    return command


def run_case(case: Case, codeformer_fidelity: float, skip_existing: bool) -> dict[str, Any]:
    case_dir = final_case_dir(case)
    metadata_path = case_dir / "metadata.json"
    if skip_existing and metadata_path.exists():
        metadata = load_json(metadata_path)
        return build_row(case, "skipped_existing", 0, "", "", metadata)

    command = command_for_case(case, codeformer_fidelity)
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    metadata = load_json(metadata_path)
    status = "pass" if result.returncode == 0 else "fail"
    if result.returncode != 0:
        case_dir.mkdir(parents=True, exist_ok=True)
        failure_payload = {
            "status": "benchmark_case_failed",
            "command": command,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
        (case_dir / "benchmark_failure.json").write_text(json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return build_row(case, status, result.returncode, result.stdout[-1000:], result.stderr[-1000:], metadata)


def build_row(case: Case, status: str, returncode: int | None, stdout_tail: str, stderr_tail: str, metadata: dict[str, Any]) -> dict[str, Any]:
    case_dir = final_case_dir(case)
    return {
        "case_id": case.case_id,
        "benchmark_group": case.benchmark_group,
        "demo_id": case.demo_id,
        "mode": case.mode,
        "backend": case.backend,
        "face_mode": case.face_mode,
        "threshold": case.threshold,
        "mask_dilate": case.mask_dilate,
        "external_mask": str(case.external_mask) if case.external_mask else "",
        "status": status,
        "returncode": returncode,
        "inpainting_backend_actual": metadata.get("inpainting_backend_actual", metadata.get("actual_backend", "")),
        "face_restoration_applied": metadata.get("face_restoration_applied", ""),
        "face_backend": metadata.get("face_backend", metadata.get("face_restoration_backend", "")),
        "codeformer_fidelity": metadata.get("codeformer_fidelity", ""),
        "final_mask_ratio": metadata.get("final_mask_ratio", ""),
        "case_dir": str(case_dir),
        "original_exists": (case_dir / "input.png").exists(),
        "final_mask_exists": (case_dir / "final_mask.png").exists(),
        "mask_overlay_exists": (case_dir / "overlay_final.png").exists(),
        "restored_before_face_exists": (case_dir / "restored_before_face.png").exists(),
        "restored_final_exists": (case_dir / "restored_final.png").exists(),
        "comparison_grid_exists": (case_dir / "comparison_grid.png").exists(),
        "metadata_path": str(case_dir / "metadata.json") if (case_dir / "metadata.json").exists() else "",
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def write_index(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "benchmark_group",
        "demo_id",
        "mode",
        "backend",
        "face_mode",
        "threshold",
        "mask_dilate",
        "external_mask",
        "status",
        "returncode",
        "inpainting_backend_actual",
        "face_restoration_applied",
        "face_backend",
        "codeformer_fidelity",
        "final_mask_ratio",
        "case_dir",
        "original_exists",
        "final_mask_exists",
        "mask_overlay_exists",
        "restored_before_face_exists",
        "restored_final_exists",
        "comparison_grid_exists",
        "metadata_path",
        "stdout_tail",
        "stderr_tail",
    ]
    with (output_root / "benchmark_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_diagnosis(output_root: Path, rows: list[dict[str, Any]], manual_masks: dict[str, Path]) -> None:
    total = len(rows)
    passed = sum(1 for row in rows if row["status"] in {"pass", "skipped_existing"})
    failed = total - passed
    oracle_rows = [row for row in rows if row["benchmark_group"] == "oracle"]
    sweep_rows = [row for row in rows if row["benchmark_group"] == "mask_refinement_sweep"]
    main_rows = [row for row in rows if row["benchmark_group"] == "main"]
    codeformer_rows = [row for row in rows if row["face_mode"] == "codeformer_if_available"]
    codeformer_applied = sum(1 for row in codeformer_rows if str(row["face_restoration_applied"]) == "True")
    lines = [
        "# Final Demo Bottleneck Diagnosis",
        "",
        f"- Output root: `{output_root}`",
        f"- Total cases: `{total}`",
        f"- Passed/skipped cases: `{passed}`",
        f"- Failed cases: `{failed}`",
        f"- Main benchmark cases: `{len(main_rows)}`",
        f"- Mask refinement sweep cases: `{len(sweep_rows)}`",
        f"- Oracle/manual mask cases: `{len(oracle_rows)}`",
        f"- CodeFormer applied cases: `{codeformer_applied}/{len(codeformer_rows)}`",
        "",
        "## Oracle Mask Diagnosis",
        "",
    ]
    if manual_masks:
        lines.append("Manual/external masks were found for:")
        lines.extend(f"- `{demo_id}`: `{path}`" for demo_id, path in manual_masks.items())
        lines.extend(
            [
                "",
                "Oracle cases exist, but quality still needs human visual review.",
                "If oracle/manual-mask restorations are still poor, bottleneck likely shifts toward inpainting.",
                "If oracle/manual-mask restorations are clearly better than automatic masks, bottleneck likely is mask quality.",
            ]
        )
    else:
        lines.append("No manual/external masks were found for demo1/demo2/demo3, so oracle diagnosis was skipped.")
    lines.extend(
        [
            "",
            "## Current Bottleneck Signal",
            "",
            "Metadata can verify case coverage, backend use, mask ratio, and CodeFormer activation, but cannot prove visual quality.",
            "Bottleneck direction is therefore `needs human review` until review sheets are inspected.",
            "",
            "## Required Human Review",
            "",
            "- Compare automatic masks vs restored outputs per case.",
            "- Compare simple_lama vs opencv for identical masks.",
            "- If manual/oracle masks are later added, rerun this benchmark to separate mask quality from inpainting quality.",
            "",
        ]
    )
    (output_root / "BOTTLENECK_DIAGNOSIS.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_config_candidates(output_root: Path, manual_masks: dict[str, Path]) -> None:
    lines = [
        "# Final Config Candidates",
        "",
        "Status: `not selected yet` because visual evidence still needs human review.",
        "",
        "## Candidate A: Stable Automatic Config",
        "",
        "- Mask: `auto_r011_union_refined`",
        "- Inpainting: `simple_lama` first, compare with `opencv` output",
        "- Face restoration: `codeformer_if_available` only when face identity/detail review accepts it",
        "- CodeFormer fidelity: `0.7`",
        "- Pros: uses stable r011 segmentation baseline and activated CodeFormer backend.",
        "- Cons: current visual quality may still be limited by mask or inpainting behavior.",
        "",
        "## Candidate B: Experimental r012 Config",
        "",
        "- Mask: `auto_r012_union_refined`",
        "- Inpainting: `simple_lama` first, compare with `opencv` output",
        "- Face restoration: `codeformer_if_available`, fidelity `0.7`",
        "- Pros: tests real/manual-mask fine-tuned segmentation path.",
        "- Cons: r012 only improved slightly on a very small test split; do not make it default without visual review.",
        "",
        "## Candidate C: Oracle/Manual Mask Config",
        "",
    ]
    if manual_masks:
        lines.extend(
            [
                "- Mask: `external` manual mask",
                "- Purpose: diagnosis ceiling, not automatic final config.",
                "- Pros: separates inpainting quality from automatic mask quality.",
                "- Cons: not usable as automatic demo unless mask is provided by user.",
            ]
        )
    else:
        lines.extend(
            [
                "- Not available in this run because demo manual/external masks were not found.",
                "- Add demo masks and rerun benchmark to test oracle ceiling.",
            ]
        )
    lines.extend(["", "Selection: `not selected yet`; inspect review sheets first.", ""])
    (output_root / "FINAL_CONFIG_CANDIDATES.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    cases, manual_masks = build_cases(input_root, output_root, include_mask_sweep=args.include_mask_sweep)
    if not cases:
        raise RuntimeError(f"Không tìm thấy demo image trong {input_root}")

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}")
        row = run_case(case, args.codeformer_fidelity, skip_existing=args.skip_existing)
        rows.append(row)
        print(f"  status={row['status']} backend={row['inpainting_backend_actual']} face={row['face_backend']}")

    write_index(output_root, rows)
    write_diagnosis(output_root, rows, manual_masks)
    write_final_config_candidates(output_root, manual_masks)
    print(f"benchmark_index: {output_root / 'benchmark_index.csv'}")
    print(f"diagnosis: {output_root / 'BOTTLENECK_DIAGNOSIS.md'}")
    print(f"final_candidates: {output_root / 'FINAL_CONFIG_CANDIDATES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
