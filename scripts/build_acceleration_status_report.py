from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DATA_ROOT = Path(r"F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "blueprint21_acceleration"
R011_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r011-repair-ft-s42" / "best_iou.ckpt"
R012_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r012-manual-repair-ft-s42" / "best_iou.ckpt"
R012_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "r012_manual_repair"
R012_VISUAL_REVIEW_DIR = PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "r012_visual_review"
R012_RESTORATION_COMPARISON_DIR = PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "r012_restoration_comparison"
CODEFORMER_FIDELITY_SWEEP_DIR = PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "codeformer_fidelity_sweep"
FINE_TUNED_LAMA_MARKERS = [
    PROJECT_ROOT / "checkpoints" / "lama" / "fine_tuned_lama",
    PROJECT_ROOT / "configs" / "lama_finetuned.yaml",
]


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def git_status_short() -> str:
    result = subprocess.run(
        ["git", "status", "-u", "--short"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return result.stdout.strip()


def app_import_pass() -> bool:
    try:
        import app_gradio  # noqa: F401

        return True
    except Exception:
        return False


def manual_mask_count() -> int:
    masks_dir = DATA_ROOT / "masks_repair_manual"
    images_dir = DATA_ROOT / "images"
    if not masks_dir.exists():
        return 0
    count = 0
    for mask_path in sorted(masks_dir.glob("*_mask.png")):
        sample_id = mask_path.stem[:-5] if mask_path.stem.endswith("_mask") else mask_path.stem
        if not any((images_dir / f"{sample_id}{ext}").exists() for ext in [".jpg", ".png", ".jpeg"]):
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if not set(int(value) for value in np.unique(mask)).issubset({0, 255}):
            continue
        if np.count_nonzero(mask > 127) == 0:
            continue
        count += 1
    return count


def latest_metric_files() -> list[str]:
    candidates = []
    for pattern in [
        "outputs/report_assets/*/metrics_summary.csv",
        "outputs/report_assets/*/SUMMARY.md",
        "outputs/**/summary.json",
        "outputs/**/metrics_summary.csv",
    ]:
        candidates.extend(PROJECT_ROOT.glob(pattern))
    return sorted(str(path.relative_to(PROJECT_ROOT)) for path in set(candidates))


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def best_iou(path: Path) -> float | None:
    summary = load_summary(path)
    metrics = summary.get("best_metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("iou")
    return float(value) if value is not None else None


def r012_metrics_summary() -> tuple[str, str]:
    r011_test_iou = best_iou(R012_OUTPUT_DIR / "eval_r011" / "test" / "summary.json")
    r012_test_iou = best_iou(R012_OUTPUT_DIR / "eval" / "test" / "summary.json")
    r011_val_iou = best_iou(R012_OUTPUT_DIR / "eval_r011" / "val" / "summary.json")
    r012_val_iou = best_iou(R012_OUTPUT_DIR / "eval" / "val" / "summary.json")
    if r011_test_iou is None or r012_test_iou is None:
        return "missing", "Chưa tìm thấy đủ r011/r012 test summary."
    delta_test = r012_test_iou - r011_test_iou
    delta_val = None if r011_val_iou is None or r012_val_iou is None else r012_val_iou - r011_val_iou
    val_text = "missing" if delta_val is None else f"{delta_val:+.6f}"
    detail = (
        f"test IoU r012={r012_test_iou:.6f}, r011={r011_test_iou:.6f}, delta={delta_test:+.6f}; "
        f"val delta={val_text}. Warning: test split rất nhỏ."
    )
    return "found", detail


def load_face_dependency_status() -> dict[str, Any]:
    return load_summary(OUTPUT_DIR / "face_dependency_status.json")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["area", "item", "status", "details"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], git_status: str, metric_files: list[str]) -> None:
    lines = [
        "# Blueprint 2.1 Acceleration Status",
        "",
        "## Module Status",
        "",
        "| Area | Item | Status | Details |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['area']} | {row['item']} | {row['status']} | {row['details']} |")
    lines.extend(["", "## Latest metric files", ""])
    if metric_files:
        lines.extend(f"- `{path}`" for path in metric_files[:20])
    else:
        lines.append("- Không tìm thấy metric file.")
    lines.extend(["", "## Git status", "", "```text", git_status or "(clean)", "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manual_count = manual_mask_count()
    simple_lama_available = package_available("simple_lama_inpainting")
    face_deps = {
        name: package_available(name)
        for name in ["torch", "cv2", "basicsr", "facexlib", "gfpgan", "codeformer"]
    }
    codeformer_output_exists = any((PROJECT_ROOT / "outputs").glob("**/face_module/face_output.png"))
    git_status = git_status_short()
    metric_files = latest_metric_files()
    r012_metric_status, r012_metric_details = r012_metrics_summary()
    face_dependency_status = load_face_dependency_status()
    subprocess_codeformer = face_dependency_status.get("subprocess_codeformer", {})

    rows = [
        {
            "area": "Module 1",
            "item": "r011 checkpoint",
            "status": "exists" if R011_CHECKPOINT.exists() else "missing",
            "details": str(R011_CHECKPOINT),
        },
        {
            "area": "Module 1",
            "item": "r012 checkpoint",
            "status": "exists" if R012_CHECKPOINT.exists() else "missing",
            "details": str(R012_CHECKPOINT),
        },
        {
            "area": "Module 1",
            "item": "r012 metrics summary",
            "status": r012_metric_status,
            "details": r012_metric_details,
        },
        {
            "area": "Module 1",
            "item": "r011 vs r012 delta warning",
            "status": "small_sample",
            "details": "r012 chỉ nhỉnh hơn r011 nhẹ trên test split 2 mẫu; cần visual review, không dùng để claim vượt trội.",
        },
        {
            "area": "Module 1",
            "item": "r012 visual review folder",
            "status": "exists" if R012_VISUAL_REVIEW_DIR.exists() else "missing",
            "details": str(R012_VISUAL_REVIEW_DIR),
        },
        {
            "area": "Module 1",
            "item": "r012 restoration comparison folder",
            "status": "exists" if R012_RESTORATION_COMPARISON_DIR.exists() else "missing",
            "details": str(R012_RESTORATION_COMPARISON_DIR),
        },
        {
            "area": "Module 1",
            "item": "manual repair masks",
            "status": "ready" if manual_count >= 10 else "not_ready",
            "details": f"{manual_count} valid masks; cần >= 10 để bắt đầu workflow r012.",
        },
        {
            "area": "Module 1",
            "item": "latest segmentation metrics",
            "status": "found" if metric_files else "missing",
            "details": f"{len(metric_files)} metric/report files found.",
        },
        {
            "area": "Module 2",
            "item": "simple_lama",
            "status": "available" if simple_lama_available else "missing",
            "details": "Baseline pretrained backend.",
        },
        {
            "area": "Module 2",
            "item": "fine_tuned_lama checkpoint/config",
            "status": "exists" if any(path.exists() for path in FINE_TUNED_LAMA_MARKERS) else "missing",
            "details": "; ".join(str(path) for path in FINE_TUNED_LAMA_MARKERS),
        },
        {
            "area": "Module 2",
            "item": "LPIPS/FID report",
            "status": "exists" if any((PROJECT_ROOT / "outputs").glob("**/*LPIPS*")) or any((PROJECT_ROOT / "outputs").glob("**/*FID*")) else "missing",
            "details": "Chưa claim fine-tuned LaMa nếu thiếu report.",
        },
        {
            "area": "Module 3",
            "item": "wrapper",
            "status": "exists" if (PROJECT_ROOT / "src" / "restoration" / "face_restoration.py").exists() else "missing",
            "details": "Dependency-gated wrapper.",
        },
        {
            "area": "Module 3",
            "item": "dependency check",
            "status": "partial" if face_deps["torch"] and face_deps["cv2"] else "missing_core",
            "details": json.dumps(face_deps, ensure_ascii=False),
        },
        {
            "area": "Module 3",
            "item": "subprocess CodeFormer",
            "status": "activated"
            if subprocess_codeformer.get("pipeline_has_codeformer_applied_true")
            else ("ready" if subprocess_codeformer.get("standalone_smoke_passed") else "not_ready"),
            "details": json.dumps(
                {
                    "conda_available": subprocess_codeformer.get("conda_available"),
                    "env_exists": subprocess_codeformer.get("codeformer_env_exists"),
                    "repo_exists": subprocess_codeformer.get("codeformer_repo_exists"),
                    "weights_exist": subprocess_codeformer.get("weights_exist"),
                    "standalone_smoke_passed": subprocess_codeformer.get("standalone_smoke_passed"),
                    "pipeline_has_codeformer_applied_true": subprocess_codeformer.get("pipeline_has_codeformer_applied_true"),
                },
                ensure_ascii=False,
            ),
        },
        {
            "area": "Module 3",
            "item": "codeformer output",
            "status": "exists" if codeformer_output_exists else "missing",
            "details": "Tìm `outputs/**/face_module/face_output.png`.",
        },
        {
            "area": "Module 3",
            "item": "codeformer fidelity sweep",
            "status": "exists" if CODEFORMER_FIDELITY_SWEEP_DIR.exists() else "missing",
            "details": f"{CODEFORMER_FIDELITY_SWEEP_DIR}. Warning: quality requires human visual review.",
        },
        {
            "area": "Demo",
            "item": "app_gradio import",
            "status": "pass" if app_import_pass() else "fail",
            "details": "Import không yêu cầu Gradio phải được cài.",
        },
        {
            "area": "Demo",
            "item": "run_restoration_pipeline.py",
            "status": "exists" if (PROJECT_ROOT / "scripts" / "run_restoration_pipeline.py").exists() else "missing",
            "details": "CLI cho UI và batch suite.",
        },
        {
            "area": "Git",
            "item": "short status",
            "status": "clean" if not git_status else "dirty",
            "details": git_status.replace("\n", " | ") if git_status else "(clean)",
        },
    ]

    write_csv(OUTPUT_DIR / "module_status.csv", rows)
    write_markdown(OUTPUT_DIR / "ACCELERATION_STATUS.md", rows, git_status, metric_files)
    print(f"markdown: {OUTPUT_DIR / 'ACCELERATION_STATUS.md'}")
    print(f"csv: {OUTPUT_DIR / 'module_status.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
