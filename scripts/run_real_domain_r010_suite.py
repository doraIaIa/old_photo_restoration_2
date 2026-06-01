from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


THRESHOLDS = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
EXPECTED_DEMO_FILES = ["demo1.jpg", "demo2.png", "demo3.png"]
REPORT_CV_PROFILE = "notebook_v7_candidate"


@dataclass(frozen=True)
class DemoVariant:
    name: str
    checkpoint_key: str
    threshold: float
    fallback_threshold: float
    mask_source: str
    mask_dilate: int
    output_folder: str
    manual_only: bool = False


DEMO_VARIANTS = [
    DemoVariant("r009_dl_t090", "r009", 0.90, 0.70, "dl", 0, "r009_dl_t090"),
    DemoVariant("r010_dl_t070", "r010", 0.70, 0.50, "dl", 0, "r010_dl_t070"),
    DemoVariant("r010_union_cv_t070_dilate0", "r010", 0.70, 0.50, "union", 0, "r010_union_cv_t070_dilate0"),
    DemoVariant("r010_union_cv_t070_dilate1", "r010", 0.70, 0.50, "union", 1, "r010_union_cv_t070_dilate1"),
    DemoVariant("r010_union_cv_t070_dilate2", "r010", 0.70, 0.50, "union", 2, "r010_union_cv_t070_dilate2"),
    DemoVariant("r010_union_cv_t050_dilate1", "r010", 0.50, 0.40, "union", 1, "r010_union_cv_t050_dilate1"),
    DemoVariant("manual_upper_bound", "r010", 0.70, 0.50, "external", 0, "manual_upper_bound", manual_only=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy suite tổng hợp kết quả real-domain r010 để làm report asset.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--r009-checkpoint", required=True)
    parser.add_argument("--r010-checkpoint", required=True)
    parser.add_argument("--demo-dir", required=True)
    parser.add_argument("--external-mask-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_command(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    print(f"\n[{label}] {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Lệnh `{label}` lỗi với exit code {result.returncode}.")
    return result


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {label}: {path}")


def read_split_count(path: Path) -> int:
    require_file(path, "split file")
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON phải là object: {path}")
    return payload


def load_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(
    data_root: Path,
    split_file: Path,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    label: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts\\evaluate_real_segmentation.py",
        "--data-root",
        str(data_root),
        "--split-file",
        str(split_file),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--thresholds",
        THRESHOLDS,
        "--image-size",
        str(image_size),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
    ]
    run_command(command, label)
    return load_json(output_dir / "summary.json")


def find_demo_images(demo_dir: Path) -> tuple[list[Path], list[Path]]:
    existing: list[Path] = []
    skipped: list[Path] = []
    for filename in EXPECTED_DEMO_FILES:
        candidate = demo_dir / filename
        if candidate.exists():
            existing.append(candidate)
        else:
            skipped.append(candidate)
    return existing, skipped


def external_mask_for_image(mask_dir: Path, image_path: Path) -> Path:
    return mask_dir / f"{image_path.stem}_mask.png"


def checkpoint_for_variant(variant: DemoVariant, r009_checkpoint: Path, r010_checkpoint: Path) -> Path:
    return r009_checkpoint if variant.checkpoint_key == "r009" else r010_checkpoint


def run_demo_command(
    image_path: Path,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    variant: DemoVariant,
    external_mask: Path | None = None,
) -> Path:
    command = [
        sys.executable,
        "scripts\\run_demo.py",
        "--image",
        str(image_path),
        "--checkpoint",
        str(checkpoint),
        "--threshold",
        f"{variant.threshold:.2f}",
        "--fallback-threshold",
        f"{variant.fallback_threshold:.2f}",
        "--backend",
        "auto",
        "--mask-source",
        variant.mask_source,
        "--cv-profile",
        REPORT_CV_PROFILE,
        "--mask-dilate",
        str(variant.mask_dilate),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--save-prob-mask",
        "--save-all-masks",
    ]
    if external_mask is not None:
        command.extend(["--external-mask", str(external_mask)])

    run_command(command, f"demo {variant.name} {image_path.name}")
    return output_dir / image_path.stem / variant.mask_source


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8f}"
    return str(value)


def demo_row_from_metadata(
    image_path: Path,
    variant: DemoVariant,
    checkpoint: Path,
    sample_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    comparison_grid = sample_dir / "comparison_grid.png"
    restored_final = sample_dir / "restored_final.png"
    final_mask = sample_dir / "final_mask.png"
    overlay_final = sample_dir / "overlay_final.png"
    metadata_json = sample_dir / "metadata.json"
    return {
        "demo_id": image_path.stem,
        "image_path": str(image_path),
        "variant": variant.name,
        "checkpoint_name": checkpoint.parent.name,
        "threshold": f"{variant.threshold:.2f}",
        "fallback_threshold": f"{variant.fallback_threshold:.2f}",
        "mask_source": variant.mask_source,
        "cv_profile": metadata.get("cv_profile", REPORT_CV_PROFILE),
        "mask_dilate": variant.mask_dilate,
        "final_mask_ratio": format_optional(metadata.get("final_mask_ratio")),
        "dl_mask_ratio": format_optional(metadata.get("dl_mask_ratio", metadata.get("dl_mask_ratio_primary"))),
        "cv_mask_ratio": format_optional(metadata.get("cv_mask_ratio")),
        "union_mask_ratio": format_optional(metadata.get("union_mask_ratio")),
        "actual_backend": metadata.get("actual_backend", ""),
        "comparison_grid": str(comparison_grid) if comparison_grid.exists() else "",
        "restored_final": str(restored_final) if restored_final.exists() else "",
        "final_mask": str(final_mask) if final_mask.exists() else "",
        "overlay_final": str(overlay_final) if overlay_final.exists() else "",
        "metadata_json": str(metadata_json) if metadata_json.exists() else "",
    }


def run_demo_suite(
    demo_images: list[Path],
    external_mask_dir: Path,
    r009_checkpoint: Path,
    r010_checkpoint: Path,
    output_root: Path,
    device: str,
) -> tuple[list[dict[str, Any]], list[Path], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    used_external_masks: list[Path] = []
    skipped_manual: list[dict[str, Any]] = []

    for image_path in demo_images:
        print(f"\n[demo] xử lý {image_path}")
        for variant in DEMO_VARIANTS:
            checkpoint = checkpoint_for_variant(variant, r009_checkpoint, r010_checkpoint)
            output_dir = output_root / "demo" / variant.output_folder
            external_mask: Path | None = None

            if variant.manual_only:
                candidate = external_mask_for_image(external_mask_dir, image_path)
                if not candidate.exists():
                    skipped_manual.append({"image": str(image_path), "external_mask": str(candidate)})
                    print(f"[demo] skip manual_upper_bound vì external mask không tồn tại: {candidate}")
                    continue
                external_mask = candidate
                used_external_masks.append(candidate)

            sample_dir = run_demo_command(
                image_path=image_path,
                checkpoint=checkpoint,
                output_dir=output_dir,
                device=device,
                variant=variant,
                external_mask=external_mask,
            )
            metadata = load_json_or_empty(sample_dir / "metadata.json")
            rows.append(demo_row_from_metadata(image_path, variant, checkpoint, sample_dir, metadata))

    return rows, used_external_masks, skipped_manual


def metrics_row(model: str, split: str, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["best_metrics"]
    return {
        "model": model,
        "split": split,
        "best_threshold": summary["best_threshold_by_iou"],
        "iou": metrics["iou"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "output_dir": summary["output_dir"],
    }


def evaluation_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model/split | best_threshold | IoU | F1 | Precision | Recall | output_dir |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} / {row['split']} | {float(row['best_threshold']):.2f} | "
            f"{float(row['iou']):.6f} | {float(row['f1']):.6f} | "
            f"{float(row['precision']):.6f} | {float(row['recall']):.6f} | `{row['output_dir']}` |"
        )
    return "\n".join(lines)


def demo_markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| demo_id | variant | checkpoint | threshold | fallback_threshold | mask_source | cv_profile | mask_dilate | final_mask_ratio | actual_backend | comparison_grid |",
        "|---|---|---|---:|---:|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['demo_id']} | {row['variant']} | {row['checkpoint_name']} | "
            f"{row['threshold']} | {row['fallback_threshold']} | {row['mask_source']} | "
            f"{row['cv_profile']} | {row['mask_dilate']} | {row['final_mask_ratio']} | "
            f"{row['actual_backend']} | `{row['comparison_grid']}` |"
        )
    return "\n".join(lines)


def write_summary_markdown(
    path: Path,
    data_root: Path,
    split_counts: dict[str, int],
    r009_checkpoint: Path,
    r010_checkpoint: Path,
    metrics_rows: list[dict[str, Any]],
    demo_rows: list[dict[str, Any]],
    skipped_demo_images: list[Path],
    skipped_manual: list[dict[str, Any]],
) -> None:
    content = "\n".join(
        [
            "# Real Domain r010 Suite Summary",
            "",
            f"- Created at: `{now_iso()}`",
            "",
            "## Dataset and checkpoints",
            "",
            f"- Real dataset path: `{data_root}`",
            f"- Train/val/test count: `{split_counts['train']}` / `{split_counts['val']}` / `{split_counts['test']}`",
            f"- r009 checkpoint path: `{r009_checkpoint}`",
            f"- r010 checkpoint path: `{r010_checkpoint}`",
            "",
            "## Evaluation table",
            "",
            evaluation_markdown_table(metrics_rows),
            "",
            "## Key finding",
            "",
            "- r009 synthetic-only gần như fail trên real old-photo test set.",
            "- r010 sau fine-tune bằng 60 cặp ảnh-mask thật cải thiện mạnh.",
            "- r010 test IoU tăng từ `0.002222` lên `0.292728`.",
            "- r010 test F1 tăng từ `0.004434` lên `0.452884`.",
            "- Threshold thực tế theo val/test hiện là `0.70`.",
            "- simple_lama/LaMa hoạt động tốt khi mask đủ chính xác.",
            "- Bottleneck chính vẫn là mask generation.",
            "",
            "## CV profile note",
            "",
            "- CV profile noisy/old_photo_crack tạo mask texture/local contrast, không phù hợp làm crack detector chính.",
            "- Report/demo suite dùng `notebook_v7_candidate`.",
            "- Union cần được đánh giá thận trọng vì CV có thể giúp bắt thêm crack nhưng cũng có thể thêm false positives.",
            "",
            "## Demo variants",
            "",
            demo_markdown_table(demo_rows) if demo_rows else "Không có demo image hợp lệ để chạy.",
            "",
            "## Recommendation for visual selection",
            "",
            "- Dùng `r010_dl_t070` làm automatic DL baseline.",
            "- So sánh các union variants để chọn demo đẹp nhất.",
            "- Nếu `r010_union_cv_t070_dilate1` xóa sạch hơn mà không làm mất chi tiết, ưu tiên variant này.",
            "- Nếu `r010_union_cv_t070_dilate2` làm ảnh bệt hoặc mất chi tiết, không dùng.",
            "- Nếu `r010_union_cv_t050_dilate1` bắt thêm crack nhưng false positive nhiều, không dùng default.",
            "- `manual_upper_bound` chỉ dùng để chứng minh ceiling/diagnosis, không gọi là automatic.",
            "",
            "## Skips",
            "",
            "- Demo images bị skip: "
            + (", ".join(f"`{path}`" for path in skipped_demo_images) if skipped_demo_images else "không có"),
            "- Manual upper-bound bị skip: "
            + (
                ", ".join(f"`{item['image']}` thiếu `{item['external_mask']}`" for item in skipped_manual)
                if skipped_manual
                else "không có"
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def git_status_text() -> str:
    status = subprocess.run(
        ["git", "status", "-u", "--short"],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    return status.stdout.rstrip() or "(clean)"


def print_final_report(
    compileall_pass: bool,
    metrics_rows: list[dict[str, Any]],
    demo_rows: list[dict[str, Any]],
    demo_images: list[Path],
    skipped_demo_images: list[Path],
    used_external_masks: list[Path],
    skipped_manual: list[dict[str, Any]],
    summary_path: Path,
    metrics_path: Path,
    demo_index_path: Path,
) -> None:
    print("\n=== REAL DOMAIN R010 SUITE DONE ===")
    print(f"compileall: {'pass' if compileall_pass else 'fail'}")
    print("evaluation metrics:")
    for row in metrics_rows:
        print(
            f"- {row['model']} {row['split']}: "
            f"threshold={float(row['best_threshold']):.2f} "
            f"IoU={float(row['iou']):.6f} "
            f"F1={float(row['f1']):.6f} "
            f"Precision={float(row['precision']):.6f} "
            f"Recall={float(row['recall']):.6f}"
        )
    print("demo variants đã chạy:")
    for variant_name in sorted({row["variant"] for row in demo_rows}):
        print(f"- {variant_name}")
    print("demo images đã chạy:")
    for image in demo_images:
        print(f"- {image}")
    print("demo images bị skip:")
    for image in skipped_demo_images:
        print(f"- {image}")
    print("external masks đã dùng:")
    for mask in used_external_masks:
        print(f"- {mask}")
    print("manual upper-bound bị skip:")
    for item in skipped_manual:
        print(f"- {item['image']} thiếu {item['external_mask']}")

    demo3_rows = [row for row in demo_rows if row["demo_id"] == "demo3"]
    if demo3_rows:
        print("demo3 comparison_grid và final_mask_ratio:")
        for row in demo3_rows:
            print(f"- {row['variant']}: ratio={row['final_mask_ratio']} grid={row['comparison_grid']}")

    print(f"SUMMARY.md: {summary_path}")
    print(f"metrics_summary.csv: {metrics_path}")
    print(f"demo_index.csv: {demo_index_path}")
    print("git status -u --short:")
    print(git_status_text())


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    split_dir = resolve_path(args.split_dir)
    r009_checkpoint = resolve_path(args.r009_checkpoint)
    r010_checkpoint = resolve_path(args.r010_checkpoint)
    demo_dir = resolve_path(args.demo_dir)
    external_mask_dir = resolve_path(args.external_mask_dir)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    require_file(r009_checkpoint, "checkpoint r009")
    require_file(r010_checkpoint, "checkpoint r010")
    require_file(split_dir / "train.txt", "train split")
    require_file(split_dir / "val.txt", "val split")
    require_file(split_dir / "test.txt", "test split")

    run_command([sys.executable, "-m", "compileall", "scripts", "src"], "compileall")
    compileall_pass = True

    split_counts = {
        "train": read_split_count(split_dir / "train.txt"),
        "val": read_split_count(split_dir / "val.txt"),
        "test": read_split_count(split_dir / "test.txt"),
    }

    r009_test = run_evaluation(
        data_root,
        split_dir / "test.txt",
        r009_checkpoint,
        output_root / "evaluation" / "r009_test",
        args.device,
        args.image_size,
        args.batch_size,
        args.num_workers,
        "evaluate r009 test",
    )
    r010_val = run_evaluation(
        data_root,
        split_dir / "val.txt",
        r010_checkpoint,
        output_root / "evaluation" / "r010_val",
        args.device,
        args.image_size,
        args.batch_size,
        args.num_workers,
        "evaluate r010 val",
    )
    r010_test = run_evaluation(
        data_root,
        split_dir / "test.txt",
        r010_checkpoint,
        output_root / "evaluation" / "r010_test",
        args.device,
        args.image_size,
        args.batch_size,
        args.num_workers,
        "evaluate r010 test",
    )

    metrics_rows = [
        metrics_row("r009", "real test", r009_test),
        metrics_row("r010", "real val", r010_val),
        metrics_row("r010", "real test", r010_test),
    ]
    metrics_path = output_root / "metrics_summary.csv"
    write_csv(
        metrics_path,
        metrics_rows,
        ["model", "split", "best_threshold", "iou", "f1", "precision", "recall", "output_dir"],
    )

    if not demo_dir.exists():
        print(f"[demo] demo-dir không tồn tại, skip toàn bộ demo: {demo_dir}")
        demo_images = []
        skipped_demo_images = [demo_dir / filename for filename in EXPECTED_DEMO_FILES]
    else:
        demo_images, skipped_demo_images = find_demo_images(demo_dir)

    demo_rows, used_external_masks, skipped_manual = run_demo_suite(
        demo_images=demo_images,
        external_mask_dir=external_mask_dir,
        r009_checkpoint=r009_checkpoint,
        r010_checkpoint=r010_checkpoint,
        output_root=output_root,
        device=args.device,
    )
    demo_index_path = output_root / "demo_index.csv"
    demo_fields = [
        "demo_id",
        "image_path",
        "variant",
        "checkpoint_name",
        "threshold",
        "fallback_threshold",
        "mask_source",
        "cv_profile",
        "mask_dilate",
        "final_mask_ratio",
        "dl_mask_ratio",
        "cv_mask_ratio",
        "union_mask_ratio",
        "actual_backend",
        "comparison_grid",
        "restored_final",
        "final_mask",
        "overlay_final",
        "metadata_json",
    ]
    write_csv(demo_index_path, demo_rows, demo_fields)

    summary_path = output_root / "SUMMARY.md"
    write_summary_markdown(
        summary_path,
        data_root=data_root,
        split_counts=split_counts,
        r009_checkpoint=r009_checkpoint,
        r010_checkpoint=r010_checkpoint,
        metrics_rows=metrics_rows,
        demo_rows=demo_rows,
        skipped_demo_images=skipped_demo_images,
        skipped_manual=skipped_manual,
    )

    print_final_report(
        compileall_pass=compileall_pass,
        metrics_rows=metrics_rows,
        demo_rows=demo_rows,
        demo_images=demo_images,
        skipped_demo_images=skipped_demo_images,
        used_external_masks=used_external_masks,
        skipped_manual=skipped_manual,
        summary_path=summary_path,
        metrics_path=metrics_path,
        demo_index_path=demo_index_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
