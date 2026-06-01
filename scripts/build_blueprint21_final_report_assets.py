from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


METRIC_ROWS = [
    {
        "experiment": "r009 synthetic-only",
        "split": "real test",
        "iou": "0.002222",
        "f1": "0.004434",
        "precision": "0.083464",
        "recall": "0.002277",
        "note": "Baseline segmentation trước fine-tune real-domain.",
    },
    {
        "experiment": "r010 real-ft",
        "split": "real test",
        "iou": "0.292728",
        "f1": "0.452884",
        "precision": "0.509123",
        "recall": "0.407834",
        "note": "Fine-tune real-domain cải thiện mạnh so với r009.",
    },
    {
        "experiment": "r011 repair-ft",
        "split": "repair_v1 test",
        "iou": "0.447877",
        "f1": "0.618667",
        "precision": "0.613738",
        "recall": "0.623676",
        "note": "Fine-tune trên repair-mask v1.",
    },
    {
        "experiment": "r011 repair-ft",
        "split": "thin GT test",
        "iou": "0.371838",
        "f1": "0.542102",
        "precision": "0.493531",
        "recall": "0.601276",
        "note": "Diagnosis so với thin/manual-style GT.",
    },
]


MODULE_ROWS = [
    {
        "module": "Module 1",
        "name": "Segmentation r011",
        "status": "implemented",
        "output": "final_mask.png, dl_prob_mask.png",
        "note": "Sinh mask crack/scratch/fold/tear tự động.",
    },
    {
        "module": "Module 1.5",
        "name": "Mask refinement",
        "status": "implemented",
        "output": "final_mask_before_refine.png, final_mask_refined.png",
        "note": "Refine bảo thủ để giảm nhiễu mask.",
    },
    {
        "module": "Module 2",
        "name": "Inpainting backend",
        "status": "implemented",
        "output": "restored_before_face.png",
        "note": "Dùng backend auto/simple_lama/opencv qua run_demo.py.",
    },
    {
        "module": "Module 3",
        "name": "Face restoration",
        "status": "dependency-gated",
        "output": "restored_final.png, face_module/face_metadata.json",
        "note": "Chỉ áp dụng khi CodeFormer/GFPGAN và adapter inference ổn định sẵn sàng.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Blueprint 2.1 final report assets.")
    parser.add_argument("--r010-summary", required=True)
    parser.add_argument("--r011-summary", required=True)
    parser.add_argument("--final-demo-root", required=True)
    parser.add_argument("--blueprint-output", required=True)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_demo_asset_rows(final_demo_root: Path) -> list[dict[str, str]]:
    rows = []
    for row in read_csv(final_demo_root / "demo_index.csv"):
        rows.append(
            {
                "demo_id": row.get("demo_id", ""),
                "mode": row.get("mode", ""),
                "final_mask_ratio": row.get("final_mask_ratio", ""),
                "face_mode": row.get("face_mode", ""),
                "face_reason": row.get("face_reason", ""),
                "comparison_grid": row.get("comparison_grid", ""),
                "restored_before_face": row.get("restored_before_face", ""),
                "restored_final": row.get("restored_final", ""),
                "final_mask": row.get("final_mask", ""),
                "overlay_final": row.get("overlay_final", ""),
                "metadata_json": row.get("metadata_json", ""),
            }
        )
    return rows


def write_summary(path: Path, r010_summary: Path, r011_summary: Path, final_demo_root: Path, demo_rows: list[dict[str, str]]) -> None:
    face_rows = [row for row in demo_rows if row["mode"].endswith("_face_auto")]
    face_reasons = sorted({row["face_reason"] for row in face_rows if row["face_reason"]})
    lines = [
        "# Blueprint 2.1 Final Summary",
        "",
        "## Dataset và split",
        "",
        "- Real dataset: `F:\\deeplearning\\old_photo_mask\\old_photo_pairs_10_hq`, 60 pairs.",
        "- Split: train 42, val 9, test 9.",
        "- Không copy dataset, checkpoint hoặc output image vào Git.",
        "",
        "## Metrics chính",
        "",
        "| Experiment | Split | IoU | F1 | Precision | Recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in METRIC_ROWS:
        lines.append(
            f"| {row['experiment']} | {row['split']} | {row['iou']} | {row['f1']} | {row['precision']} | {row['recall']} |"
        )
    lines.extend(
        [
            "",
            "## Demo suite",
            "",
            f"- Final demo root: `{final_demo_root}`",
            f"- Số dòng demo index: {len(demo_rows)}",
            f"- Face-auto variants: {len(face_rows)}",
            f"- Face reasons ghi nhận: `{', '.join(face_reasons) if face_reasons else 'none'}`",
            "",
            "## Kết luận",
            "",
            "- Bottleneck chính của pipeline là mask generation.",
            "- Fine-tune real-domain cải thiện mạnh so với r009 synthetic-only.",
            "- `manual_upper_bound`/external mask chỉ là ceiling hoặc diagnosis, không phải automatic pipeline.",
            "- `auto_r011_union_refined_face_auto` chỉ nên được trình bày là face-auto variant nếu visual review xác nhận hữu ích.",
            "",
            "## Nguồn liên quan",
            "",
            f"- r010 summary: `{r010_summary}`",
            f"- r011 summary: `{r011_summary}`",
            f"- final demo summary: `{final_demo_root / 'SUMMARY.md'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_limitations(path: Path) -> None:
    lines = [
        "# Limitations And Next Steps",
        "",
        "## Limitations",
        "",
        "- Mask generation vẫn là bottleneck chính; false negative làm crack còn sót, false positive làm inpainting phá vùng ảnh sạch.",
        "- Module 3 face restoration hiện là dependency-gated wrapper; chưa bật inference CodeFormer/GFPGAN khi thiếu adapter ổn định.",
        "- External/manual mask chỉ dùng để diagnosis hoặc upper bound, không đại diện cho automatic pipeline.",
        "",
        "## Next steps",
        "",
        "- Review visual các mode `auto_r011`, `auto_r011_union`, `auto_r011_refined`, `auto_r011_union_refined` và `_face_auto`.",
        "- Nếu chọn face restoration, cấu hình adapter CodeFormer/GFPGAN có version pin, input/output contract và smoke test riêng.",
        "- Mở rộng manual repair-mask nếu cần r012, nhưng không train trong Blueprint 2.1 hiện tại.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    r010_summary = resolve_path(args.r010_summary)
    r011_summary = resolve_path(args.r011_summary)
    final_demo_root = resolve_path(args.final_demo_root)
    blueprint_output = resolve_path(args.blueprint_output)
    blueprint_output.mkdir(parents=True, exist_ok=True)

    demo_rows = build_demo_asset_rows(final_demo_root)
    write_csv(
        blueprint_output / "ablation_table.csv",
        METRIC_ROWS,
        ["experiment", "split", "iou", "f1", "precision", "recall", "note"],
    )
    write_csv(
        blueprint_output / "module_status.csv",
        MODULE_ROWS,
        ["module", "name", "status", "output", "note"],
    )
    write_csv(
        blueprint_output / "demo_asset_index.csv",
        demo_rows,
        [
            "demo_id",
            "mode",
            "final_mask_ratio",
            "face_mode",
            "face_reason",
            "comparison_grid",
            "restored_before_face",
            "restored_final",
            "final_mask",
            "overlay_final",
            "metadata_json",
        ],
    )
    write_summary(blueprint_output / "BLUEPRINT21_FINAL_SUMMARY.md", r010_summary, r011_summary, final_demo_root, demo_rows)
    write_limitations(blueprint_output / "limitation_and_next_steps.md")

    print(f"summary: {blueprint_output / 'BLUEPRINT21_FINAL_SUMMARY.md'}")
    print(f"ablation_table: {blueprint_output / 'ablation_table.csv'}")
    print(f"module_status: {blueprint_output / 'module_status.csv'}")
    print(f"demo_asset_index: {blueprint_output / 'demo_asset_index.csv'}")
    print(f"limitations: {blueprint_output / 'limitation_and_next_steps.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
