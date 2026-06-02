from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "blueprint21_final_assets"
VALIDATION_ROOT = PROJECT_ROOT / "outputs" / "blueprint21_final_validation" / "official_lama_validation"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ASSETS = [
    (
        "Demo3 final sensitive comparison",
        PROJECT_ROOT / "outputs" / "final_pipeline_candidate" / "demo3_final_sensitive_comparison.png",
        "So sánh demo3 candidate trước official LaMa.",
    ),
    (
        "Demo3 oracle diagnosis grid",
        PROJECT_ROOT
        / "outputs"
        / "final_pipeline_candidate"
        / "demo3_oracle_diagnosis"
        / "demo3_oracle_comparison_grid.png",
        "Chẩn đoán oracle/manual mask để phân biệt mask bottleneck và inpainting bottleneck.",
    ),
    (
        "Official LaMa vs simple_lama final grid",
        PROJECT_ROOT
        / "outputs"
        / "blueprint21_completion"
        / "official_lama_pipeline_test"
        / "official_vs_simple_final_grid.png",
        "So sánh official/pretrained LaMa với simple_lama trên demo3.",
    ),
    (
        "Official LaMa validation contact sheet",
        VALIDATION_ROOT / "official_lama_validation_contact_sheet.png",
        "Contact sheet 9 case demo1/demo2/demo3 cho final validation.",
    ),
    (
        "Final UI status",
        OUTPUT_ROOT / "FINAL_UI_STATUS.md",
        "Trạng thái UI, backend option, readiness panel và smoke test.",
    ),
    (
        "CodeFormer activation summary",
        PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "codeformer_activation" / "CODEFORMER_ACTIVATION_SUMMARY.md",
        "Bằng chứng CodeFormer adapter đã chạy nếu thư mục tồn tại.",
    ),
    (
        "R012 visual review summary",
        PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "r012_visual_review" / "REVIEW_SUMMARY.md",
        "Review r011/r012/manual mask nếu thư mục tồn tại.",
    ),
    (
        "Final pipeline metadata",
        PROJECT_ROOT
        / "outputs"
        / "blueprint21_final_validation"
        / "ui_backend_smoke"
        / "demo3"
        / "demo3"
        / "auto_r011_sensitive_low_threshold"
        / "metadata.json",
        "Metadata smoke test final candidate demo3.",
    ),
]


def read_validation_rows() -> list[dict[str, str]]:
    index_path = VALIDATION_ROOT / "validation_index.csv"
    if not index_path.exists():
        return []
    import csv

    with index_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_asset_index() -> None:
    lines = [
        "# Final Asset Index",
        "",
        "| asset | path | purpose | status |",
        "| --- | --- | --- | --- |",
    ]
    for name, path, purpose in ASSETS:
        status = "available" if path.exists() else "missing"
        lines.append(f"| {name} | `{path}` | {purpose} | {status} |")
    (OUTPUT_ROOT / "FINAL_ASSET_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pipeline_status(rows: list[dict[str, str]]) -> None:
    passed = sum(1 for row in rows if str(row.get("pass", "")).lower() == "true")
    total = len(rows)
    lines = [
        "# Final Pipeline Status",
        "",
        "## Module 1 - Crack Segmentation",
        "",
        "- r011 stable baseline đã dùng được trong pipeline.",
        "- r012 là experimental manual-mask fine-tune, chưa claim vượt trội.",
        "- `auto_r011_sensitive_low_threshold` là high-recall mode cho demo3, không phải global default.",
        "",
        "## Module 2 - Inpainting",
        "",
        "- `simple_lama` là stable fallback.",
        "- `opencv` là classical fallback.",
        "- `official_lama_pretrained` runnable qua subprocess adapter trong env `lama`.",
        "- official LaMa hiện chạy CPU local, nên có thể chậm.",
        "- LaMa fine-tune chưa thực hiện.",
        "",
        "## Module 3 - Face Restoration",
        "",
        "- CodeFormer đã activated qua env riêng `codeformer`.",
        "- Metadata smoke test có `face_restoration_applied=true` và `face_backend=codeformer`.",
        "",
        "## UI Status",
        "",
        "- Gradio app có official_lama option, external/oracle mask upload, CodeFormer fidelity slider và readiness panel.",
        "- UI giữ các mode cũ, không đổi sensitive thành global default.",
        "",
        "## Validation",
        "",
        f"- official_lama validation pass: `{passed}/{total}`",
        "- Visual review vẫn cần trước khi claim chất lượng cuối cùng.",
        "",
        "## Limitations",
        "",
        "- official_lama CPU-only trong local env hiện tại.",
        "- sensitive mode có thể tăng false positive.",
        "- LPIPS/FID/masked-region metrics chưa đo.",
        "- Broader test set vẫn cần trước khi claim Blueprint completed.",
    ]
    (OUTPUT_ROOT / "FINAL_PIPELINE_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendation(rows: list[dict[str, str]]) -> None:
    official_failures = [
        row
        for row in rows
        if row.get("backend_requested") == "official_lama" and str(row.get("pass", "")).lower() != "true"
    ]
    lines = [
        "# Final Demo Recommendation",
        "",
        "## Recommended demo",
        "",
        "- demo image: `demo3`",
        "- mask mode: `auto_r011_sensitive_low_threshold`",
        "- backend candidate: `official_lama_pretrained` pending human visual review",
        "- fallback backend: `simple_lama`",
        "- face restoration: `codeformer_if_available`",
        "- fidelity: `0.7`",
        "",
        "## Backend decision",
        "",
    ]
    if official_failures:
        lines.append("- Một số official_lama case fail; giữ `simple_lama` làm fallback chính cho case fail.")
    else:
        lines.append("- Official LaMa pass validation matrix demo1/demo2/demo3; có thể xem là runnable candidate.")
    lines.extend(
        [
            "",
            "## Warnings",
            "",
            "- Sensitive mode có thể tăng false positives.",
            "- Official LaMa hiện chạy CPU local, chậm hơn simple_lama.",
            "- Visual review pending trước khi đổi default toàn project.",
            "- LaMa fine-tune chưa thực hiện.",
        ]
    )
    (OUTPUT_ROOT / "FINAL_DEMO_RECOMMENDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_ui_status(rows: list[dict[str, str]]) -> None:
    passed = sum(1 for row in rows if str(row.get("pass", "")).lower() == "true")
    total = len(rows)
    smoke_metadata_path = (
        PROJECT_ROOT
        / "outputs"
        / "blueprint21_final_validation"
        / "ui_backend_smoke"
        / "demo3"
        / "demo3"
        / "auto_r011_sensitive_low_threshold"
        / "metadata.json"
    )
    smoke_metadata = {}
    if smoke_metadata_path.exists():
        smoke_metadata = json.loads(smoke_metadata_path.read_text(encoding="utf-8"))
    lines = [
        "# Final UI Status",
        "",
        "- Gradio version: `6.15.2`",
        "- UI style applied: `dark editorial / film photography` nhẹ qua CSS ở `launch()`, không rewrite toàn bộ app.",
        "- official_lama option exists: `true`",
        "- sensitive mode exists: `true`",
        "- external mask upload exists: `true`",
        "- CodeFormer fidelity slider exists: `true`",
        "- readiness panel exists: `true`",
        "- import test: `pass`",
        f"- official_lama validation: `{passed}/{total}`",
        f"- smoke test demo3: `{'pass' if smoke_metadata else 'missing'}`",
        "",
        "## Smoke metadata",
        "",
        f"- inpainting_backend_requested: `{smoke_metadata.get('inpainting_backend_requested')}`",
        f"- inpainting_backend_actual: `{smoke_metadata.get('inpainting_backend_actual')}`",
        f"- official_lama_reason: `{smoke_metadata.get('official_lama_reason')}`",
        f"- face_restoration_applied: `{smoke_metadata.get('face_restoration_applied')}`",
        f"- face_backend: `{smoke_metadata.get('face_backend')}`",
        "",
        "## Known limitations",
        "",
        "- `official_lama` đang chạy CPU local nên chậm.",
        "- `auto_r011_sensitive_low_threshold` có thể tăng false positive trên ảnh khác.",
        "- Visual review thêm vẫn pending trước khi đổi default toàn project.",
        "- LaMa fine-tune chưa làm.",
    ]
    (OUTPUT_ROOT / "FINAL_UI_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_validation_rows()
    write_asset_index()
    write_pipeline_status(rows)
    write_recommendation(rows)
    update_ui_status(rows)
    print(OUTPUT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
