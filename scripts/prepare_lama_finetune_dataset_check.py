from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "blueprint21_completion" / "lama_finetune_feasibility"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm kê dữ liệu khả thi cho fine-tune LaMa.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def list_files(root: Path, extensions: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def inspect_path(path: Path, extensions: set[str]) -> dict[str, Any]:
    files = list_files(path, extensions)
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "sample_files": [str(item) for item in files[:10]],
    }


def write_outputs(inventory: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# LaMa Fine-tune Feasibility",
        "",
        f"- feasible_now: `{inventory['feasible_now']}`",
        f"- clean images available: `{inventory['clean_total']}`",
        f"- mask files available: `{inventory['mask_total']}`",
        f"- synthetic pairs possible upper-bound: `{inventory['synthetic_pairs_possible_upper_bound']}`",
        "",
        "## Clean sources",
        "",
        "| path | exists | file_count |",
        "| --- | --- | ---: |",
    ]
    for item in inventory["clean_sources"]:
        lines.append(f"| `{item['path']}` | {item['exists']} | {item['file_count']} |")
    lines.extend(["", "## Mask bank", "", "| path | exists | file_count |", "| --- | --- | ---: |"])
    for item in inventory["mask_sources"]:
        lines.append(f"| `{item['path']}` | {item['exists']} | {item['file_count']} |")
    lines.extend(["", "## Missing requirements", ""])
    for item in inventory["missing_requirements"]:
        lines.append(f"- {item}")
    if not inventory["missing_requirements"]:
        lines.append("- Không phát hiện thiếu dữ liệu chính.")
    lines.extend(["", "## Next steps", ""])
    for item in inventory["next_steps"]:
        lines.append(f"- {item}")
    (output_dir / "LAMA_FINETUNE_FEASIBILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    clean_paths = [
        PROJECT_ROOT / "data" / "clean",
        PROJECT_ROOT / "data" / "raw_clean",
        PROJECT_ROOT / "data" / "DIV2K",
        Path(r"F:\deeplearning\datasets\DIV2K"),
    ]
    mask_paths = [
        Path(r"F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq\masks"),
        Path(r"F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq\masks_repair_manual"),
        Path(r"F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq\masks_repair_repair_v1"),
    ]
    clean_sources = [inspect_path(path, IMAGE_EXTENSIONS) for path in clean_paths]
    mask_sources = [inspect_path(path, MASK_EXTENSIONS) for path in mask_paths]
    clean_total = sum(item["file_count"] for item in clean_sources)
    mask_total = sum(item["file_count"] for item in mask_sources)
    missing: list[str] = []
    if clean_total == 0:
        missing.append("Chưa có nguồn ảnh clean đủ rõ để tạo synthetic inpainting pairs.")
    if mask_total == 0:
        missing.append("Chưa có mask bank crack/scratch để ghép synthetic LaMa pairs.")
    missing.append("Chưa xác nhận official LaMa pretrained inference chạy được trong env riêng.")
    missing.append("Chưa có recipe fine-tune LaMa đã smoke-test trên Windows/WSL.")
    feasible_now = False
    next_steps = [
        "Chạy official/pretrained LaMa inference trên demo3 trước khi fine-tune.",
        "Chuẩn hóa clean source và mask bank thành manifest train/val nhỏ.",
        "Chạy tiny overfit 1-2 ảnh chỉ sau khi env LaMa và checkpoint pretrained đã pass.",
    ]
    inventory = {
        "clean_sources": clean_sources,
        "mask_sources": mask_sources,
        "clean_total": clean_total,
        "mask_total": mask_total,
        "synthetic_pairs_possible_upper_bound": min(clean_total, mask_total),
        "feasible_now": feasible_now,
        "missing_requirements": missing,
        "next_steps": next_steps,
    }
    write_outputs(inventory, Path(args.output_dir))
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
