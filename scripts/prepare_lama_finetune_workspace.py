from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chuẩn bị workspace LaMa fine-tune không phá repo.")
    parser.add_argument("--output-dir", default="outputs/blueprint21_acceleration/lama_finetune_workspace")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    for subdir in ["configs", "data_links", "reports", "runs", "scripts"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    config = {
        "status": "template_only",
        "baseline": "simple_lama pretrained",
        "do_not_install_into_main_venv": True,
        "required_inputs": {
            "clean_images": "DIV2K / Flickr / existing clean source",
            "masks": "crack bank / repair masks / manual repair masks",
            "train_pairs": "degraded image + mask + clean target",
        },
        "completion_criteria": [
            "fine-tuned LaMa checkpoint exists",
            "inference output exists",
            "LPIPS/FID or masked-region LPIPS report exists",
            "pipeline metadata confirms fine_tuned_lama backend",
        ],
    }
    (output_dir / "configs" / "lama_finetune_template.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = [
        "# LaMa Fine-tune Workspace",
        "",
        "Workspace này chỉ là checklist/config template, không chứa dataset và không cài dependency.",
        "",
        "## Cần liên kết dữ liệu ngoài repo",
        "",
        "- `data_links/clean_images`: DIV2K/Flickr/clean source.",
        "- `data_links/masks`: crack bank hoặc repair masks.",
        "- `data_links/pairs`: degraded image + mask + clean target.",
        "",
        "## Baseline",
        "",
        "`simple_lama` pretrained là baseline hiện tại.",
        "",
        "## Điều kiện completed",
        "",
        "Fine-tuned LaMa chỉ được xem là completed khi có checkpoint, inference output và LPIPS/FID hoặc masked-region LPIPS.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"workspace: {output_dir}")
    print(f"config: {output_dir / 'configs' / 'lama_finetune_template.json'}")
    print(f"readme: {output_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
