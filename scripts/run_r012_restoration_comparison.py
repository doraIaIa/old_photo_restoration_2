from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

MODES = [
    "auto_r011",
    "auto_r011_refined",
    "auto_r011_union_refined",
    "auto_r012",
    "auto_r012_refined",
    "auto_r012_union_refined",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy restoration comparison r011 vs r012 trên demo images.")
    parser.add_argument("--demo-dir", default="data/demo_inputs/real_manual_3")
    parser.add_argument("--output-root", default="outputs/blueprint21_acceleration/r012_restoration_comparison")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--modes", default=",".join(MODES))
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def find_images(demo_dir: Path) -> list[Path]:
    return sorted(path for path in demo_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def run_pipeline(image_path: Path, mode: str, output_root: Path, device: str, backend: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts\\run_restoration_pipeline.py",
        "--image",
        str(image_path),
        "--mode",
        mode,
        "--output-dir",
        str(output_root),
        "--backend",
        backend,
        "--device",
        device,
        "--face-mode",
        "off",
    ]
    print(f"\n[{image_path.name} {mode} {backend}] {' '.join(command)}")
    return subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)


def load_metadata(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    args = parse_args()
    demo_dir = resolve_path(args.demo_dir)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not demo_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy demo-dir: {demo_dir}")

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for image_path in find_images(demo_dir):
        for mode in modes:
            backend_used = "simple_lama"
            result = run_pipeline(image_path, mode, output_root, args.device, backend_used)
            if result.stdout:
                print(result.stdout.rstrip())
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            if result.returncode != 0:
                print(f"simple_lama failed for {image_path.name} {mode}; retry opencv", file=sys.stderr)
                backend_used = "opencv"
                result = run_pipeline(image_path, mode, output_root, args.device, backend_used)
                if result.stdout:
                    print(result.stdout.rstrip())
                if result.stderr:
                    print(result.stderr.rstrip(), file=sys.stderr)
            output_dir = output_root / image_path.stem / mode
            metadata = load_metadata(output_dir)
            rows.append(
                {
                    "image": str(image_path),
                    "mode": mode,
                    "backend_requested": backend_used,
                    "returncode": result.returncode,
                    "checkpoint_used": metadata.get("checkpoint_used", ""),
                    "mask_source": metadata.get("mask_source", ""),
                    "mask_refine": metadata.get("mask_refine", ""),
                    "final_mask_ratio": metadata.get("final_mask_ratio", ""),
                    "inpainting_backend_actual": metadata.get("inpainting_backend_actual", metadata.get("actual_backend", "")),
                    "output_dir": str(output_dir),
                    "comparison_grid": str(output_dir / "comparison_grid.png") if (output_dir / "comparison_grid.png").exists() else "",
                    "restored_final": str(output_dir / "restored_final.png") if (output_dir / "restored_final.png").exists() else "",
                    "metadata_json": str(output_dir / "metadata.json") if (output_dir / "metadata.json").exists() else "",
                }
            )
            if result.returncode != 0:
                raise RuntimeError(f"Pipeline failed after opencv fallback: {image_path.name} {mode}")

    index_path = output_root / "comparison_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0].keys()) if rows else ["image", "mode"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_lines = [
        "# r012 Restoration Comparison",
        "",
        f"- Demo dir: `{demo_dir}`",
        f"- Output root: `{output_root}`",
        f"- Cases: {len(rows)}",
        "",
        "| image | mode | backend | final_mask_ratio | comparison_grid |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        summary_lines.append(
            f"| {Path(row['image']).name} | {row['mode']} | {row['inpainting_backend_actual']} | {row['final_mask_ratio']} | `{row['comparison_grid']}` |"
        )
    (output_root / "RESTORATION_COMPARISON_SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"comparison_index: {index_path}")
    print(f"summary: {output_root / 'RESTORATION_COMPARISON_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
