from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "codeformer_fidelity_sweep"
INDEX_CSV = SWEEP_ROOT / "fidelity_sweep_index.csv"
SUMMARY_MD = SWEEP_ROOT / "FIDELITY_SWEEP_SUMMARY.md"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not SWEEP_ROOT.exists():
        return rows
    for metadata_path in sorted(SWEEP_ROOT.glob("demo*/fidelity_*/demo*/auto_r011_union_refined/metadata.json")):
        metadata = load_json(metadata_path)
        case_dir = metadata_path.parent
        fidelity_dir = case_dir.parent.parent.name
        fidelity_text = fidelity_dir.replace("fidelity_", "").replace("p", ".")
        try:
            fidelity = float(fidelity_text)
        except ValueError:
            fidelity = metadata.get("codeformer_fidelity")
        rows.append(
            {
                "demo_id": case_dir.parent.name,
                "fidelity": fidelity,
                "ok": bool(metadata.get("face_restoration_applied") is True and metadata.get("face_backend") == "codeformer"),
                "face_restoration_applied": metadata.get("face_restoration_applied"),
                "face_backend": metadata.get("face_backend"),
                "face_reason": metadata.get("face_reason"),
                "inpainting_backend_actual": metadata.get("inpainting_backend_actual"),
                "codeformer_fidelity": metadata.get("codeformer_fidelity"),
                "restored_before_face_exists": (case_dir / "restored_before_face.png").exists(),
                "restored_final_exists": (case_dir / "restored_final.png").exists(),
                "comparison_grid_exists": (case_dir / "comparison_grid.png").exists(),
                "metadata_path": str(metadata_path),
            }
        )
    return rows


def main() -> int:
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    fields = [
        "demo_id",
        "fidelity",
        "ok",
        "face_restoration_applied",
        "face_backend",
        "face_reason",
        "inpainting_backend_actual",
        "codeformer_fidelity",
        "restored_before_face_exists",
        "restored_final_exists",
        "comparison_grid_exists",
        "metadata_path",
    ]
    with INDEX_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    passed = sum(1 for row in rows if row["ok"])
    failed = total - passed
    lines = [
        "# CodeFormer Fidelity Sweep Summary",
        "",
        f"- Sweep root: `{SWEEP_ROOT}`",
        f"- Cases found: `{total}`",
        f"- Cases passed: `{passed}`",
        f"- Cases failed: `{failed}`",
        "- Metadata evidence: pass means `face_restoration_applied=true` and `face_backend=codeformer`.",
        "- Quality conclusion: needs human visual review; this summary does not claim visual improvement.",
        "",
        "## Default guidance",
        "",
        "- Use `0.7` as the conservative default if visual review wants a balanced setting.",
        "- Use `1.0` if preserving identity is more important.",
        "- Use `0.5` if stronger face restoration is preferred after visual review.",
        "",
        "| demo_id | fidelity | ok | face_backend | inpainting_backend | metadata |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['demo_id']} | {row['fidelity']} | {row['ok']} | {row['face_backend']} | "
            f"{row['inpainting_backend_actual']} | `{row['metadata_path']}` |"
        )
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"summary: {SUMMARY_MD}")
    print(f"index: {INDEX_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
