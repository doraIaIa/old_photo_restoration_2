from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "codeformer_activation"
FACE_STATUS_JSON = PROJECT_ROOT / "outputs" / "blueprint21_acceleration" / "face_dependency_status.json"
CODEFORMER_REPO = Path(r"F:\deeplearning\external_models\CodeFormer")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_probe(command: list[str], timeout_sec: int = 60) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_sec,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout_tail": "", "stderr_tail": str(exc)}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def blocked_reason(summary: dict[str, Any]) -> str:
    if summary["activated"]:
        return ""
    if not summary["codeformer_repo_exists"]:
        return "codeformer_repo_missing"
    if not summary["conda_available"]:
        return "conda_not_available"
    if not summary["codeformer_env_exists"]:
        return "codeformer_env_missing"
    if not summary["weights_exist"]:
        return "weights_missing"
    if not summary["standalone_smoke"]["ok"]:
        return "standalone_smoke_missing_or_failed"
    if not summary["pipeline_test"]["ok"]:
        return "pipeline_metadata_true_missing"
    return "unknown"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    face_status = load_json(FACE_STATUS_JSON)
    subprocess_status = face_status.get("subprocess_codeformer", {})
    conda_create_probe = run_probe(["conda", "run", "-n", "codeformer", "python", "--version"])
    standalone = {
        "ok": bool(subprocess_status.get("standalone_smoke_passed")),
        "outputs": subprocess_status.get("standalone_smoke_outputs", []),
        "reason": "" if subprocess_status.get("standalone_smoke_passed") else "not_run_or_no_output",
    }
    pipeline = {
        "ok": bool(subprocess_status.get("pipeline_has_codeformer_applied_true")),
        "metadata_tail": subprocess_status.get("pipeline_face_metadata_tail", []),
        "reason": "" if subprocess_status.get("pipeline_has_codeformer_applied_true") else "not_run_or_no_metadata_true",
    }
    summary = {
        "codeformer_env": "codeformer",
        "codeformer_repo": str(CODEFORMER_REPO),
        "codeformer_repo_exists": CODEFORMER_REPO.exists(),
        "conda_available": subprocess_status.get("conda_available"),
        "codeformer_env_exists": subprocess_status.get("codeformer_env_exists"),
        "codeformer_env_probe": conda_create_probe,
        "weights": subprocess_status.get("weights", []),
        "weights_exist": subprocess_status.get("weights_exist"),
        "standalone_smoke": standalone,
        "pipeline_test": pipeline,
        "activated": standalone["ok"] and pipeline["ok"],
        "known_limitations": [
            "CodeFormer chạy qua subprocess trong env conda riêng, không import trực tiếp vào env chính.",
            "Kết quả activation chỉ xác nhận backend chạy được, không phải đánh giá chất lượng phục hồi khuôn mặt.",
            "Không claim Blueprint 2.1 completed chỉ từ việc bật được CodeFormer.",
        ],
    }
    summary["blocked_reason"] = blocked_reason(summary)
    (OUTPUT_DIR / "standalone_test_summary.json").write_text(json.dumps(standalone, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "pipeline_test_summary.json").write_text(json.dumps(pipeline, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "activation_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    status_text = (
        "CodeFormer đã được activated theo điều kiện kỹ thuật: standalone output tồn tại và pipeline metadata có "
        "`face_restoration_applied=true`, `face_backend=codeformer`."
        if summary["activated"]
        else f"CodeFormer chưa được activated. Lý do chặn hiện tại: `{summary['blocked_reason']}`."
    )
    lines = [
        "# CodeFormer Activation Summary",
        "",
        f"- CodeFormer env: `{summary['codeformer_env']}`",
        f"- CodeFormer repo: `{summary['codeformer_repo']}`",
        f"- Repo exists: `{summary['codeformer_repo_exists']}`",
        f"- Conda available: `{summary['conda_available']}`",
        f"- Env exists: `{summary['codeformer_env_exists']}`",
        f"- Weights exist: `{summary['weights_exist']}`",
        f"- Standalone smoke pass: `{standalone['ok']}`",
        f"- Pipeline metadata true: `{pipeline['ok']}`",
        f"- Activated: `{summary['activated']}`",
        "",
        "## Status",
        "",
        status_text,
        "",
        "## Evidence",
        "",
        f"- `standalone_test_summary.json`: `{OUTPUT_DIR / 'standalone_test_summary.json'}`",
        f"- `pipeline_test_summary.json`: `{OUTPUT_DIR / 'pipeline_test_summary.json'}`",
        f"- `activation_status.json`: `{OUTPUT_DIR / 'activation_status.json'}`",
        "",
        "## Known limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["known_limitations"])
    lines.append("")
    (OUTPUT_DIR / "CODEFORMER_ACTIVATION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"summary: {OUTPUT_DIR / 'CODEFORMER_ACTIVATION_SUMMARY.md'}")
    print(f"activation_status: {OUTPUT_DIR / 'activation_status.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
