from __future__ import annotations

import argparse
import importlib
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

DEPENDENCIES = ["torch", "cv2", "basicsr", "facexlib", "gfpgan", "codeformer"]
CODEFORMER_REPO = Path(r"F:\deeplearning\external_models\CodeFormer")
CODEFORMER_ENV = "codeformer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra dependency cho Module 3 face restoration.")
    parser.add_argument("--output-json", default="outputs/blueprint21_acceleration/face_dependency_status.json")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def check_dependency(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "")
        location = getattr(module, "__file__", "")
        return {"name": name, "available": True, "version": str(version), "location": str(location), "error": ""}
    except Exception as exc:
        return {"name": name, "available": False, "version": "", "location": "", "error": str(exc)}


def run_probe(command: list[str], cwd: Path | None = None, timeout_sec: int = 60) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_sec,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-1000:],
        }
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout_tail": "", "stderr_tail": str(exc)}


def conda_env_exists(env_name: str) -> bool:
    probe = run_probe(["conda", "env", "list"])
    if not probe["ok"]:
        return False
    for line in probe["stdout_tail"].splitlines():
        parts = line.split()
        if parts and parts[0] == env_name:
            return True
    return False


def list_weights(repo_path: Path) -> list[str]:
    weights_dir = repo_path / "weights"
    if not weights_dir.exists():
        return []
    ignored_names = {"README.md", ".gitkeep"}
    return sorted(
        str(path.relative_to(repo_path))
        for path in weights_dir.rglob("*")
        if path.is_file() and path.name not in ignored_names
    )


def smoke_outputs(repo_path: Path) -> list[str]:
    outputs: list[str] = []
    for results_dir in [
        repo_path / "results" / "codeformer_test",
        repo_path / "results" / "codex_smoke",
    ]:
        if not results_dir.exists():
            continue
        outputs.extend(
            str(path.relative_to(repo_path))
            for path in results_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    return sorted(set(outputs))


def pipeline_face_metadata() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outputs_dir = PROJECT_ROOT / "outputs"
    if not outputs_dir.exists():
        return rows
    metadata_paths = sorted(outputs_dir.glob("**/metadata.json"), key=lambda path: path.stat().st_mtime)
    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "face_restoration_applied" in payload or "face_backend" in payload:
            rows.append(
                {
                    "path": str(metadata_path.relative_to(PROJECT_ROOT)),
                    "face_restoration_applied": payload.get("face_restoration_applied"),
                    "face_backend": payload.get("face_backend", payload.get("face_restoration_backend")),
                    "face_reason": payload.get("face_reason", payload.get("reason")),
                }
            )
    applied_rows = [
        row
        for row in rows
        if row.get("face_restoration_applied") is True and row.get("face_backend") == "codeformer"
    ]
    return (applied_rows + rows[-20:])[-40:]


def subprocess_status() -> dict[str, Any]:
    conda_probe = run_probe(["conda", "--version"])
    env_exists = conda_env_exists(CODEFORMER_ENV) if conda_probe["ok"] else False
    env_python = run_probe(["conda", "run", "-n", CODEFORMER_ENV, "python", "--version"]) if env_exists else None
    weights = list_weights(CODEFORMER_REPO)
    smoke = smoke_outputs(CODEFORMER_REPO)
    pipeline_metadata = pipeline_face_metadata()
    return {
        "conda_available": conda_probe["ok"],
        "conda_version": conda_probe["stdout_tail"].strip() if conda_probe["ok"] else "",
        "conda_error": "" if conda_probe["ok"] else conda_probe["stderr_tail"],
        "codeformer_env": CODEFORMER_ENV,
        "codeformer_env_exists": env_exists,
        "codeformer_env_python": env_python,
        "codeformer_repo": str(CODEFORMER_REPO),
        "codeformer_repo_exists": CODEFORMER_REPO.exists(),
        "weights": weights,
        "weights_exist": bool(weights),
        "standalone_smoke_outputs": smoke,
        "standalone_smoke_passed": bool(smoke),
        "pipeline_face_metadata_tail": pipeline_metadata,
        "pipeline_has_codeformer_applied_true": any(
            row.get("face_restoration_applied") is True and row.get("face_backend") == "codeformer"
            for row in pipeline_metadata
        ),
    }


def main() -> int:
    args = parse_args()
    rows = [check_dependency(name) for name in DEPENDENCIES]
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dependencies": rows,
        "all_core_available": all(row["available"] for row in rows if row["name"] in {"torch", "cv2"}),
        "codeformer_stack_available_in_main_env": all(
            row["available"] for row in rows if row["name"] in {"basicsr", "facexlib", "gfpgan", "codeformer"}
        ),
        "subprocess_codeformer": subprocess_status(),
        "recommendation": "Nếu thiếu basicsr/facexlib/gfpgan/codeformer trong env chính, hãy dùng môi trường riêng; script này không tự pip install.",
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("| package | available | version | note |")
    print("|---|---|---|---|")
    for row in rows:
        note = row["location"] if row["available"] else row["error"]
        print(f"| {row['name']} | {row['available']} | {row['version']} | {note} |")
    print("")
    sub = payload["subprocess_codeformer"]
    print("Subprocess CodeFormer:")
    print(f"- conda_available: {sub['conda_available']} {sub['conda_version']}")
    print(f"- codeformer_env_exists: {sub['codeformer_env_exists']}")
    print(f"- codeformer_repo_exists: {sub['codeformer_repo_exists']} {sub['codeformer_repo']}")
    print(f"- weights_exist: {sub['weights_exist']} count={len(sub['weights'])}")
    print(f"- standalone_smoke_passed: {sub['standalone_smoke_passed']}")
    print(f"- pipeline_has_codeformer_applied_true: {sub['pipeline_has_codeformer_applied_true']}")
    print("")
    print(payload["recommendation"])
    print(f"json: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
