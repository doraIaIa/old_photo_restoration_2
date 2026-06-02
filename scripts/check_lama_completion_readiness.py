from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = Path(r"F:\deeplearning\external_models\lama")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "blueprint21_completion" / "lama_readiness"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiểm tra readiness cho official/pretrained LaMa.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--conda-env", default="lama")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def run_command(command: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "ok": result.returncode == 0,
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc), "ok": False}


def find_files(root: Path, patterns: tuple[str, ...]) -> list[str]:
    if not root.exists():
        return []
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(str(path) for path in root.rglob(pattern))
    return sorted(set(matches))


def detect_repo_markers(workspace: Path) -> dict[str, Any]:
    markers = {
        "predict_py": find_files(workspace, ("predict.py",)),
        "saicinpainting_dir": [str(path) for path in workspace.rglob("saicinpainting") if path.is_dir()] if workspace.exists() else [],
        "configs_dir": [str(path) for path in workspace.rglob("configs") if path.is_dir()] if workspace.exists() else [],
    }
    markers["looks_like_lama_repo"] = bool(markers["predict_py"] and markers["saicinpainting_dir"])
    return markers


def conda_env_exists(conda_env_list_stdout: str, env_name: str) -> bool:
    for line in conda_env_list_stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == env_name:
            return True
    return False


def write_report(data: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lama_completion_readiness.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# LaMa Completion Readiness",
        "",
        f"- workspace: `{data['workspace']}`",
        f"- conda available: `{data['conda_available']}`",
        f"- conda env `{data['conda_env']}` exists: `{data['conda_env_exists']}`",
        f"- workspace exists: `{data['workspace_exists']}`",
        f"- looks like LaMa repo: `{data['repo_markers']['looks_like_lama_repo']}`",
        f"- checkpoint candidates: `{len(data['checkpoint_candidates'])}`",
        f"- torch import in env: `{data['torch_probe'].get('ok', False)}`",
        f"- official inference ready: `{data['official_inference_ready']}`",
        "",
        "## Missing requirements",
        "",
    ]
    for item in data["missing_requirements"]:
        lines.append(f"- {item}")
    if not data["missing_requirements"]:
        lines.append("- Không phát hiện thiếu điều kiện chính.")
    lines.extend(
        [
            "",
            "## Ghi chú",
            "",
            "- Script này chỉ kiểm tra readiness, không cài dependency nặng và không tải checkpoint.",
            "- Nếu dependency official LaMa trên Windows rủi ro, nên chạy baseline official trong WSL/Colab/Kaggle trước khi tích hợp sâu.",
        ]
    )
    (output_dir / "lama_completion_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace)
    output_dir = Path(args.output_dir)

    conda_path = shutil.which("conda")
    conda_version = run_command(["conda", "--version"], timeout=30) if conda_path else {"ok": False, "stdout": "", "stderr": "conda not found"}
    conda_env_list = run_command(["conda", "env", "list"], timeout=60) if conda_path else {"ok": False, "stdout": "", "stderr": "conda not found"}
    env_exists = conda_env_exists(conda_env_list.get("stdout", ""), args.conda_env) if conda_env_list.get("ok") else False
    python_probe = run_command(["conda", "run", "-n", args.conda_env, "python", "--version"], timeout=60) if env_exists else {"ok": False}
    torch_probe = (
        run_command(
            [
                "conda",
                "run",
                "-n",
                args.conda_env,
                "python",
                "-c",
                "import torch; print(torch.__version__); print(torch.cuda.is_available())",
            ],
            timeout=120,
        )
        if env_exists
        else {"ok": False, "stdout": "", "stderr": "env missing"}
    )

    repo_markers = detect_repo_markers(workspace)
    checkpoint_candidates = find_files(workspace, ("*.ckpt", "*.pth", "*.pt"))
    missing: list[str] = []
    if not conda_path:
        missing.append("Không tìm thấy conda trên PATH.")
    if not env_exists:
        missing.append(f"Conda env `{args.conda_env}` chưa tồn tại hoặc chưa truy cập được.")
    if not workspace.exists():
        missing.append(f"Workspace LaMa chưa tồn tại: {workspace}")
    if not repo_markers["looks_like_lama_repo"]:
        missing.append("Chưa thấy source official LaMa đầy đủ trong workspace.")
    if not checkpoint_candidates:
        missing.append("Chưa thấy pretrained checkpoint LaMa (.ckpt/.pth/.pt) trong workspace.")
    if not torch_probe.get("ok", False):
        missing.append("Chưa import được PyTorch trong env LaMa.")

    data = {
        "workspace": str(workspace),
        "workspace_exists": workspace.exists(),
        "conda_env": args.conda_env,
        "conda_available": bool(conda_path and conda_version.get("ok")),
        "conda_path": conda_path,
        "conda_version": conda_version,
        "conda_env_exists": env_exists,
        "conda_env_list": conda_env_list,
        "python_probe": python_probe,
        "torch_probe": torch_probe,
        "repo_markers": repo_markers,
        "checkpoint_candidates": checkpoint_candidates,
        "official_inference_ready": bool(env_exists and repo_markers["looks_like_lama_repo"] and checkpoint_candidates and torch_probe.get("ok")),
        "missing_requirements": missing,
    }
    write_report(data, output_dir)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
