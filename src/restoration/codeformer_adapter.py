from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


CODEFORMER_REPO = Path(r"F:\deeplearning\external_models\CodeFormer")
CODEFORMER_ENV = "codeformer"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:] if text else ""


def _list_output_images(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        [path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _select_final_output(raw_output_dir: Path, input_stem: str) -> Path | None:
    final_results = raw_output_dir / "final_results"
    if final_results.exists():
        candidates = [
            path
            for path in final_results.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem == input_stem
        ]
        if candidates:
            return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        images = _list_output_images(final_results)
        if images:
            return images[0]
    images = _list_output_images(raw_output_dir)
    return images[0] if images else None


def _run(command: list[str], timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=CODEFORMER_REPO,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_sec,
    )


def run_codeformer_subprocess(
    input_image_path: Path,
    output_dir: Path,
    fidelity: float = 0.7,
    face_upsample: bool = True,
    timeout_sec: int = 300,
) -> dict[str, Any]:
    """Chạy CodeFormer qua conda subprocess, không import trực tiếp vào env chính."""
    input_path = Path(input_image_path)
    output_path = Path(output_dir)
    if not input_path.exists():
        return {"ok": False, "reason": "input_missing", "input": str(input_path), "backend": "codeformer"}
    if not CODEFORMER_REPO.exists():
        return {"ok": False, "reason": "repo_missing", "repo": str(CODEFORMER_REPO), "backend": "codeformer"}

    run_id = f"{input_path.stem}_{int(time.time() * 1000)}"
    temp_input_dir = output_path / "_codeformer_input" / run_id
    raw_output_dir = output_path / "codeformer_raw" / run_id
    temp_input_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    copied_input = temp_input_dir / f"{input_path.stem}{input_path.suffix.lower()}"
    shutil.copy2(input_path, copied_input)

    base_command = [
        "conda",
        "run",
        "-n",
        CODEFORMER_ENV,
        "python",
        "inference_codeformer.py",
        "-w",
        f"{float(fidelity):.3f}",
        "--input_path",
        str(temp_input_dir),
        "--output_path",
        str(raw_output_dir),
    ]
    commands = [base_command + (["--face_upsample"] if face_upsample else [])]
    if face_upsample:
        commands.append(base_command)

    attempts: list[dict[str, Any]] = []
    for command in commands:
        try:
            result = _run(command, timeout_sec=timeout_sec)
            attempt = {
                "command": command,
                "returncode": result.returncode,
                "stdout_tail": _tail(result.stdout),
                "stderr_tail": _tail(result.stderr),
            }
            attempts.append(attempt)
        except Exception as exc:
            attempt = {"command": command, "returncode": None, "stdout_tail": "", "stderr_tail": str(exc)}
            attempts.append(attempt)
            continue

        if result.returncode == 0:
            selected = _select_final_output(raw_output_dir, copied_input.stem)
            if selected is not None:
                return {
                    "ok": True,
                    "reason": "applied",
                    "backend": "codeformer",
                    "input": str(input_path),
                    "copied_input": str(copied_input),
                    "output": str(selected),
                    "raw_output_dir": str(raw_output_dir),
                    "fidelity": float(fidelity),
                    "face_upsample_requested": bool(face_upsample),
                    "face_upsample_used": "--face_upsample" in command,
                    "attempts": attempts,
                    "returncode": result.returncode,
                    "stdout_tail": _tail(result.stdout),
                    "stderr_tail": _tail(result.stderr),
                }

    last = attempts[-1] if attempts else {}
    return {
        "ok": False,
        "reason": "subprocess_failed",
        "backend": "codeformer",
        "input": str(input_path),
        "raw_output_dir": str(raw_output_dir),
        "fidelity": float(fidelity),
        "attempts": attempts,
        "returncode": last.get("returncode"),
        "stdout_tail": last.get("stdout_tail", ""),
        "stderr_tail": last.get("stderr_tail", ""),
    }
