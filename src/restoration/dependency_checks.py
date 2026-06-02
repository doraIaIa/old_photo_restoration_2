from __future__ import annotations

import subprocess
from typing import Any

from src.restoration.official_lama_adapter import (
    OFFICIAL_LAMA_CHECKPOINT,
    OFFICIAL_LAMA_CPU_ENV,
    OFFICIAL_LAMA_GPU_ENV,
    OFFICIAL_LAMA_REPO,
    select_official_lama_runtime,
)


def check_simple_lama_available() -> dict[str, Any]:
    """Kiểm tra SimpleLama bằng import thật, không chỉ tìm spec."""
    try:
        from simple_lama_inpainting import SimpleLama  # noqa: F401
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "") or ""
        if missing_name == "simple_lama_inpainting":
            return {
                "available": False,
                "reason": "module_not_found",
                "detail": "missing simple_lama_inpainting",
                "module": "simple_lama_inpainting",
            }
        return {
            "available": False,
            "reason": "dependency_module_not_found",
            "detail": str(exc),
            "module": "simple_lama_inpainting",
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": "import_error",
            "detail": str(exc),
            "module": "simple_lama_inpainting",
        }
    return {
        "available": True,
        "reason": "available",
        "detail": "import ok",
        "module": "simple_lama_inpainting",
    }


def check_opencv_available() -> dict[str, Any]:
    try:
        import cv2

        version = cv2.__version__
    except Exception as exc:
        return {"available": False, "reason": "import_error", "detail": str(exc), "module": "cv2"}
    return {"available": True, "reason": "available", "detail": f"cv2 {version}", "module": "cv2"}


def check_official_lama_available(timeout_sec: int = 20) -> dict[str, Any]:
    missing: list[str] = []
    predict_script = OFFICIAL_LAMA_REPO / "bin" / "predict.py"
    if not OFFICIAL_LAMA_REPO.exists():
        missing.append("repo_missing")
    if not predict_script.exists():
        missing.append("predict_script_missing")
    if not OFFICIAL_LAMA_CHECKPOINT.exists():
        missing.append("checkpoint_missing")
    if missing:
        return {
            "available": False,
            "status": "unavailable",
            "reason": ",".join(missing),
            "device": "cpu",
            "env": OFFICIAL_LAMA_CPU_ENV,
            "repo": str(OFFICIAL_LAMA_REPO),
            "checkpoint": str(OFFICIAL_LAMA_CHECKPOINT),
        }

    runtime = select_official_lama_runtime(prefer_gpu=True)
    if runtime.get("available"):
        device = runtime.get("official_lama_device_actual")
        env_name = runtime.get("official_lama_env_actual")
        return {
            "available": True,
            "status": "gpu" if device == "cuda" else "cpu-only",
            "reason": runtime.get("reason", "available"),
            "detail": (
                f"env={env_name}, device={device}, "
                f"torch={runtime.get('official_lama_torch_version')}, "
                f"cuda_build={runtime.get('official_lama_cuda_build')}"
            ),
            "device": device,
            "env": env_name,
            "selected_env": env_name,
            "selected_device": device,
            "gpu_env": OFFICIAL_LAMA_GPU_ENV,
            "cpu_env": OFFICIAL_LAMA_CPU_ENV,
            "gpu_probe": runtime.get("gpu_probe"),
            "cpu_probe": runtime.get("cpu_probe"),
            "cuda_available": bool(runtime.get("official_lama_cuda_available", False)),
            "torch_version": runtime.get("official_lama_torch_version"),
            "cuda_build": runtime.get("official_lama_cuda_build"),
            "repo": str(OFFICIAL_LAMA_REPO),
            "checkpoint": str(OFFICIAL_LAMA_CHECKPOINT),
        }

    command = ["conda", "run", "-n", OFFICIAL_LAMA_CPU_ENV, "python", "--version"]
    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "status": "unavailable",
            "reason": "conda_env_timeout",
            "device": "cpu",
            "env": OFFICIAL_LAMA_CPU_ENV,
            "repo": str(OFFICIAL_LAMA_REPO),
            "checkpoint": str(OFFICIAL_LAMA_CHECKPOINT),
        }
    except Exception as exc:
        return {
            "available": False,
            "status": "unavailable",
            "reason": f"conda_env_check_failed: {exc}",
            "device": "cpu",
            "env": OFFICIAL_LAMA_CPU_ENV,
            "repo": str(OFFICIAL_LAMA_REPO),
            "checkpoint": str(OFFICIAL_LAMA_CHECKPOINT),
        }

    if completed.returncode != 0:
        return {
            "available": False,
            "status": "unavailable",
            "reason": "conda_env_unavailable",
            "detail": (completed.stderr or completed.stdout or "").strip()[-500:],
            "device": "cpu",
            "env": OFFICIAL_LAMA_CPU_ENV,
            "repo": str(OFFICIAL_LAMA_REPO),
            "checkpoint": str(OFFICIAL_LAMA_CHECKPOINT),
        }
    return {
        "available": True,
        "status": "cpu-only",
        "reason": "available",
        "detail": (completed.stdout or completed.stderr or "").strip(),
        "device": "cpu",
        "env": OFFICIAL_LAMA_CPU_ENV,
        "repo": str(OFFICIAL_LAMA_REPO),
        "checkpoint": str(OFFICIAL_LAMA_CHECKPOINT),
    }
