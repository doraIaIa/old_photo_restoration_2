from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "post_commit_validation"
DEFAULT_DEMO_DIR = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy validation nhẹ sau commit cho pipeline phục hồi ảnh cũ."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Thư mục ghi JSON/MD report runtime.")
    parser.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR), help="Thư mục demo input sẵn có trong repo.")
    parser.add_argument("--run-smoke", action="store_true", help="Chạy một smoke path nhỏ nếu demo input và checkpoint có sẵn.")
    parser.add_argument("--smoke-backend", default="opencv", choices=["opencv", "simple_lama", "official_lama"])
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_command(command: list[str], timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "ok": completed.returncode == 0,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "ok": False, "error": str(exc)}


def import_probe(module_name: str) -> dict[str, Any]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return {"module": module_name, "ok": False, "reason": type(exc).__name__, "detail": str(exc)}
    return {"module": module_name, "ok": True, "reason": "import_ok"}


def find_demo_image(demo_dir: Path) -> Path | None:
    for stem in ("demo1", "demo2", "demo3"):
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = demo_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def check_metadata_contract(metadata_path: Path) -> dict[str, Any]:
    required_fields = [
        "inpainting_backend_requested",
        "inpainting_backend_actual",
        "fallback_applied",
        "fallback_chain",
        "final_mask_ratio",
        "face_restoration_applied",
        "codeformer_fidelity",
    ]
    if not metadata_path.exists():
        return {"ok": False, "reason": "metadata_missing", "metadata_path": str(metadata_path)}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": "metadata_read_failed", "detail": str(exc), "metadata_path": str(metadata_path)}
    missing = [field for field in required_fields if field not in payload]
    return {
        "ok": not missing,
        "missing_fields": missing,
        "metadata_path": str(metadata_path),
        "backend_requested": payload.get("inpainting_backend_requested"),
        "backend_actual": payload.get("inpainting_backend_actual"),
        "fallback_applied": payload.get("fallback_applied"),
        "mask_ratio": payload.get("final_mask_ratio"),
    }


def run_smoke_if_requested(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    if not args.run_smoke:
        return {"requested": False, "ok": True, "reason": "smoke_not_requested"}

    demo_dir = resolve_path(args.demo_dir)
    image_path = find_demo_image(demo_dir)
    checkpoint = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r011-repair-ft-s42" / "best_iou.ckpt"
    if image_path is None:
        return {
            "requested": True,
            "ok": False,
            "reason": "demo_input_missing",
            "detail": f"Không tìm thấy demo1/demo2/demo3 trong {demo_dir}.",
        }
    if not checkpoint.exists():
        return {
            "requested": True,
            "ok": False,
            "reason": "checkpoint_missing",
            "detail": f"Không tìm thấy checkpoint r011: {checkpoint}.",
        }

    smoke_root = output_dir / "smoke_run"
    command = [
        sys.executable,
        "scripts\\run_restoration_pipeline.py",
        "--image",
        str(image_path),
        "--mode",
        "auto_r011_union_refined",
        "--backend",
        args.smoke_backend,
        "--face-mode",
        "off",
        "--codeformer-fidelity",
        "0.70",
        "--output-dir",
        str(smoke_root),
        "--device",
        args.device,
    ]
    result = run_command(command, timeout=900)
    metadata_path = smoke_root / image_path.stem / "auto_r011_union_refined" / "metadata.json"
    contract = check_metadata_contract(metadata_path)
    return {
        "requested": True,
        "ok": bool(result.get("ok") and contract.get("ok")),
        "image": str(image_path),
        "command_result": result,
        "metadata_contract": contract,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "post_commit_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    import_ok = sum(1 for item in report["imports"] if item.get("ok"))
    import_total = len(report["imports"])
    lines = [
        "# Post-Commit Validation",
        "",
        f"- timestamp: `{report['timestamp']}`",
        f"- git commit: `{report['git']['last_commit'].get('stdout_tail', '').strip()}`",
        f"- import checks: `{import_ok}/{import_total}`",
        f"- opencv: `{report['dependencies']['opencv'].get('available')}`",
        f"- simple_lama: `{report['dependencies']['simple_lama'].get('available')}`",
        f"- official_lama: `{report['dependencies']['official_lama'].get('status')}`",
        f"- smoke requested: `{report['smoke']['requested']}`",
        f"- smoke ok: `{report['smoke']['ok']}`",
        "",
        "## Ghi chú",
        "",
        "- Report này chỉ xác nhận import, dependency readiness và smoke metadata nếu được yêu cầu.",
        "- GPU chỉ được ghi nhận ở mức smoke/readiness; script không kết luận GPU nhanh hơn CPU.",
    ]
    (output_dir / "post_commit_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)

    from src.restoration.dependency_checks import (
        check_official_lama_available,
        check_opencv_available,
        check_simple_lama_available,
    )

    imports = [
        import_probe("app_gradio"),
        import_probe("src.restoration.dependency_checks"),
        import_probe("src.restoration.official_lama_adapter"),
        import_probe("src.restoration.face_restoration"),
        import_probe("src.postprocess.mask_refinement"),
        import_probe("src.models.segmenter"),
    ]
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "git": {
            "status_short": run_command(["git", "status", "-u", "--short"], timeout=30),
            "last_commit": run_command(["git", "log", "--oneline", "-1"], timeout=30),
        },
        "imports": imports,
        "dependencies": {
            "opencv": check_opencv_available(),
            "simple_lama": check_simple_lama_available(),
            "official_lama": check_official_lama_available(),
        },
        "smoke": run_smoke_if_requested(args, output_dir),
        "gpu_speedup_claim": "not_evaluated",
    }
    write_report(report, output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    import_pass = all(item.get("ok") for item in imports)
    smoke_pass = bool(report["smoke"].get("ok"))
    return 0 if import_pass and smoke_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
