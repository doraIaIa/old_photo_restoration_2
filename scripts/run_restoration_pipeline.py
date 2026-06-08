from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.postprocess.mask_refinement import VALID_REFINE_MODES
from src.restoration.dependency_checks import (
    check_official_lama_available,
    check_opencv_available,
    check_simple_lama_available,
)
from src.restoration.face_restoration import VALID_FACE_MODES, apply_face_restoration
from src.restoration.official_lama_adapter import OFFICIAL_LAMA_CHECKPOINT, run_official_lama_subprocess


R011_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r011-repair-ft-s42" / "best_iou.ckpt"
R012_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r012-manual-repair-ft-s42" / "best_iou.ckpt"
R013_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r013-gen120-fixed118-local" / "best_val_iou.pth"

MODE_CONFIGS = {
    "auto_r011": {"mask_source": "dl", "mask_refine": "none", "checkpoint": R011_CHECKPOINT, "model_version": "r011"},
    "auto_r011_union": {"mask_source": "union", "mask_refine": "none", "checkpoint": R011_CHECKPOINT, "model_version": "r011"},
    "auto_r011_refined": {
        "mask_source": "dl",
        "mask_refine": "repair_v3_conservative",
        "checkpoint": R011_CHECKPOINT,
        "model_version": "r011",
    },
    "auto_r011_union_refined": {
        "mask_source": "union",
        "mask_refine": "repair_v3_conservative",
        "checkpoint": R011_CHECKPOINT,
        "model_version": "r011",
    },
    "auto_r011_union_refined_face_auto": {
        "mask_source": "union",
        "mask_refine": "repair_v3_conservative",
        "checkpoint": R011_CHECKPOINT,
        "model_version": "r011",
    },
    "auto_r011_sensitive_low_threshold": {
        "mask_source": "dl",
        "mask_refine": "close_dilate1",
        "checkpoint": R011_CHECKPOINT,
        "model_version": "r011",
        "threshold": 0.15,
        "fallback_threshold": 0.15,
        "mask_dilate": 0,
        "experimental": True,
        "warning": "Recall-sensitive mask may increase false positives.",
        "original_baseline": "auto_r011_union_refined",
    },
    "auto_r012": {"mask_source": "dl", "mask_refine": "none", "checkpoint": R012_CHECKPOINT, "model_version": "r012"},
    "auto_r012_refined": {
        "mask_source": "dl",
        "mask_refine": "repair_v3_conservative",
        "checkpoint": R012_CHECKPOINT,
        "model_version": "r012",
    },
    "auto_r012_union_refined": {
        "mask_source": "union",
        "mask_refine": "repair_v3_conservative",
        "checkpoint": R012_CHECKPOINT,
        "model_version": "r012",
    },
    "auto_r013": {
        "mask_source": "dl",
        "mask_refine": "none",
        "checkpoint": R013_CHECKPOINT,
        "model_version": "r013",
        "threshold": 0.50,
        "fallback_threshold": 0.40,
        "experimental": True,
        "warning": "r013 candidate mode for local recovery; it is not r011.",
    },
    "auto_r013_union": {
        "mask_source": "union",
        "mask_refine": "none",
        "checkpoint": R013_CHECKPOINT,
        "model_version": "r013",
        "threshold": 0.50,
        "fallback_threshold": 0.40,
        "experimental": True,
        "warning": "r013 candidate mode for local recovery; it is not r011.",
    },
    "auto_r013_union_refined": {
        "mask_source": "union",
        "mask_refine": "repair_v3_conservative",
        "checkpoint": R013_CHECKPOINT,
        "model_version": "r013",
        "threshold": 0.50,
        "fallback_threshold": 0.40,
        "experimental": True,
        "warning": "r013 candidate mode for local recovery; it is not r011.",
    },
    "auto_r013_union_repair_wide": {
        "mask_source": "union",
        "mask_refine": "repair_wide_v1",
        "checkpoint": R013_CHECKPOINT,
        "model_version": "r013",
        "threshold": 0.50,
        "fallback_threshold": 0.40,
        "experimental": True,
        "warning": "r013 union repair-wide candidate tuned primarily for demo3 LaMa inpainting.",
    },
    "auto_r013_sensitive": {
        "mask_source": "dl",
        "mask_refine": "none",
        "checkpoint": R013_CHECKPOINT,
        "model_version": "r013",
        "threshold": 0.40,
        "fallback_threshold": 0.35,
        "experimental": True,
        "warning": "r013 sensitive candidate mode for local recovery; it is not r011.",
    },
    "auto_r013_sensitive_union": {
        "mask_source": "union",
        "mask_refine": "none",
        "checkpoint": R013_CHECKPOINT,
        "model_version": "r013",
        "threshold": 0.40,
        "fallback_threshold": 0.35,
        "experimental": True,
        "warning": "r013 sensitive candidate mode for local recovery; it is not r011.",
    },
    "auto_r013_sensitive_union_refined": {
        "mask_source": "union",
        "mask_refine": "repair_v3_conservative",
        "checkpoint": R013_CHECKPOINT,
        "model_version": "r013",
        "threshold": 0.40,
        "fallback_threshold": 0.35,
        "experimental": True,
        "warning": "r013 sensitive candidate mode for local recovery; it is not r011.",
    },
    "external": {"mask_source": "external", "mask_refine": "none", "checkpoint": R011_CHECKPOINT, "model_version": "external"},
    "external_face_auto": {
        "mask_source": "external",
        "mask_refine": "none",
        "checkpoint": R011_CHECKPOINT,
        "model_version": "external",
    },
}
SUPPORTED_RUN_DEMO_BACKENDS = {"auto", "simple_lama", "opencv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrypoint pipeline restoration cuối kỳ.")
    parser.add_argument("--image", "--input", dest="image", required=True)
    parser.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), default="auto_r011_union_refined")
    parser.add_argument("--checkpoint", default="", help="Checkpoint segmentation override. Mặc định theo mode.")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--fallback-threshold", type=float, default=0.40)
    parser.add_argument("--cv-profile", default="notebook_v7_candidate")
    parser.add_argument("--mask-dilate", type=int, default=0)
    parser.add_argument("--mask-refine", choices=sorted(VALID_REFINE_MODES), default=None)
    parser.add_argument("--external-mask", default="")
    parser.add_argument(
        "--backend",
        "--inpaint-backend",
        dest="backend",
        default="auto",
        choices=["auto", "simple_lama", "opencv", "official_lama", "fine_tuned_lama"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--face-mode", choices=sorted(VALID_FACE_MODES), default="auto")
    parser.add_argument("--face-strength", type=float, default=0.5)
    parser.add_argument("--codeformer-fidelity", type=float, default=None)
    parser.add_argument("--skip-face-restoration", action="store_true")
    parser.add_argument("--save-all-masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-mask-debug", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_checkpoint(args: argparse.Namespace, mode_config: dict[str, Any]) -> Path:
    if args.checkpoint:
        return resolve_path(args.checkpoint)
    return Path(mode_config["checkpoint"]).resolve()


def checkpoint_missing_message(mode: str, checkpoint_path: Path, mode_config: dict[str, Any]) -> str:
    model_version = str(mode_config.get("model_version", "unknown"))
    if model_version == "r011":
        return (
            f"Không tìm thấy checkpoint r011 cho mode {mode}: {checkpoint_path}. "
            "r011 checkpoint đang missing; không dùng r013 để giả mạo r011. "
            "Hãy khôi phục đúng r011 checkpoint hoặc chọn mode auto_r013/auto_r013_sensitive."
        )
    return f"Không tìm thấy checkpoint cho mode {mode}: {checkpoint_path}"


def copy_required_outputs(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "input.png",
        "final_mask.png",
        "overlay_final.png",
        "comparison_grid.png",
        "metadata.json",
        "dl_prob_mask.png",
        "cv_mask.png",
        "union_mask.png",
        "external_mask.png",
        "final_mask_before_refine.png",
        "final_mask_refined.png",
        "rejected_cv_by_final.png",
        "rejected_cv_overlay.png",
        "kept_cv_by_final.png",
        "mask_debug_stats.json",
        "dl_mask.png",
        "union_before_refine.png",
    ]:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, target_dir / filename)


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def fit_tile(image_rgb: np.ndarray, size: tuple[int, int] = (420, 420)) -> np.ndarray:
    target_width, target_height = size
    height, width = image_rgb.shape[:2]
    scale = min(target_width / max(width, 1), target_height / max(height, 1))
    resized = cv2.resize(
        image_rgb,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((target_height, target_width, 3), 245, dtype=np.uint8)
    y = (target_height - resized.shape[0]) // 2
    x = (target_width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def label_tile(tile_rgb: np.ndarray, label: str) -> np.ndarray:
    tile = tile_rgb.copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 36), (20, 20, 20), thickness=-1)
    cv2.putText(tile, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def rebuild_pipeline_comparison_grid(final_dir: Path) -> None:
    items = [
        ("input", final_dir / "input.png"),
        ("final mask", final_dir / "final_mask.png"),
        ("overlay final", final_dir / "overlay_final.png"),
        ("restored before face", final_dir / "restored_before_face.png"),
        ("restored final", final_dir / "restored_final.png"),
    ]
    tiles: list[np.ndarray] = []
    for label, path in items:
        if path.exists():
            tiles.append(label_tile(fit_tile(read_rgb(path)), label))
    if not tiles:
        return
    blank = label_tile(np.full_like(tiles[0], 245), "")
    while len(tiles) % 3:
        tiles.append(blank)
    grid = np.vstack([np.hstack(tiles[index : index + 3]) for index in range(0, len(tiles), 3)])
    write_rgb(final_dir / "comparison_grid.png", grid)


def run_demo(command: list[str]) -> None:
    print("run_demo:", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"run_demo.py lỗi với exit code {result.returncode}")


def build_errors_or_warnings(metadata: dict[str, Any], face_metadata: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for key in ["backend_warning", "inpaint_failed", "face_warning"]:
        value = metadata.get(key)
        if value:
            warnings.append(str(value))
    face_reason = str(face_metadata.get("reason", ""))
    if face_reason in {"dependency_not_available", "adapter_not_configured"}:
        warnings.append(f"face_restoration: {face_reason}")
    return warnings


def resolve_safe_inpaint_backend(requested_backend: str) -> tuple[str, dict[str, Any]]:
    simple_lama_status = check_simple_lama_available()
    opencv_status = check_opencv_available()
    official_lama_status = check_official_lama_available() if requested_backend == "official_lama" else {}
    if not opencv_status.get("available"):
        raise RuntimeError(f"OpenCV không khả dụng để làm fallback: {opencv_status.get('reason')}")

    fallback_info: dict[str, Any] = {
        "simple_lama_status": simple_lama_status,
        "opencv_status": opencv_status,
        "official_lama_status": official_lama_status,
        "run_demo_backend": requested_backend,
        "fallback_backend": requested_backend,
        "fallback_applied": False,
        "fallback_chain": [requested_backend],
        "backend_warning": None,
    }

    if requested_backend in {"auto", "opencv"}:
        return requested_backend, fallback_info

    if requested_backend == "simple_lama":
        if simple_lama_status.get("available"):
            return "simple_lama", fallback_info
        fallback_info.update(
            {
                "run_demo_backend": "opencv",
                "fallback_backend": "opencv",
                "fallback_applied": True,
                "fallback_chain": ["simple_lama", "opencv"],
                "backend_warning": (
                    "simple_lama unavailable, fallback opencv: "
                    f"{simple_lama_status.get('reason', 'unknown')}"
                ),
            }
        )
        return "opencv", fallback_info

    if requested_backend == "official_lama":
        if simple_lama_status.get("available"):
            fallback_info.update(
                {
                    "run_demo_backend": "simple_lama",
                    "fallback_backend": "simple_lama",
                    "fallback_chain": ["official_lama", "simple_lama"],
                }
            )
            return "simple_lama", fallback_info
        fallback_info.update(
            {
                "run_demo_backend": "opencv",
                "fallback_backend": "opencv",
                "fallback_chain": ["official_lama", "simple_lama", "opencv"],
                "backend_warning": (
                    "simple_lama unavailable before official_lama fallback chain: "
                    f"{simple_lama_status.get('reason', 'unknown')}"
                ),
            }
        )
        return "opencv", fallback_info

    return requested_backend, fallback_info


def main() -> int:
    args = parse_args()
    image_path = resolve_path(args.image)
    output_root = resolve_path(args.output_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy image: {image_path}")

    mode_config = MODE_CONFIGS[args.mode]
    checkpoint_path = resolve_checkpoint(args, mode_config)
    if not checkpoint_path.exists() and str(mode_config.get("model_version", "unknown")) == "r011":
        raise FileNotFoundError(checkpoint_missing_message(args.mode, checkpoint_path, mode_config))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint cho mode {args.mode}: {checkpoint_path}")

    if args.backend == "fine_tuned_lama":
        raise RuntimeError(
            "fine_tuned_lama chưa có adapter inference trong pipeline hiện tại. "
            "Dùng simple_lama/opencv hoặc chuẩn bị workspace theo docs/LAMA_FINETUNE_ACCELERATION_PLAN.md."
        )
    requested_backend = args.backend
    run_demo_backend, fallback_info = resolve_safe_inpaint_backend(requested_backend)

    mask_source = str(mode_config["mask_source"])
    mask_refine = args.mask_refine if args.mask_refine is not None else str(mode_config["mask_refine"])
    effective_threshold = float(mode_config.get("threshold", args.threshold))
    effective_fallback_threshold = float(mode_config.get("fallback_threshold", args.fallback_threshold))
    effective_mask_dilate = int(mode_config.get("mask_dilate", args.mask_dilate))
    if mask_source == "external" and not args.external_mask:
        raise ValueError("Mode external yêu cầu --external-mask.")
    external_mask_path = resolve_path(args.external_mask) if args.external_mask else None
    if external_mask_path is not None and not external_mask_path.exists():
        raise FileNotFoundError(f"Không tìm thấy external mask: {external_mask_path}")

    scratch_root = output_root / "_run_demo_raw" / args.mode
    command = [
        sys.executable,
        "scripts\\run_demo.py",
        "--image",
        str(image_path),
        "--checkpoint",
        str(checkpoint_path),
        "--threshold",
        f"{effective_threshold:.2f}",
        "--fallback-threshold",
        f"{effective_fallback_threshold:.2f}",
        "--backend",
        run_demo_backend,
        "--mask-source",
        mask_source,
        "--cv-profile",
        args.cv_profile,
        "--mask-dilate",
        str(effective_mask_dilate),
        "--mask-refine",
        mask_refine,
        "--output-dir",
        str(scratch_root),
        "--device",
        args.device,
        "--save-prob-mask",
    ]
    if args.save_all_masks:
        command.append("--save-all-masks")
    if external_mask_path is not None:
        command.extend(["--external-mask", str(external_mask_path)])
    run_demo(command)

    run_demo_dir = scratch_root / image_path.stem / mask_source
    if args.save_mask_debug:
        primary_tag = f"t{effective_threshold:.2f}".replace(".", "p")
        dl_mask_path = run_demo_dir / f"dl_mask_{primary_tag}.png"
        cv_mask_path = run_demo_dir / "cv_mask.png"
        final_mask_path = run_demo_dir / "final_mask.png"
        union_mask_path = run_demo_dir / "union_mask.png"
        input_path = run_demo_dir / "input.png"

        dl_mask = cv2.imread(str(dl_mask_path), cv2.IMREAD_GRAYSCALE) if dl_mask_path.exists() else None
        cv_mask = cv2.imread(str(cv_mask_path), cv2.IMREAD_GRAYSCALE) if cv_mask_path.exists() else None
        final_mask = cv2.imread(str(final_mask_path), cv2.IMREAD_GRAYSCALE) if final_mask_path.exists() else None
        union_mask = cv2.imread(str(union_mask_path), cv2.IMREAD_GRAYSCALE) if union_mask_path.exists() else None
        input_img = cv2.imread(str(input_path)) if input_path.exists() else None

        if final_mask is not None:
            h, w = final_mask.shape[:2]
            if dl_mask is None:
                dl_mask = np.zeros((h, w), dtype=np.uint8)
            if cv_mask is None:
                cv_mask = np.zeros((h, w), dtype=np.uint8)
            if union_mask is None:
                union_mask = cv2.max(dl_mask, cv_mask)

            rejected_cv = cv2.bitwise_and(cv_mask, cv2.bitwise_not(final_mask))
            kept_cv = cv2.bitwise_and(cv_mask, final_mask)

            cv2.imwrite(str(run_demo_dir / "rejected_cv_by_final.png"), rejected_cv)
            cv2.imwrite(str(run_demo_dir / "kept_cv_by_final.png"), kept_cv)
            cv2.imwrite(str(run_demo_dir / "dl_mask.png"), dl_mask)
            cv2.imwrite(str(run_demo_dir / "union_before_refine.png"), union_mask)

            if input_img is not None:
                overlay_img = input_img.copy()
                color = np.array([0, 0, 255], dtype=np.uint8)
                mask_bool = rejected_cv > 0
                overlay_img[mask_bool] = np.clip(overlay_img[mask_bool] * 0.55 + color * 0.45, 0, 255).astype(np.uint8)
                cv2.imwrite(str(run_demo_dir / "rejected_cv_overlay.png"), overlay_img)

            dl_cnt = np.count_nonzero(dl_mask)
            cv_cnt = np.count_nonzero(cv_mask)
            union_cnt = np.count_nonzero(union_mask)
            final_cnt = np.count_nonzero(final_mask)
            rej_cnt = np.count_nonzero(rejected_cv)
            kept_cnt = np.count_nonzero(kept_cv)
            total_pixels = dl_mask.size

            num_labels_cv, _, _, _ = cv2.connectedComponentsWithStats(cv_mask, connectivity=8)
            num_labels_final, _, _, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)

            stats = {
                "dl_mask_ratio": float(dl_cnt / total_pixels),
                "cv_mask_ratio": float(cv_cnt / total_pixels),
                "union_before_refine_ratio": float(union_cnt / total_pixels),
                "final_mask_ratio": float(final_cnt / total_pixels),
                "rejected_cv_ratio": float(rej_cnt / total_pixels),
                "kept_cv_ratio": float(kept_cnt / total_pixels),
                "rejected_cv_over_cv_ratio": float(rej_cnt / cv_cnt) if cv_cnt > 0 else 0.0,
                "final_over_union_ratio": float(final_cnt / union_cnt) if union_cnt > 0 else 0.0,
                "number_connected_components_cv": int(num_labels_cv - 1),
                "number_connected_components_final": int(num_labels_final - 1),
                "mode": args.mode,
                "checkpoint": str(checkpoint_path),
                "threshold": effective_threshold
            }

            with open(run_demo_dir / "mask_debug_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

    final_dir = output_root / image_path.stem / args.mode
    copy_required_outputs(run_demo_dir, final_dir)

    run_demo_restored = run_demo_dir / "restored_final.png"
    if not run_demo_restored.exists():
        raise FileNotFoundError(f"Không tìm thấy restored_final từ run_demo: {run_demo_restored}")
    restored_before_face = final_dir / "restored_before_face.png"
    shutil.copy2(run_demo_restored, restored_before_face)
    official_lama_result: dict[str, Any] | None = None
    actual_backend = str(fallback_info.get("fallback_backend", run_demo_backend))
    fallback_applied = bool(fallback_info.get("fallback_applied", False))
    fallback_chain = list(fallback_info.get("fallback_chain", [requested_backend]))
    backend_warning = fallback_info.get("backend_warning")
    if requested_backend == "official_lama":
        official_lama_result = run_official_lama_subprocess(
            final_dir / "input.png",
            final_dir / "final_mask.png",
            final_dir / "official_lama_module",
        )
        if official_lama_result.get("ok"):
            official_output = Path(str(official_lama_result["output"]))
            shutil.copy2(official_output, restored_before_face)
            actual_backend = "official_lama"
            fallback_applied = False
            fallback_chain = ["official_lama"]
            backend_warning = None
        else:
            actual_backend = str(fallback_info.get("fallback_backend", run_demo_backend))
            fallback_applied = True
            if "official_lama" not in fallback_chain:
                fallback_chain.insert(0, "official_lama")
            backend_warning = (
                f"official_lama failed, fallback {actual_backend}: "
                f"{official_lama_result.get('reason', 'unknown')}"
            )
            print(
                f"official_lama_failed_fallback_{actual_backend}: "
                f"{official_lama_result.get('reason', 'unknown')}",
                file=sys.stderr,
            )

    effective_face_mode = "off" if args.skip_face_restoration else args.face_mode
    codeformer_fidelity = args.codeformer_fidelity if args.codeformer_fidelity is not None else args.face_strength
    face_restored_rgb, face_metadata = apply_face_restoration(
        read_rgb(restored_before_face),
        mode=effective_face_mode,
        strength=codeformer_fidelity,
        output_dir=final_dir / "face_module",
    )
    restored_final = final_dir / "restored_final.png"
    write_rgb(restored_final, face_restored_rgb)

    metadata_path = final_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if requested_backend != "official_lama":
        actual_backend = str(metadata.get("actual_backend") or actual_backend)
        if requested_backend == "simple_lama" and actual_backend != "simple_lama":
            fallback_applied = True
            fallback_chain = ["simple_lama", actual_backend]
            backend_warning = backend_warning or metadata.get("backend_warning")
        elif requested_backend == "auto":
            fallback_chain = ["auto", actual_backend]
    metadata["pipeline_mode"] = args.mode
    metadata["mask_mode"] = args.mode
    metadata["experimental"] = bool(mode_config.get("experimental", False))
    if mode_config.get("experimental"):
        metadata["low_threshold"] = effective_threshold
        metadata["original_baseline"] = mode_config.get("original_baseline")
        metadata["warning"] = mode_config.get("warning")
    metadata["pipeline_output_dir"] = str(final_dir)
    metadata["checkpoint_used"] = str(checkpoint_path)
    metadata["segmentation_model_version"] = str(mode_config.get("model_version", "unknown"))
    metadata["segmentation_checkpoint"] = str(checkpoint_path)
    metadata["segmentation_threshold"] = float(effective_threshold)
    metadata["mask_source"] = mask_source
    metadata["mask_refine"] = mask_refine
    metadata["cv_mask_used"] = mask_source in ("union", "cv_only")
    metadata["union_refinement_used"] = mask_refine != "none"
    metadata["r011_checkpoint_available"] = R011_CHECKPOINT.exists()
    metadata["r013_checkpoint_available"] = R013_CHECKPOINT.exists()
    metadata["inpainting_backend_requested"] = requested_backend
    metadata["inpainting_backend_actual"] = actual_backend
    metadata["actual_backend"] = actual_backend
    metadata["fallback_applied"] = bool(fallback_applied)
    metadata["fallback_chain"] = fallback_chain
    metadata["simple_lama_available"] = bool(fallback_info["simple_lama_status"].get("available"))
    metadata["simple_lama_reason"] = fallback_info["simple_lama_status"].get("reason")
    metadata["simple_lama_detail"] = fallback_info["simple_lama_status"].get("detail")
    metadata["opencv_available"] = bool(fallback_info["opencv_status"].get("available"))
    metadata["opencv_reason"] = fallback_info["opencv_status"].get("reason")
    if requested_backend == "official_lama":
        metadata["official_lama_available"] = bool(fallback_info.get("official_lama_status", {}).get("available"))
        metadata["official_lama_status"] = fallback_info.get("official_lama_status")
        metadata["official_lama_result"] = official_lama_result
        metadata["official_lama_checkpoint"] = str(OFFICIAL_LAMA_CHECKPOINT)
        metadata["official_lama_env_requested"] = (official_lama_result or {}).get("official_lama_env_requested")
        metadata["official_lama_env_actual"] = (official_lama_result or {}).get("official_lama_env_actual")
        metadata["official_lama_device_requested"] = (official_lama_result or {}).get("official_lama_device_requested")
        metadata["official_lama_device_actual"] = (official_lama_result or {}).get("official_lama_device_actual")
        metadata["official_lama_device"] = (official_lama_result or {}).get("official_lama_device_actual")
        metadata["official_lama_cuda_available"] = (official_lama_result or {}).get("official_lama_cuda_available")
        metadata["official_lama_torch_version"] = (official_lama_result or {}).get("official_lama_torch_version")
        metadata["official_lama_cuda_build"] = (official_lama_result or {}).get("official_lama_cuda_build")
        metadata["official_lama_reason"] = (official_lama_result or {}).get(
            "reason",
            fallback_info.get("official_lama_status", {}).get("reason"),
        )
    if backend_warning:
        metadata["backend_warning"] = backend_warning
    elif metadata.get("backend_warning") and not fallback_applied:
        metadata["backend_warning"] = None
    metadata["face_mode"] = effective_face_mode
    metadata["face_strength"] = float(args.face_strength)
    metadata["codeformer_fidelity_requested"] = None if args.codeformer_fidelity is None else float(args.codeformer_fidelity)
    metadata["restored_before_face_path"] = str(restored_before_face)
    metadata["restored_final_path"] = str(restored_final)
    metadata["face_module_enabled"] = face_metadata.get("face_module_enabled", False)
    metadata["face_detection_backend"] = face_metadata.get("face_detection_backend", "none")
    metadata["faces_detected"] = face_metadata.get("faces_detected", 0)
    metadata["face_restoration_applied"] = face_metadata.get("face_restoration_applied", False)
    metadata["face_restoration_backend"] = face_metadata.get("face_restoration_backend", "none")
    metadata["face_backend"] = face_metadata.get("face_restoration_backend", "none")
    metadata["face_reason"] = face_metadata.get("reason", "")
    metadata["face_restore_reason"] = face_metadata.get("reason", "")
    metadata["face_warning"] = face_metadata.get("warning")
    metadata["codeformer_fidelity"] = face_metadata.get("codeformer_fidelity")
    metadata["codeformer_output"] = face_metadata.get("codeformer_output")
    metadata["codeformer_result"] = face_metadata.get("codeformer_result")
    metadata["errors_or_warnings"] = build_errors_or_warnings(metadata, face_metadata)

    mask_debug_stats_path = final_dir / "mask_debug_stats.json"
    if mask_debug_stats_path.exists():
        try:
            stats = json.loads(mask_debug_stats_path.read_text(encoding="utf-8"))
            for key in ["dl_mask_ratio", "cv_mask_ratio", "union_before_refine_ratio", "final_mask_ratio", "rejected_cv_over_cv_ratio"]:
                if key in stats:
                    metadata[key] = stats[key]
        except Exception:
            pass

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    rebuild_pipeline_comparison_grid(final_dir)

    print(f"image: {image_path}")
    print(f"mode: {args.mode}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"inpainting_backend_requested: {requested_backend}")
    print(f"inpainting_backend_actual: {metadata.get('inpainting_backend_actual', '')}")
    print(f"final_mask_ratio: {metadata.get('final_mask_ratio', '')}")
    print(f"face_mode: {effective_face_mode}")
    print(f"face_restoration_applied: {metadata.get('face_restoration_applied', False)}")
    print(f"face_reason: {metadata.get('face_reason', '')}")
    print(f"output_dir: {final_dir}")
    print(f"comparison_grid: {final_dir / 'comparison_grid.png'}")
    print(f"restored_before_face: {restored_before_face}")
    print(f"restored_final: {restored_final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
