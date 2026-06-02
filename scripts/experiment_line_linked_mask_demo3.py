from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from scripts.run_demo import (  # noqa: E402
    build_cv_crack_mask,
    build_inference_tensor,
    ensure_gray_uint8,
    ensure_rgb_uint8,
    inpaint_with_simple_lama,
    inpaint_with_opencv,
    load_model,
    load_rgb,
    make_comparison_grid,
    make_gray_rgb,
    make_overlay,
    mask_ratio,
    or_masks,
    probability_to_uint8,
    resize_probability_mask,
    save_gray,
    save_rgb,
    try_create_simple_lama,
)
from src.postprocess.mask_refinement import refine_mask  # noqa: E402


DEMO_ROOT = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_demo_benchmark" / "demo3_line_linking_experiment"
CHECKPOINTS = {
    "r011": PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r011-repair-ft-s42" / "best_iou.ckpt",
    "r012": PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r012-manual-repair-ft-s42" / "best_iou.ckpt",
}
BACKENDS = ["simple_lama", "opencv"]
HYSTERESIS_SWEEP = [
    (low_threshold, high_threshold, dilation, close_kernel)
    for low_threshold in [0.10, 0.15, 0.20]
    for high_threshold in [0.45, 0.50]
    for dilation in [3, 5]
    for close_kernel in [3, 5]
]


@dataclass
class MaskVariant:
    name: str
    checkpoint_tag: str
    mask: np.ndarray
    probability: np.ndarray
    line_response: np.ndarray
    metadata: dict[str, Any]


def find_demo3_image() -> Path:
    for extension in [".jpg", ".png", ".jpeg"]:
        path = DEMO_ROOT / f"demo3{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Không tìm thấy demo3 trong {DEMO_ROOT}")


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def infer_probability(image_rgb: np.ndarray, checkpoint_path: Path, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    model, checkpoint = load_model(checkpoint_path, device)
    image_size = 512
    input_tensor = build_inference_tensor(image_rgb, image_size).to(device)
    logits = model(input_tensor)
    probability = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
    if probability.ndim != 2:
        raise ValueError(f"Probability map phải có 2 chiều, nhận được {probability.shape}")
    probability = resize_probability_mask(probability, image_rgb.shape[:2])
    probability = np.clip(probability, 0.0, 1.0).astype(np.float32)
    return probability, {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "checkpoint_model_config": checkpoint.get("model_config"),
        "probability_available": True,
    }


def normalize_uint8(score: np.ndarray) -> np.ndarray:
    score_float = score.astype(np.float32)
    min_value = float(np.min(score_float))
    max_value = float(np.max(score_float))
    if max_value <= min_value:
        return np.zeros(score.shape, dtype=np.uint8)
    return np.clip((score_float - min_value) * 255.0 / (max_value - min_value), 0, 255).astype(np.uint8)


def build_line_response(image_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(ensure_rgb_uint8(image_rgb), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    responses: list[np.ndarray] = []
    for size in [7, 11, 15, 21]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        responses.append(cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel))
    canny = cv2.Canny(enhanced, threshold1=45, threshold2=130)
    combined = np.maximum.reduce(responses + [(0.35 * canny).astype(np.uint8)])
    combined = cv2.GaussianBlur(combined, (3, 3), 0)
    line_response = normalize_uint8(combined)
    return line_response, {
        "line_response_method": "clahe_blackhat_multi_kernel_plus_canny",
        "line_response_p90": float(np.percentile(line_response, 90)),
        "line_response_p95": float(np.percentile(line_response, 95)),
    }


def component_elongation(stats: np.ndarray, label_index: int) -> tuple[float, int, int, int]:
    area = int(stats[label_index, cv2.CC_STAT_AREA])
    width = int(stats[label_index, cv2.CC_STAT_WIDTH])
    height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
    elongation = float(max(width, height) / max(1, min(width, height)))
    return elongation, width, height, area


def hysteresis_line_linked_mask(
    probability: np.ndarray,
    line_response: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    dilation: int,
    close_kernel_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    strong = (probability >= high_threshold).astype(np.uint8)
    weak = (probability >= low_threshold).astype(np.uint8)
    line_cutoff = max(35.0, float(np.percentile(line_response, 86.0)))
    line_like = (line_response >= line_cutoff).astype(np.uint8)
    candidate = (weak & line_like).astype(np.uint8)

    strong_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    near_strong = cv2.dilate(strong, strong_dilate_kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    kept = np.zeros_like(candidate)
    kept_components = 0
    elongation_kept = 0
    near_strong_kept = 0
    for label_index in range(1, num_labels):
        component = labels == label_index
        elongation, width, height, area = component_elongation(stats, label_index)
        touches_strong = bool(np.any(near_strong[component] > 0))
        elongated = elongation >= 2.8 and max(width, height) >= 18 and area >= 8
        large_enough = area >= 18
        if (touches_strong and area >= 5) or (elongated and large_enough):
            kept[component] = 1
            kept_components += 1
            near_strong_kept += int(touches_strong)
            elongation_kept += int(elongated)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    linked = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    merged = np.maximum(strong, linked)
    if dilation > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        merged = cv2.dilate(merged, dilate_kernel, iterations=1)
    final = (merged > 0).astype(np.uint8) * 255
    return final, {
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "line_response_cutoff": float(line_cutoff),
        "dilation_kernel": int(dilation),
        "close_kernel": int(close_kernel_size),
        "candidate_ratio": mask_ratio(candidate * 255),
        "strong_ratio": mask_ratio(strong * 255),
        "kept_component_count": int(kept_components),
        "near_strong_kept": int(near_strong_kept),
        "elongation_kept": int(elongation_kept),
    }


def low_threshold_mask(probability: np.ndarray, line_response: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    low = (probability >= 0.15).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    low = cv2.dilate(low, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return low, {
        "low_threshold": 0.15,
        "line_response_used": False,
        "warning": "Recall-sensitive low threshold has higher false-positive risk.",
        "line_response_p90": float(np.percentile(line_response, 90)),
    }


def original_union_refined_mask(image_rgb: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cv_mask, cv_info, _ = build_cv_crack_mask(image_rgb, profile="notebook_v7_candidate", auto_invert=True)
    dl_mask = (probability >= 0.70).astype(np.uint8) * 255
    union = or_masks(dl_mask, cv_mask)
    if union is None:
        raise RuntimeError("Không tạo được union mask.")
    refined = refine_mask(ensure_gray_uint8(union), "repair_v3_conservative")
    return refined, {
        "threshold": 0.70,
        "mask_refine": "repair_v3_conservative",
        "cv_mask_ratio": mask_ratio(cv_mask),
        "dl_mask_ratio": mask_ratio(dl_mask),
        "union_mask_ratio": mask_ratio(union),
    }


def build_variants_for_checkpoint(
    image_rgb: np.ndarray,
    checkpoint_tag: str,
    checkpoint_path: Path,
    device: torch.device,
    line_response: np.ndarray,
    line_metadata: dict[str, Any],
) -> list[MaskVariant]:
    if not checkpoint_path.exists():
        print(f"skip {checkpoint_tag}: thiếu checkpoint {checkpoint_path}")
        return []
    probability, prob_metadata = infer_probability(image_rgb, checkpoint_path, device)
    original_mask, original_meta = original_union_refined_mask(image_rgb, probability)
    sensitive_mask, sensitive_meta = low_threshold_mask(probability, line_response)
    linked_mask, linked_meta = hysteresis_line_linked_mask(
        probability,
        line_response,
        low_threshold=0.15,
        high_threshold=0.50,
        dilation=3,
        close_kernel_size=3,
    )
    common_meta = {
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_path": str(checkpoint_path),
        **prob_metadata,
        **line_metadata,
    }
    return [
        MaskVariant(
            name=f"{checkpoint_tag}_original_union_refined",
            checkpoint_tag=checkpoint_tag,
            mask=original_mask,
            probability=probability,
            line_response=line_response,
            metadata={**common_meta, **original_meta, "variant_type": "original_union_refined"},
        ),
        MaskVariant(
            name=f"{checkpoint_tag}_sensitive_low_threshold",
            checkpoint_tag=checkpoint_tag,
            mask=sensitive_mask,
            probability=probability,
            line_response=line_response,
            metadata={**common_meta, **sensitive_meta, "variant_type": "sensitive_low_threshold"},
        ),
        MaskVariant(
            name=f"{checkpoint_tag}_hysteresis_line_linked",
            checkpoint_tag=checkpoint_tag,
            mask=linked_mask,
            probability=probability,
            line_response=line_response,
            metadata={**common_meta, **linked_meta, "variant_type": "hysteresis_line_linked"},
        ),
    ]


def run_hysteresis_sweep(
    checkpoint_tag: str,
    probability: np.ndarray,
    line_response: np.ndarray,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sweep_dir = output_dir / "hysteresis_sweep" / checkpoint_tag
    for low_threshold, high_threshold, dilation, close_kernel in HYSTERESIS_SWEEP:
        mask, metadata = hysteresis_line_linked_mask(
            probability,
            line_response,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            dilation=dilation,
            close_kernel_size=close_kernel,
        )
        tag = f"low{low_threshold:.2f}_high{high_threshold:.2f}_d{dilation}_c{close_kernel}".replace(".", "p")
        save_gray(sweep_dir / f"{tag}.png", mask)
        rows.append(
            {
                "checkpoint_tag": checkpoint_tag,
                "sweep_tag": tag,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "dilation": dilation,
                "close_kernel": close_kernel,
                "final_mask_ratio": mask_ratio(mask),
                **metadata,
            }
        )
    return rows


def inpaint_variant(image_rgb: np.ndarray, mask: np.ndarray, backend: str, simple_lama: Any | None) -> np.ndarray:
    if backend == "simple_lama":
        if simple_lama is None:
            simple_lama = try_create_simple_lama()
        return inpaint_with_simple_lama(image_rgb, mask, simple_lama)
    if backend == "opencv":
        return inpaint_with_opencv(image_rgb, mask)
    raise ValueError(f"Backend không hợp lệ: {backend}")


def copy_visual_assets(
    image_rgb: np.ndarray,
    variant: MaskVariant,
    backend: str,
    restored: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    variant_dir = output_dir / "variants" / variant.name / backend
    overlay = make_overlay(image_rgb, variant.mask)
    probability_vis = probability_to_uint8(variant.probability)
    line_vis = ensure_gray_uint8(variant.line_response)
    comparison = make_comparison_grid(
        [
            ("original", image_rgb),
            ("probability", make_gray_rgb(probability_vis, fallback_shape=image_rgb.shape[:2])),
            ("line response", make_gray_rgb(line_vis, fallback_shape=image_rgb.shape[:2])),
            ("final mask", make_gray_rgb(variant.mask, fallback_shape=image_rgb.shape[:2])),
            ("overlay", overlay),
            (f"restored {backend}", restored),
        ],
        max_columns=3,
    )
    save_rgb(variant_dir / "original.png", image_rgb)
    save_gray(variant_dir / "probability_score.png", probability_vis)
    save_gray(variant_dir / "line_response.png", line_vis)
    save_gray(variant_dir / "final_mask.png", variant.mask)
    save_rgb(variant_dir / "overlay_final.png", overlay)
    save_rgb(variant_dir / "restored_before_face.png", restored)
    save_rgb(variant_dir / "restored_final.png", restored)
    save_rgb(variant_dir / "comparison_grid.png", comparison)
    metadata = {
        **variant.metadata,
        "variant_name": variant.name,
        "backend": backend,
        "actual_backend": backend,
        "face_mode": "off",
        "final_mask_ratio": mask_ratio(variant.mask),
        "output_dir": str(variant_dir),
        "quality_status": "needs human visual review",
    }
    (variant_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def fit_tile(image_rgb: np.ndarray | None, label: str, size: tuple[int, int] = (340, 240)) -> np.ndarray:
    tile_w, tile_h = size
    canvas = np.full((tile_h, tile_w, 3), 246, dtype=np.uint8)
    if image_rgb is not None:
        image = ensure_rgb_uint8(image_rgb)
        height, width = image.shape[:2]
        scale = min(tile_w / max(width, 1), (tile_h - 34) / max(height, 1))
        resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        x0 = (tile_w - resized.shape[1]) // 2
        y0 = 34 + (tile_h - 34 - resized.shape[0]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (tile_w, 32), (24, 24, 24), thickness=-1)
    cv2.putText(canvas, label[:38], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def build_contact_sheets(output_dir: Path, variants: list[MaskVariant]) -> list[Path]:
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for checkpoint_tag in sorted({variant.checkpoint_tag for variant in variants}):
        selected = [variant for variant in variants if variant.checkpoint_tag == checkpoint_tag]
        columns = []
        for variant in selected:
            variant_root = output_dir / "variants" / variant.name
            overlay = read_rgb_or_none(variant_root / "simple_lama" / "overlay_final.png")
            restored_simple = read_rgb_or_none(variant_root / "simple_lama" / "restored_before_face.png")
            restored_opencv = read_rgb_or_none(variant_root / "opencv" / "restored_before_face.png")
            column = np.concatenate(
                [
                    fit_tile(overlay, f"{variant.name} overlay"),
                    fit_tile(restored_simple, "restored simple_lama"),
                    fit_tile(restored_opencv, "restored opencv"),
                ],
                axis=0,
            )
            columns.append(column)
        sheet = np.concatenate(columns, axis=1)
        output_path = contact_dir / f"{checkpoint_tag}_line_linking_contact.png"
        save_rgb(output_path, sheet)
        outputs.append(output_path)
    return outputs


def read_rgb_or_none(path: Path) -> np.ndarray | None:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def write_index(output_dir: Path, rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant_name",
        "checkpoint_tag",
        "variant_type",
        "backend",
        "actual_backend",
        "final_mask_ratio",
        "candidate_ratio",
        "strong_ratio",
        "kept_component_count",
        "near_strong_kept",
        "elongation_kept",
        "output_dir",
        "quality_status",
    ]
    with (output_dir / "experiment_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if sweep_rows:
        sweep_fields = sorted({key for row in sweep_rows for key in row.keys()})
        with (output_dir / "hysteresis_sweep_index.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sweep_fields)
            writer.writeheader()
            writer.writerows(sweep_rows)


def write_summary(output_dir: Path, rows: list[dict[str, Any]], contact_sheets: list[Path], probability_available: bool) -> None:
    variant_names = sorted({row["variant_name"] for row in rows})
    lines = [
        "# Demo3 Line-Linking Summary",
        "",
        f"- Output root: `{output_dir}`",
        f"- Probability map available: `{probability_available}`",
        f"- Mask variants: `{len(variant_names)}`",
        f"- Inpainting cases: `{len(rows)}`",
        "- Face restoration: `off` for all cases.",
        "- Quality decision: `not selected yet`; needs human visual review.",
        "",
        "## Contact Sheets",
        "",
    ]
    lines.extend(f"- `{path}`" for path in contact_sheets)
    lines.extend(
        [
            "",
            "## Visual Review Notes",
            "",
            "- `sensitive_low_threshold` should catch more weak crack pixels but has higher false-positive risk.",
            "- `hysteresis_line_linked` keeps weak line-like pixels only when near strong cracks or elongated, so it is the safer recall-sensitive candidate.",
            "- Check whether long thin cracks become more continuous without masking real image texture.",
            "- Do not pick a final mode until contact sheets are inspected.",
            "",
            "## Candidate Risk",
            "",
        ]
    )
    for row in rows:
        if row["backend"] != "simple_lama":
            continue
        lines.append(
            f"- `{row['variant_name']}`: mask_ratio={row.get('final_mask_ratio')}, "
            f"candidate_ratio={row.get('candidate_ratio', '')}, kept_components={row.get('kept_component_count', '')}."
        )
    lines.append("")
    (output_dir / "LINE_LINKING_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    image_path = find_demo3_image()
    image_rgb = load_rgb(image_path)
    device = resolve_device()
    line_response, line_metadata = build_line_response(image_rgb)
    save_rgb(OUTPUT_ROOT / "original.png", image_rgb)
    save_gray(OUTPUT_ROOT / "line_response.png", line_response)

    variants: list[MaskVariant] = []
    sweep_rows: list[dict[str, Any]] = []
    for checkpoint_tag, checkpoint_path in CHECKPOINTS.items():
        checkpoint_variants = build_variants_for_checkpoint(
            image_rgb,
            checkpoint_tag,
            checkpoint_path,
            device,
            line_response,
            line_metadata,
        )
        variants.extend(checkpoint_variants)
        for variant in checkpoint_variants:
            save_gray(OUTPUT_ROOT / "masks" / f"{variant.name}.png", variant.mask)
            save_gray(OUTPUT_ROOT / "probability" / f"{variant.checkpoint_tag}_probability.png", probability_to_uint8(variant.probability))
        if checkpoint_variants:
            sweep_rows.extend(run_hysteresis_sweep(checkpoint_tag, checkpoint_variants[0].probability, line_response, OUTPUT_ROOT))

    simple_lama = None
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for backend in BACKENDS:
            if backend == "simple_lama" and simple_lama is None:
                simple_lama = try_create_simple_lama()
            restored = inpaint_variant(image_rgb, variant.mask, backend, simple_lama)
            rows.append(copy_visual_assets(image_rgb, variant, backend, restored, OUTPUT_ROOT))

    contact_sheets = build_contact_sheets(OUTPUT_ROOT, variants)
    write_index(OUTPUT_ROOT, rows, sweep_rows)
    write_summary(OUTPUT_ROOT, rows, contact_sheets, probability_available=bool(variants))
    print(f"output_root: {OUTPUT_ROOT}")
    print(f"mask_variants: {len(variants)}")
    print(f"inpainting_cases: {len(rows)}")
    print(f"contact_sheets: {len(contact_sheets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
