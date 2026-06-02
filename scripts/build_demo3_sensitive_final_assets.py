from __future__ import annotations

import json
import shutil
import sys
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
    build_inference_tensor,
    inpaint_with_simple_lama,
    load_model,
    load_rgb,
    make_comparison_grid,
    make_overlay,
    mask_ratio,
    probability_to_uint8,
    resize_probability_mask,
    save_gray,
    save_rgb,
    try_create_simple_lama,
)
from src.restoration.face_restoration import apply_face_restoration  # noqa: E402


DEMO_ROOT = PROJECT_ROOT / "data" / "demo_inputs" / "real_manual_3"
EXPERIMENT_ROOT = PROJECT_ROOT / "outputs" / "final_demo_benchmark" / "demo3_line_linking_experiment"
SOURCE_BASELINE = EXPERIMENT_ROOT / "variants" / "r011_original_union_refined" / "simple_lama"
SOURCE_SENSITIVE = EXPERIMENT_ROOT / "variants" / "r011_sensitive_low_threshold" / "simple_lama"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_demo_candidate" / "demo3_sensitive_r011"
SANITY_ROOT = PROJECT_ROOT / "outputs" / "final_demo_candidate" / "sensitive_sanity_check"
R011_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r011-repair-ft-s42" / "best_iou.ckpt"


def find_demo_image(demo_id: str) -> Path:
    for extension in [".jpg", ".png", ".jpeg"]:
        path = DEMO_ROOT / f"{demo_id}{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Không tìm thấy {demo_id} trong {DEMO_ROOT}")


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Thiếu source asset: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def fit_tile(image_rgb: np.ndarray, label: str, size: tuple[int, int] = (430, 310)) -> np.ndarray:
    tile_w, tile_h = size
    canvas = np.full((tile_h, tile_w, 3), 246, dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    scale = min(tile_w / max(width, 1), (tile_h - 34) / max(height, 1))
    resized = cv2.resize(
        image_rgb,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x0 = (tile_w - resized.shape[1]) // 2
    y0 = 34 + (tile_h - 34 - resized.shape[0]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (tile_w, 32), (24, 24, 24), thickness=-1)
    cv2.putText(canvas, label[:42], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def build_final_grid(codeformer_available: bool) -> None:
    tiles = [
        fit_tile(read_rgb(OUTPUT_ROOT / "original.png"), "Original"),
        fit_tile(read_rgb(OUTPUT_ROOT / "baseline_overlay.png"), "Baseline mask overlay"),
        fit_tile(read_rgb(OUTPUT_ROOT / "sensitive_overlay.png"), "Sensitive mask overlay"),
        fit_tile(read_rgb(OUTPUT_ROOT / "baseline_restored_before_face.png"), "Baseline restored"),
        fit_tile(read_rgb(OUTPUT_ROOT / "sensitive_restored_before_face.png"), "Sensitive restored"),
    ]
    if codeformer_available:
        tiles.append(fit_tile(read_rgb(OUTPUT_ROOT / "sensitive_codeformer_final.png"), "Sensitive + CodeFormer"))
    first_row = np.concatenate(tiles[:3], axis=1)
    second_row_tiles = tiles[3:]
    while len(second_row_tiles) < 3:
        second_row_tiles.append(np.full_like(tiles[0], 250))
    second_row = np.concatenate(second_row_tiles[:3], axis=1)
    save_rgb(OUTPUT_ROOT / "demo3_final_comparison_grid.png", np.concatenate([first_row, second_row], axis=0))


@torch.no_grad()
def infer_probability(image_rgb: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(R011_CHECKPOINT, device)
    tensor = build_inference_tensor(image_rgb, 512).to(device)
    probability = torch.sigmoid(model(tensor)).squeeze().detach().cpu().numpy().astype(np.float32)
    probability = resize_probability_mask(probability, image_rgb.shape[:2])
    return np.clip(probability, 0.0, 1.0)


def build_sensitive_mask(probability: np.ndarray) -> np.ndarray:
    mask = (probability >= 0.15).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return mask


def build_sanity_case(demo_id: str, simple_lama: Any) -> dict[str, Any]:
    image_path = find_demo_image(demo_id)
    image_rgb = load_rgb(image_path)
    probability = infer_probability(image_rgb)
    mask = build_sensitive_mask(probability)
    overlay = make_overlay(image_rgb, mask)
    restored = inpaint_with_simple_lama(image_rgb, mask, simple_lama)
    output_dir = SANITY_ROOT / demo_id / "r011_sensitive_low_threshold_simple_lama"
    save_rgb(output_dir / "original.png", image_rgb)
    save_gray(output_dir / "probability_score.png", probability_to_uint8(probability))
    save_gray(output_dir / "final_mask.png", mask)
    save_rgb(output_dir / "overlay_final.png", overlay)
    save_rgb(output_dir / "restored_before_face.png", restored)
    comparison = make_comparison_grid(
        [
            ("original", image_rgb),
            ("sensitive mask", np.repeat(mask[:, :, None], 3, axis=2)),
            ("overlay", overlay),
            ("restored", restored),
        ],
        max_columns=4,
    )
    save_rgb(output_dir / "comparison_grid.png", comparison)
    metadata = {
        "demo_id": demo_id,
        "variant": "r011_sensitive_low_threshold",
        "backend": "simple_lama",
        "face_mode": "off",
        "low_threshold": 0.15,
        "final_mask_ratio": mask_ratio(mask),
        "warning": "Sanity check only; inspect for false positives before generalizing.",
        "output_dir": str(output_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def write_summary(
    baseline_metadata: dict[str, Any],
    sensitive_metadata: dict[str, Any],
    codeformer_metadata: dict[str, Any] | None,
    sanity_rows: list[dict[str, Any]],
) -> None:
    baseline_ratio = baseline_metadata.get("final_mask_ratio")
    sensitive_ratio = sensitive_metadata.get("final_mask_ratio")
    codeformer_text = "available" if codeformer_metadata and codeformer_metadata.get("face_restoration_applied") else "not applied"
    lines = [
        "# Demo3 Sensitive R011 Final Candidate",
        "",
        "## Context",
        "",
        "Demo3 is difficult because long, thin cracks are partially detected as sparse dots in the baseline mask.",
        "The selected candidate uses a recall-sensitive lower threshold to retain more weak crack pixels.",
        "",
        "## Selected Candidate",
        "",
        "- Segmentation/checkpoint: `r011`",
        "- Mask: `r011_sensitive_low_threshold`",
        "- Inpainting: `simple_lama`",
        "- Face restoration: optional CodeFormer case only, fidelity `0.7`",
        "- Project default: unchanged.",
        "",
        "## Mask Ratio",
        "",
        f"- Baseline r011 original union refined: `{baseline_ratio}`",
        f"- Sensitive low threshold: `{sensitive_ratio}`",
        "",
        "## Human Review Finding",
        "",
        "User visual review selected `r011_sensitive_low_threshold + simple_lama` as the best current demo3 result.",
        "This should be treated as a demo3-specific candidate, not a global claim.",
        "",
        "## Risk",
        "",
        "The sensitive mask may increase false positives on other images because it keeps more weak pixels.",
        "Sanity checks on demo1/demo2 are provided for review, but do not establish a default for the whole project.",
        "",
        "## CodeFormer",
        "",
        f"- Sensitive + CodeFormer status: `{codeformer_text}`",
        "",
        "## Sanity Check",
        "",
    ]
    if sanity_rows:
        for row in sanity_rows:
            lines.append(f"- `{row['demo_id']}` sensitive mask ratio: `{row['final_mask_ratio']}`")
    else:
        lines.append("- Not run.")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Use this as the current demo3 candidate after visual review.",
            "Do not change the project-wide default yet.",
            "Do not claim the method is best on all old photos.",
            "",
        ]
    )
    (OUTPUT_ROOT / "DEMO3_SENSITIVE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not SOURCE_BASELINE.exists() or not SOURCE_SENSITIVE.exists():
        raise FileNotFoundError("Thiếu output line-linking experiment cho baseline hoặc sensitive.")

    copy_file(SOURCE_BASELINE / "original.png", OUTPUT_ROOT / "original.png")
    copy_file(SOURCE_BASELINE / "final_mask.png", OUTPUT_ROOT / "baseline_mask.png")
    copy_file(SOURCE_BASELINE / "overlay_final.png", OUTPUT_ROOT / "baseline_overlay.png")
    copy_file(SOURCE_BASELINE / "restored_before_face.png", OUTPUT_ROOT / "baseline_restored_before_face.png")
    copy_file(SOURCE_BASELINE / "metadata.json", OUTPUT_ROOT / "metadata_baseline.json")

    copy_file(SOURCE_SENSITIVE / "final_mask.png", OUTPUT_ROOT / "sensitive_mask.png")
    copy_file(SOURCE_SENSITIVE / "overlay_final.png", OUTPUT_ROOT / "sensitive_overlay.png")
    copy_file(SOURCE_SENSITIVE / "restored_before_face.png", OUTPUT_ROOT / "sensitive_restored_before_face.png")
    copy_file(SOURCE_SENSITIVE / "metadata.json", OUTPUT_ROOT / "metadata_sensitive.json")

    baseline_metadata = load_json(OUTPUT_ROOT / "metadata_baseline.json")
    sensitive_metadata = load_json(OUTPUT_ROOT / "metadata_sensitive.json")

    codeformer_metadata: dict[str, Any] | None = None
    sensitive_rgb = read_rgb(OUTPUT_ROOT / "sensitive_restored_before_face.png")
    codeformer_rgb, face_metadata = apply_face_restoration(
        sensitive_rgb,
        mode="codeformer_if_available",
        strength=0.7,
        output_dir=OUTPUT_ROOT / "face_module",
    )
    if face_metadata.get("face_restoration_applied"):
        save_rgb(OUTPUT_ROOT / "sensitive_codeformer_final.png", codeformer_rgb)
        codeformer_metadata = {
            **sensitive_metadata,
            "face_mode": "codeformer_if_available",
            "face_restoration_applied": True,
            "face_backend": "codeformer",
            "codeformer_fidelity": face_metadata.get("codeformer_fidelity"),
            "codeformer_output": face_metadata.get("codeformer_output"),
        }
        (OUTPUT_ROOT / "metadata_sensitive_codeformer.json").write_text(
            json.dumps(codeformer_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        codeformer_metadata = {
            **sensitive_metadata,
            "face_mode": "codeformer_if_available",
            "face_restoration_applied": False,
            "face_reason": face_metadata.get("reason"),
            "face_warning": face_metadata.get("warning"),
        }
        (OUTPUT_ROOT / "metadata_sensitive_codeformer.json").write_text(
            json.dumps(codeformer_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    build_final_grid(codeformer_available=(OUTPUT_ROOT / "sensitive_codeformer_final.png").exists())

    simple_lama = try_create_simple_lama()
    sanity_rows = [build_sanity_case("demo1", simple_lama), build_sanity_case("demo2", simple_lama)]
    write_summary(baseline_metadata, sensitive_metadata, codeformer_metadata, sanity_rows)

    print(f"output_root: {OUTPUT_ROOT}")
    print(f"comparison_grid: {OUTPUT_ROOT / 'demo3_final_comparison_grid.png'}")
    print(f"baseline_mask_ratio: {baseline_metadata.get('final_mask_ratio')}")
    print(f"sensitive_mask_ratio: {sensitive_metadata.get('final_mask_ratio')}")
    print(f"codeformer_applied: {codeformer_metadata.get('face_restoration_applied') if codeformer_metadata else False}")
    print(f"sanity_root: {SANITY_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
