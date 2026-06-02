from __future__ import annotations

import argparse
import csv
import json
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

from scripts.evaluate_real_segmentation import find_image_path, load_mask, load_model, load_rgb, resolve_device


DEFAULT_IDS = ["003", "005", "007", "014", "015", "016", "017", "018", "020", "027", "033", "034", "039", "040", "045"]
R011_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r011-repair-ft-s42" / "best_iou.ckpt"
R012_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r012-manual-repair-ft-s42" / "best_iou.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual review contact sheets cho r011 vs r012.")
    parser.add_argument("--data-root", default=r"F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq")
    parser.add_argument("--ids", default=",".join(DEFAULT_IDS))
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks_repair_manual")
    parser.add_argument("--r011-checkpoint", default=str(R011_CHECKPOINT))
    parser.add_argument("--r012-checkpoint", default=str(R012_CHECKPOINT))
    parser.add_argument("--output-dir", default="outputs/blueprint21_acceleration/r012_visual_review")
    parser.add_argument("--threshold-r011", type=float, default=0.50)
    parser.add_argument("--threshold-r012", type=float, default=0.50)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_ids(raw: str) -> list[str]:
    return [item.strip().zfill(3) for item in raw.split(",") if item.strip()]


def to_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).unsqueeze(0).to(device)


@torch.no_grad()
def predict_mask(model: torch.nn.Module | None, image_rgb: np.ndarray, threshold: float, device: torch.device) -> np.ndarray | None:
    if model is None:
        return None
    logits = model(to_tensor(image_rgb, device))
    probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return (probability >= threshold).astype(np.uint8) * 255


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return np.repeat(mask[:, :, None], 3, axis=2)


def overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    canvas = image_rgb.astype(np.float32).copy()
    layer = np.zeros_like(canvas)
    layer[:, :, 0] = color[0]
    layer[:, :, 1] = color[1]
    layer[:, :, 2] = color[2]
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
    return np.clip(canvas * (1.0 - alpha) + layer * alpha, 0, 255).astype(np.uint8)


def diff_rgb(pred_mask: np.ndarray | None, manual_mask: np.ndarray, image_size: int) -> np.ndarray:
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    if pred_mask is None:
        canvas[:] = 32
        return canvas
    pred = pred_mask > 127
    gt = manual_mask > 127
    true_positive = pred & gt
    false_positive = pred & ~gt
    false_negative = ~pred & gt
    canvas[true_positive] = (255, 255, 255)
    canvas[false_positive] = (255, 0, 0)
    canvas[false_negative] = (0, 160, 255)
    return canvas


def metrics(pred_mask: np.ndarray | None, manual_mask: np.ndarray) -> dict[str, float | None]:
    if pred_mask is None:
        return {"iou": None, "f1": None, "precision": None, "recall": None}
    pred = pred_mask > 127
    gt = manual_mask > 127
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


def draw_label(image_rgb: np.ndarray, text: str) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image_bgr, (0, 0), (image_bgr.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(image_bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def make_contact_sheet(
    image_rgb: np.ndarray,
    manual_mask: np.ndarray,
    r011_mask: np.ndarray | None,
    r012_mask: np.ndarray | None,
    sample_id: str,
) -> np.ndarray:
    missing = np.full_like(image_rgb, 32)
    panels = [
        draw_label(image_rgb, f"{sample_id} original"),
        draw_label(mask_to_rgb(manual_mask), "manual repair mask"),
        draw_label(overlay(image_rgb, manual_mask), "manual overlay"),
        draw_label(mask_to_rgb(r011_mask) if r011_mask is not None else missing, "r011 predicted mask"),
        draw_label(mask_to_rgb(r012_mask) if r012_mask is not None else missing, "r012 predicted mask"),
        draw_label(diff_rgb(r011_mask, manual_mask, image_rgb.shape[0]), "r011 vs manual diff"),
        draw_label(diff_rgb(r012_mask, manual_mask, image_rgb.shape[0]), "r012 vs manual diff"),
    ]
    top = np.concatenate(panels[:4], axis=1)
    bottom = np.concatenate(panels[4:] + [np.zeros_like(image_rgb)], axis=1)
    return np.concatenate([top, bottom], axis=0)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Không ghi được ảnh: {path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "image_path",
        "manual_mask_path",
        "contact_sheet_path",
        "manual_positive_ratio",
        "r011_iou",
        "r011_f1",
        "r012_iou",
        "r012_f1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace, r011_exists: bool, r012_exists: bool) -> None:
    r011_ious = [float(row["r011_iou"]) for row in rows if row["r011_iou"] != ""]
    r012_ious = [float(row["r012_iou"]) for row in rows if row["r012_iou"] != ""]
    lines = [
        "# r012 Visual Review",
        "",
        f"- Data root: `{args.data_root}`",
        f"- IDs: `{args.ids}`",
        f"- r011 checkpoint exists: `{r011_exists}`",
        f"- r012 checkpoint exists: `{r012_exists}`",
        f"- Threshold r011/r012: `{args.threshold_r011}` / `{args.threshold_r012}`",
        f"- Contact sheets: `{path.parent}`",
        "",
        "## Metric tóm tắt trên 15 manual masks",
        "",
        f"- r011 mean IoU: `{sum(r011_ious) / len(r011_ious) if r011_ious else None}`",
        f"- r012 mean IoU: `{sum(r012_ious) / len(r012_ious) if r012_ious else None}`",
        "",
        "Ghi chú: đây là visual/QC review trên tập rất nhỏ, không đủ để claim r012 vượt trội.",
        "",
        "| id | manual_positive_ratio | r011_iou | r012_iou | contact_sheet |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['manual_positive_ratio']} | {row['r011_iou']} | {row['r012_iou']} | `{row['contact_sheet_path']}` |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    device = resolve_device(args.device)
    r011_checkpoint = resolve_path(args.r011_checkpoint)
    r012_checkpoint = resolve_path(args.r012_checkpoint)
    model_r011 = load_model(r011_checkpoint, device) if r011_checkpoint.exists() else None
    model_r012 = load_model(r012_checkpoint, device) if r012_checkpoint.exists() else None

    rows: list[dict[str, Any]] = []
    for sample_id in parse_ids(args.ids):
        image_path = find_image_path(data_root / args.image_dir, sample_id)
        mask_path = data_root / args.mask_dir / f"{sample_id}_mask.png"
        image_rgb = cv2.resize(load_rgb(image_path), (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        manual_mask = cv2.resize(load_mask(mask_path), (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
        r011_mask = predict_mask(model_r011, image_rgb, args.threshold_r011, device)
        r012_mask = predict_mask(model_r012, image_rgb, args.threshold_r012, device)
        sheet = make_contact_sheet(image_rgb, manual_mask, r011_mask, r012_mask, sample_id)
        contact_path = output_dir / "contact_sheets" / f"{sample_id}_r011_vs_r012_review.png"
        save_rgb(contact_path, sheet)
        r011_metrics = metrics(r011_mask, manual_mask)
        r012_metrics = metrics(r012_mask, manual_mask)
        rows.append(
            {
                "id": sample_id,
                "image_path": str(image_path),
                "manual_mask_path": str(mask_path),
                "contact_sheet_path": str(contact_path),
                "manual_positive_ratio": f"{float(np.count_nonzero(manual_mask > 127) / manual_mask.size):.6f}",
                "r011_iou": "" if r011_metrics["iou"] is None else f"{float(r011_metrics['iou']):.6f}",
                "r011_f1": "" if r011_metrics["f1"] is None else f"{float(r011_metrics['f1']):.6f}",
                "r012_iou": "" if r012_metrics["iou"] is None else f"{float(r012_metrics['iou']):.6f}",
                "r012_f1": "" if r012_metrics["f1"] is None else f"{float(r012_metrics['f1']):.6f}",
            }
        )

    write_csv(output_dir / "review_index.csv", rows)
    write_summary(output_dir / "REVIEW_SUMMARY.md", rows, args, r011_checkpoint.exists(), r012_checkpoint.exists())
    payload = {"rows": rows, "r011_checkpoint": str(r011_checkpoint), "r012_checkpoint": str(r012_checkpoint)}
    (output_dir / "review_index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"contact_sheets: {output_dir / 'contact_sheets'}")
    print(f"review_index: {output_dir / 'review_index.csv'}")
    print(f"summary: {output_dir / 'REVIEW_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
