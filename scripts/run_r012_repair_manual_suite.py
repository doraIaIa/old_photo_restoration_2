from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
R011_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "seg-unet-attn-r011-repair-ft-s42" / "best_iou.ckpt"
R012_RUN_ID = "seg-unet-attn-r012-manual-repair-ft-s42"
R012_DIR = PROJECT_ROOT / "checkpoints" / "segmenter" / R012_RUN_ID
R012_CHECKPOINT = R012_DIR / "best_iou.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe suite cho r012 manual repair-mask fine-tune.")
    parser.add_argument("--data-root", default=r"F:\deeplearning\old_photo_mask\old_photo_pairs_10_hq")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks_repair_manual")
    parser.add_argument("--output-dir", default="outputs/blueprint21_acceleration/r012_manual_repair")
    parser.add_argument("--min-masks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute-train", action="store_true", help="Mặc định chỉ chuẩn bị command; bật flag này mới train thật.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def mask_id(path: Path) -> str:
    name = path.stem
    return name[:-5] if name.endswith("_mask") else name


def find_image(images_dir: Path, sample_id: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{sample_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def valid_manual_ids(data_root: Path, image_dir: str, mask_dir: str) -> list[str]:
    images_dir = data_root / image_dir
    masks_dir = data_root / mask_dir
    ids: list[str] = []
    for mask_path in sorted(masks_dir.glob("*_mask.png")):
        sample_id = mask_id(mask_path)
        if find_image(images_dir, sample_id) is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        unique_values = set(int(value) for value in np.unique(mask))
        if not unique_values.issubset({0, 255}):
            continue
        if np.count_nonzero(mask > 127) == 0:
            continue
        ids.append(sample_id)
    return ids


def write_splits(split_dir: Path, ids: list[str], seed: int) -> dict[str, list[str]]:
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_val = max(1, round(n_total * 0.1))
    n_test = max(1, round(n_total * 0.1))
    n_train = max(1, n_total - n_val - n_test)
    splits = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, split_ids in splits.items():
        (split_dir / f"{name}.txt").write_text("\n".join(split_ids) + "\n", encoding="utf-8")
    return splits


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run_command(command: list[str], label: str) -> None:
    print(f"\n[{label}] {command_text(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Lệnh `{label}` lỗi với exit code {result.returncode}.")


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ids = valid_manual_ids(data_root, args.image_dir, args.mask_dir)
    status: dict[str, Any] = {
        "data_root": str(data_root),
        "mask_dir": str(data_root / args.mask_dir),
        "valid_manual_masks": len(ids),
        "min_masks_required": args.min_masks,
        "r011_checkpoint": str(R011_CHECKPOINT),
        "r011_checkpoint_exists": R011_CHECKPOINT.exists(),
        "r012_checkpoint": str(R012_CHECKPOINT),
        "r012_checkpoint_exists": R012_CHECKPOINT.exists(),
        "executed_train": False,
        "commands": [],
    }

    if len(ids) < args.min_masks:
        missing = args.min_masks - len(ids)
        status["ready_to_train"] = False
        status["message"] = f"Chưa đủ manual repair masks hợp lệ để train r012. Cần thêm ít nhất {missing} mask."
        (output_dir / "r012_manual_repair_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(status["message"])
        print(f"valid_manual_masks: {len(ids)}")
        print(f"status_json: {output_dir / 'r012_manual_repair_status.json'}")
        return 0

    split_dir = output_dir / "splits"
    splits = write_splits(split_dir, ids, args.seed)
    train_command = [
        sys.executable,
        "scripts\\finetune_real_segmentation.py",
        "--data-root",
        str(data_root),
        "--image-dir",
        args.image_dir,
        "--mask-dir",
        args.mask_dir,
        "--split-dir",
        str(split_dir),
        "--init-checkpoint",
        str(R011_CHECKPOINT),
        "--run-id",
        R012_RUN_ID,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    status["ready_to_train"] = True
    status["splits"] = splits
    status["commands"].append({"label": "train_r012", "command": train_command})

    if args.execute_train:
        if not R011_CHECKPOINT.exists():
            raise FileNotFoundError(f"Không tìm thấy r011 init checkpoint: {R011_CHECKPOINT}")
        run_command(train_command, "train_r012")
        status["executed_train"] = True
    else:
        print("Dry-run: thêm --execute-train nếu muốn train r012 thật.")
        print(f"train_command: {command_text(train_command)}")

    if R012_CHECKPOINT.exists():
        for split_name in ["val", "test"]:
            eval_command = [
                sys.executable,
                "scripts\\evaluate_real_segmentation.py",
                "--data-root",
                str(data_root),
                "--image-dir",
                args.image_dir,
                "--mask-dir",
                args.mask_dir,
                "--split-file",
                str(split_dir / f"{split_name}.txt"),
                "--checkpoint",
                str(R012_CHECKPOINT),
                "--output-dir",
                str(output_dir / "eval" / split_name),
                "--device",
                args.device,
            ]
            status["commands"].append({"label": f"eval_r012_{split_name}", "command": eval_command})
            if args.execute_train:
                run_command(eval_command, f"eval_r012_{split_name}")
    else:
        status["comparison_grid_message"] = "Chưa có r012 checkpoint nên chưa sinh comparison grid r011 vs r012."

    status["r012_checkpoint_exists"] = R012_CHECKPOINT.exists()
    (output_dir / "r012_manual_repair_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"status_json: {output_dir / 'r012_manual_repair_status.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
