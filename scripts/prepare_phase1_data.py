#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    return argparse.ArgumentParser(
        description="Chuẩn bị subset dữ liệu cho Phase 1."
    ).parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_div2k_subset(src_dir: Path, dst_dir: Path, start_id: int, end_id: int) -> int:
    ensure_dir(dst_dir)
    copied = 0
    for image_id in range(start_id, end_id + 1):
        file_name = f"{image_id:04d}.png"
        src_path = src_dir / file_name
        dst_path = dst_dir / file_name
        if not src_path.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh DIV2K: {src_path}")
        shutil.copy2(src_path, dst_path)
        copied += 1
    return copied


def count_png_files(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png")


def run_git_clone(repo_url: str, target_dir: Path) -> None:
    ensure_dir(target_dir.parent)
    cmd = ["git", "clone", "--depth", "1", repo_url, str(target_dir)]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "Clone CrackForest thất bại.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def locate_crack_repo(project_root: Path, external_repo_dir: Path) -> tuple[Path, bool]:
    candidate_names = [
        "CrackForest-dataset",
        "temp_crackforest",
        "crackforest",
        "crackforest-dataset",
    ]

    for name in candidate_names:
        candidate = project_root / name
        if candidate.is_dir():
            return candidate, True

    if external_repo_dir.is_dir():
        return external_repo_dir, False

    run_git_clone("https://github.com/cuilimeng/CrackForest-dataset", external_repo_dir)
    return external_repo_dir, False


def find_crack_images(repo_dir: Path) -> list[Path]:
    image_files = [
        path
        for path in repo_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    image_files.sort(key=lambda path: str(path.relative_to(repo_dir)).lower())
    return image_files


def copy_crack_images(repo_dir: Path, crack_raw_dir: Path, limit: int = 20) -> int:
    ensure_dir(crack_raw_dir)
    crack_images = find_crack_images(repo_dir)
    if not crack_images:
        raise FileNotFoundError(f"Không tìm thấy ảnh crack .png/.jpg trong repo: {repo_dir}")

    selected = crack_images[:limit]
    for index, src_path in enumerate(selected, start=1):
        extension = src_path.suffix.lower()
        dst_path = crack_raw_dir / f"crack_{index:03d}{extension}"
        shutil.copy2(src_path, dst_path)
    return len(selected)


def move_repo_out_of_project(repo_dir: Path, external_repo_dir: Path, project_root: Path) -> Path | None:
    try:
        repo_dir.relative_to(project_root)
    except ValueError:
        return repo_dir

    if external_repo_dir.exists():
        raise FileExistsError(
            "Thư mục đích CrackForest đã tồn tại, cần hỏi lại trước khi ghi đè: "
            f"{external_repo_dir}"
        )

    ensure_dir(external_repo_dir.parent)
    shutil.move(str(repo_dir), str(external_repo_dir))
    return external_repo_dir


def main() -> int:
    parse_args()
    project_root = Path(__file__).resolve().parents[1]

    train_src = Path(r"F:\deeplearning\DIV2K_train_HR\DIV2K_train_HR")
    val_src = Path(r"F:\deeplearning\DIV2K_valid_HR\DIV2K_valid_HR")
    train_dst = project_root / "data" / "clean" / "div2k" / "train"
    val_dst = project_root / "data" / "clean" / "div2k" / "val"
    crack_raw_dir = project_root / "data" / "crack_bank" / "raw"
    external_repo_dir = Path(r"F:\deeplearning\_external_datasets\CrackForest-dataset")

    div2k_train_copied = copy_div2k_subset(train_src, train_dst, 1, 50)
    div2k_val_copied = copy_div2k_subset(val_src, val_dst, 801, 810)

    train_png_count = count_png_files(train_dst)
    val_png_count = count_png_files(val_dst)
    if train_png_count != 50:
        raise RuntimeError(f"Số ảnh train không đúng: kỳ vọng 50, thực tế {train_png_count}")
    if val_png_count != 10:
        raise RuntimeError(f"Số ảnh val không đúng: kỳ vọng 10, thực tế {val_png_count}")

    crack_repo_dir, repo_inside_project = locate_crack_repo(project_root, external_repo_dir)
    crack_copied = copy_crack_images(crack_repo_dir, crack_raw_dir, limit=20)
    repo_new_location = move_repo_out_of_project(
        crack_repo_dir,
        external_repo_dir,
        project_root,
    )

    print(f"DIV2K train copied: {div2k_train_copied}")
    print(f"DIV2K val copied: {div2k_val_copied}")
    print(f"DIV2K train png count: {train_png_count}")
    print(f"DIV2K val png count: {val_png_count}")
    print(f"Crack images copied: {crack_copied}")
    if repo_inside_project or repo_new_location == external_repo_dir:
        print(f"CrackForest repo location: {external_repo_dir}")
    elif repo_new_location is not None:
        print(f"CrackForest repo location: {repo_new_location}")
    else:
        print("CrackForest repo location: không thay đổi")

    return 0


if __name__ == "__main__":
    sys.exit(main())
