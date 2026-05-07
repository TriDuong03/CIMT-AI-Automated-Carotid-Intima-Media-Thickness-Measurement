"""
Chia dữ liệu thành train/val/test theo tỉ lệ 7:2:1.
Dùng symlink để không tốn dung lượng.

Chiến lược:
  - Test  : 100 ảnh tech_401→tech_500, mask lấy từ GT_masks/ (KHÔNG crop)
  - Train/Val: 821 ảnh còn lại, chia 7:2 (tech crop theo rect, clin dùng symlink)

Cấu trúc output:
  data/
  ├── train/images/  train/masks/
  ├── val/images/    val/masks/
  └── test/images/   test/masks/
"""

import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR    = PROJECT_DIR / "data"
TECH_DIR    = PROJECT_DIR / "DATASET_CUBS_tech"
GT_MASK_DIR = TECH_DIR / "GT_masks"
RECT_DIR    = TECH_DIR / "LIMA-Profiles" / "Manual-A1"
SEED        = 42


def read_rect(stem: str) -> tuple:
    """Đọc file _rect.txt → (x, y, w, h) dạng int."""
    x, y, w, h = map(float, (RECT_DIR / f"{stem}_rect.txt").read_text().split())
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def crop_and_save(src_img: Path, src_mask: Path, dst_img: Path, dst_mask: Path, stem: str):
    """Crop ảnh và mask theo rect rồi lưu file thực."""
    x, y, w, h = read_rect(stem)
    img  = np.array(Image.open(src_img))
    mask = np.array(Image.open(src_mask))
    Image.fromarray(img [y:y+h, x:x+w]).save(dst_img)
    Image.fromarray(mask[y:y+h, x:x+w]).save(dst_mask)


def place(stem: str, split: str, img_dir: Path, mask_dir: Path,
          gt_mask_dir: Path = None, crop: bool = False):
    """Đặt một cặp ảnh-mask vào thư mục split."""
    img_out  = DATA_DIR / split / "images"
    mask_out = DATA_DIR / split / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    src_img  = img_dir  / f"{stem}.tiff"
    dst_img  = img_out  / f"{stem}.tiff"
    dst_mask = mask_out / f"{stem}_mask.png"

    if dst_img.exists() or dst_img.is_symlink():
        return

    # Chọn nguồn mask
    if gt_mask_dir and (gt_mask_dir / f"{stem}_mask.png").exists():
        src_mask = gt_mask_dir / f"{stem}_mask.png"
    else:
        src_mask = mask_dir / f"{stem}_mask.png"

    if crop and stem.startswith("tech"):
        crop_and_save(src_img.resolve(), src_mask.resolve(), dst_img, dst_mask, stem)
    else:
        dst_img.symlink_to(src_img.resolve())
        dst_mask.symlink_to(src_mask.resolve())


def clear_split(split: str):
    split_dir = DATA_DIR / split
    if split_dir.exists():
        shutil.rmtree(split_dir)


def main():
    img_dir  = DATA_DIR / "images"
    mask_dir = DATA_DIR / "masks"

    # ── Test: cố định = 100 ảnh có GT_masks (tech_401→tech_500) ──────────────
    test_stems = sorted(
        p.stem.replace("_mask", "")
        for p in GT_MASK_DIR.glob("*_mask.png")
    )

    # ── Train/Val: toàn bộ ảnh còn lại ───────────────────────────────────────
    all_stems    = sorted(p.stem for p in img_dir.glob("*.tiff"))
    trainval     = [s for s in all_stems if s not in set(test_stems)]

    random.seed(SEED)
    random.shuffle(trainval)
    n_train  = int(len(trainval) * 7 / 9)      # 7 phần trong 7+2
    train_stems = trainval[:n_train]
    val_stems   = trainval[n_train:]

    total = len(train_stems) + len(val_stems) + len(test_stems)
    print(f"Tổng : {total}")
    print(f"Train: {len(train_stems)} ({len(train_stems)/total*100:.1f}%)")
    print(f"Val  : {len(val_stems)}  ({len(val_stems)/total*100:.1f}%)")
    print(f"Test : {len(test_stems)}  ({len(test_stems)/total*100:.1f}%)  ← GT_masks")

    # ── Xóa và tạo lại các split folder ──────────────────────────────────────
    for split in ("train", "val", "test"):
        clear_split(split)
        print(f"\nĐang tạo {split}/...")

    for stem in train_stems:
        place(stem, "train", img_dir, mask_dir)
    for stem in val_stems:
        place(stem, "val",   img_dir, mask_dir)
    for stem in test_stems:
        place(stem, "test",  img_dir, mask_dir, gt_mask_dir=GT_MASK_DIR)

    print(f"\nXong! Thư mục output: {DATA_DIR}")
    for split in ("train", "val", "test"):
        n_img  = len(list((DATA_DIR / split / "images").glob("*")))
        n_mask = len(list((DATA_DIR / split / "masks").glob("*")))
        print(f"  {split:5s}: images={n_img}  masks={n_mask}")


if __name__ == "__main__":
    main()
