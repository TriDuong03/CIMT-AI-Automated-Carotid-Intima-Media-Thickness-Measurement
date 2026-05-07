import os
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_profile(path: str) -> np.ndarray:
    coords = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = map(float, line.split())
                coords.append((x, y))
    return np.array(coords)


def make_mask_from_profiles(li_path: str, ma_path: str, img_shape: tuple) -> np.ndarray:
    li = load_profile(li_path)
    ma = load_profile(ma_path)

    x_min = int(np.ceil(max(li[:, 0].min(), ma[:, 0].min())))
    x_max = int(np.floor(min(li[:, 0].max(), ma[:, 0].max())))
    x_cols = np.arange(x_min, x_max + 1)

    li_y = np.interp(x_cols, li[:, 0], li[:, 1])
    ma_y = np.interp(x_cols, ma[:, 0], ma[:, 1])

    H, W = img_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for i, x in enumerate(x_cols):
        y_top = max(0, min(int(round(li_y[i])), H - 1))
        y_bot = max(0, min(int(round(ma_y[i])), H - 1))
        mask[y_top:y_bot + 1, x] = 1

    return mask


def generate_all_masks(image_dir: str, profile_dir: str, output_dir: str, ext: str = ".tiff"):
    os.makedirs(output_dir, exist_ok=True)

    image_ids = sorted(
        Path(f).stem
        for f in os.listdir(image_dir)
        if f.endswith(ext)
    )

    skipped = []
    for image_id in tqdm(image_ids, desc="Generating masks"):
        li_path = os.path.join(profile_dir, f"{image_id}-LI.txt")
        ma_path = os.path.join(profile_dir, f"{image_id}-MA.txt")

        if not os.path.exists(li_path) or not os.path.exists(ma_path):
            skipped.append(image_id)
            continue

        img_shape = np.array(Image.open(os.path.join(image_dir, f"{image_id}{ext}"))).shape[:2]
        mask = make_mask_from_profiles(li_path, ma_path, img_shape)

        out_path = os.path.join(output_dir, f"{image_id}_mask.png")
        Image.fromarray(mask * 255).save(out_path)

    print(f"\nDone: {len(image_ids) - len(skipped)}/{len(image_ids)} masks saved to {output_dir}")
    if skipped:
        print(f"Skipped ({len(skipped)}): {skipped[:10]}{'...' if len(skipped) > 10 else ''}")


if __name__ == "__main__":
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    dataset_dir  = cfg["dataset_dir"]
    image_dir    = os.path.join(dataset_dir, cfg["image_subdir"])
    profile_dir  = os.path.join(dataset_dir, cfg["profile_subdir"])
    output_dir   = cfg["output_dir"] or os.path.join(dataset_dir, "MASKS")
    ext          = cfg.get("ext", ".tiff")

    print(f"Dataset   : {dataset_dir}")
    print(f"Images    : {image_dir}")
    print(f"Profiles  : {profile_dir}")
    print(f"Output    : {output_dir}\n")

    for d, label in [(image_dir, "image_subdir"), (profile_dir, "profile_subdir")]:
        if not os.path.isdir(d):
            raise SystemExit(f"[ERROR] {label} không tồn tại: {d}")

    generate_all_masks(image_dir, profile_dir, output_dir, ext=ext)
