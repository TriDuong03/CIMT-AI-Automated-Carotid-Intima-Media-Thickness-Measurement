import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

DATASET_DIR = "/work/cuc.buithi/cubs_tri/DATASET_CUBS_tech"
IMAGE_ID = "tech_401"


def load_profile(path):
    coords = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                x, y = map(float, line.split())
                coords.append((x, y))
    return np.array(coords)


def visualize(image_id):
    img_path = os.path.join(DATASET_DIR, "images", f"{image_id}.tiff")
    li_path  = os.path.join(DATASET_DIR, "LIMA-Profiles", "Manual-A1", f"{image_id}-LI.txt")
    ma_path  = os.path.join(DATASET_DIR, "LIMA-Profiles", "Manual-A1", f"{image_id}-MA.txt")

    img = np.array(Image.open(img_path))
    li  = load_profile(li_path)
    ma  = load_profile(ma_path)

    _, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(img, cmap="gray")
    ax.plot(li[:, 0], li[:, 1], color="cyan",   linewidth=1.5, label="LI (Lumen–Intima)")
    ax.plot(ma[:, 0], ma[:, 1], color="yellow",  linewidth=1.5, label="MA (Media–Adventitia)")
    ax.set_title(f"CUBS – {image_id}  |  Manual-A1 ground truth")
    ax.legend(loc="upper right")
    ax.axis("off")

    out_path = os.path.join(os.path.dirname(__file__), f"{image_id}_viz.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    visualize(IMAGE_ID)
