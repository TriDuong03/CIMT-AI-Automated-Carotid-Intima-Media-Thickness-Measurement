"""
Đánh giá lớp nội trung mạc (Intima-Media Layer) với model BasicUNetPlusPlus.

So sánh với:
  - GT mask  : Dice, IoU, HD95 (pixel-level)
  - Manual-A1: MAE/RMSE trên biên LI và MA (pixel)
  - IMT (mm) : MAE, RMSE, Bland-Altman, Pearson
"""

import csv
import datetime
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import pearsonr

SCRIPTS_DIR  = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_ROOT = PROJECT_ROOT / "DATASET_CUBS_tech"
IMAGE_DIR    = DATASET_ROOT / "images"
GT_MASK_DIR  = DATASET_ROOT / "GT_masks"
LIMA_DIR     = DATASET_ROOT / "LIMA-Profiles-interpolated" / "Manual-A1"
CF_DIR       = DATASET_ROOT / "CF"
WEIGHT_PATH  = PROJECT_ROOT / "weights" / "best_model_dilated.pt"
OUTPUT_DIR   = PROJECT_ROOT / "logs" / "imt_eval"
IMG_SIZE     = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ─── Model ────────────────────────────────────────────────────────────────────
def load_model():
    from models.unetplusplus_dilated import model
    ckpt  = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')}, best_val_dice={ckpt.get('best_val_dice',0):.4f}")
    return model.to(DEVICE).eval()


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(img_gray: np.ndarray):
    h, w   = img_gray.shape[:2]
    scale  = IMG_SIZE / max(h, w)
    new_h  = int(round(h * scale))
    new_w  = int(round(w * scale))
    resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top   = (IMG_SIZE - new_h) // 2
    pad_bot   = IMG_SIZE - new_h - pad_top
    pad_left  = (IMG_SIZE - new_w) // 2
    pad_right = IMG_SIZE - new_w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bot, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=0)

    img_f = padded.astype(np.float32)
    mn, mx = img_f.min(), img_f.max()
    img_f  = (img_f - mn) / (mx - mn + 1e-8)

    tensor = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return tensor, dict(scale=scale, pad_top=pad_top, pad_left=pad_left,
                        orig_h=h, orig_w=w, new_h=new_h, new_w=new_w)


def pred_to_mask_orig(output, params):
    """
    Đưa logit dự đoán (512×512) về không gian ảnh gốc.
    Bước 1: softmax → prob channel 1
    Bước 2: cắt bỏ padding (top/left/right/bottom)
    Bước 3: resize về kích thước gốc (orig_h × orig_w)
    """
    if isinstance(output, (list, tuple)):
        output = output[0]
    prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()   # (512, 512)

    pt = params["pad_top"]
    pl = params["pad_left"]
    nh = params["new_h"]
    nw = params["new_w"]
    # Chỉ lấy vùng thực sự là ảnh (loại bỏ toàn bộ padding)
    cropped = prob[pt: pt + nh, pl: pl + nw]                  # (new_h, new_w)

    restored = cv2.resize(cropped, (params["orig_w"], params["orig_h"]),
                          interpolation=cv2.INTER_LINEAR)      # (orig_h, orig_w)
    return (restored > 0.5).astype(np.uint8)


# ─── Boundary extraction (vectorized) ────────────────────────────────────────
def extract_boundaries(mask: np.ndarray):
    """
    Trả về (xs, li_ys, ma_ys) trên các cột có vùng IMT.
    Vectorized: không dùng Python loop.
    """
    # Dùng argmax từ trên xuống (LI) và từ dưới lên (MA)
    col_has_imt = mask.any(axis=0)                      # (W,) bool
    xs = np.where(col_has_imt)[0]
    if len(xs) == 0:
        return np.array([]), np.array([]), np.array([])

    sub = mask[:, xs]                                   # (H, n_cols)
    li_ys = np.argmax(sub,         axis=0).astype(float)
    ma_ys = (sub.shape[0] - 1 - np.argmax(sub[::-1], axis=0)).astype(float)
    return xs, li_ys, ma_ys


# ─── LIMA profile utils ───────────────────────────────────────────────────────
def load_profile(stem: str, boundary: str):
    path = LIMA_DIR / f"{stem}-{boundary}.txt"
    vals = np.array(path.read_text().split(), dtype=float)
    n    = len(vals) // 2
    return vals[:n].astype(int), vals[n:]   # xs (int), ys (float)


def load_cf(stem: str) -> float:
    return float((CF_DIR / f"{stem}_CF.txt").read_text().strip())


# ─── Pixel-level metrics (fast) ───────────────────────────────────────────────
def dice_iou(pred: np.ndarray, gt: np.ndarray):
    p, g   = pred.astype(bool), gt.astype(bool)
    inter  = (p & g).sum()
    dice   = 2 * inter / (p.sum() + g.sum() + 1e-8)
    iou    = inter / ((p | g).sum() + 1e-8)
    return float(dice), float(iou)


def hausdorff_95_fast(pred: np.ndarray, gt: np.ndarray):
    """
    HD95 nhanh: chỉ dùng distance transform thay vì cKDTree trên toàn boundary.
    O(H*W) thay vì O(N^2).
    """
    from scipy.ndimage import distance_transform_edt, binary_erosion

    def get_boundary(m):
        b = m.astype(bool)
        return b & ~binary_erosion(b)

    pb = get_boundary(pred)
    gb = get_boundary(gt)
    if not pb.any() or not gb.any():
        return float("nan")

    # distance từ mỗi pixel đến boundary của ảnh kia
    dist_gt   = np.asarray(distance_transform_edt(~gb))
    dist_pred = np.asarray(distance_transform_edt(~pb))
    d_pred2gt = dist_gt[pb]
    d_gt2pred = dist_pred[gb]
    return float(np.percentile(np.concatenate([d_pred2gt, d_gt2pred]), 95))


def profile_mae_rmse(pred_y, gt_y):
    diff = pred_y - gt_y
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2)))


# ─── Main ────────────────────────────────────────────────────────────────────
def evaluate():
    model  = load_model()
    gt_ids = sorted(
        f.replace("_mask.png", "")
        for f in os.listdir(GT_MASK_DIR)
        if f.endswith("_mask.png")
    )
    print(f"\nSố ảnh có GT mask: {len(gt_ids)}\n")

    results = []

    for i, stem in enumerate(gt_ids, 1):
        img_gray = cv2.imread(str(IMAGE_DIR / f"{stem}.tiff"), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"[WARN] Không tìm thấy: {stem}.tiff"); continue

        gt_raw  = cv2.imread(str(GT_MASK_DIR / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if gt_raw is None:
            print(f"[WARN] Không tìm thấy: {stem}_mask.png"); continue
        gt_mask = (gt_raw > 127).astype(np.uint8)

        tensor, params = preprocess(img_gray)
        with torch.no_grad():
            output = model(tensor)
        pred_mask = pred_to_mask_orig(output, params)

        # ── Pixel-level metrics (pred vs GT mask, cùng không gian gốc) ──────
        d, iou = dice_iou(pred_mask, gt_mask)
        hd95   = hausdorff_95_fast(pred_mask, gt_mask)

        row = dict(stem=stem, dice=d, iou=iou, hd95=hd95)

        # ── Lấy vùng cột hợp lệ từ GT mask (ROI per-image) ─────────────────
        # GT mask đã ở không gian gốc (orig_h × orig_w)
        gt_cols = np.where(gt_mask.any(axis=0))[0]
        if len(gt_cols) == 0:
            results.append(row); continue

        # ── Trích biên LI/MA từ pred_mask (đã ở không gian gốc) ─────────────
        px_pred, li_pred, ma_pred = extract_boundaries(pred_mask)

        # ── Load profile Manual-A1 ───────────────────────────────────────────
        try:
            xs_li_gt, ys_li_gt = load_profile(stem, "LI")
            xs_ma_gt, ys_ma_gt = load_profile(stem, "MA")
        except FileNotFoundError:
            results.append(row); continue

        # Giao của: cột GT mask  ∩  cột pred có IMT  ∩  cột có profile LI & MA
        common_x = np.intersect1d(
            gt_cols,
            np.intersect1d(px_pred, np.intersect1d(xs_li_gt, xs_ma_gt))
        )
        if len(common_x) < 10:
            results.append(row); continue

        def pick(xs_ref: np.ndarray, ys_ref: np.ndarray, common: np.ndarray) -> np.ndarray:
            idx = np.searchsorted(xs_ref, common)
            idx = np.clip(idx, 0, len(xs_ref) - 1)
            return ys_ref[idx]

        li_p = li_pred[np.searchsorted(px_pred, common_x)]
        ma_p = ma_pred[np.searchsorted(px_pred, common_x)]
        li_g = pick(xs_li_gt, ys_li_gt, common_x)
        ma_g = pick(xs_ma_gt, ys_ma_gt, common_x)

        cf = load_cf(stem)
        imt_pred = (ma_p - li_p) * cf
        imt_gt   = (ma_g - li_g) * cf

        li_mae,  li_rmse  = profile_mae_rmse(li_p * cf, li_g * cf)
        ma_mae,  ma_rmse  = profile_mae_rmse(ma_p * cf, ma_g * cf)
        imt_mae, imt_rmse = profile_mae_rmse(imt_pred, imt_gt)

        row["li_mae"]        = li_mae
        row["li_rmse"]       = li_rmse
        row["ma_mae"]        = ma_mae
        row["ma_rmse"]       = ma_rmse
        row["imt_mae"]       = imt_mae
        row["imt_rmse"]      = imt_rmse
        row["mean_imt_pred"] = float(imt_pred.mean())
        row["mean_imt_gt"]   = float(imt_gt.mean())
        row["imt_pred_arr"]  = imt_pred
        row["imt_gt_arr"]    = imt_gt
        results.append(row)

        if i % 10 == 0:
            print(f"  [{i:3d}/{len(gt_ids)}] {stem}  Dice={d:.3f}  IMT-MAE={imt_mae:.4f}mm")

    # ─── Tổng hợp ─────────────────────────────────────────────────────────────
    def ms(key):
        v = [r[key] for r in results if key in r]
        return np.mean(v), np.std(v)

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("\n" + "=" * 65)
    log("  KẾT QUẢ ĐÁNH GIÁ LỚP NỘI TRUNG MẠC (INTIMA-MEDIA LAYER)")
    log("=" * 65)
    log(f"  Thời gian: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Checkpoint: {WEIGHT_PATH.name}")
    log(f"  Số ảnh đánh giá: {len(results)}")

    d_m, d_s = ms("dice");  i_m, i_s = ms("iou");  h_m, h_s = ms("hd95")
    log(f"\n── PIXEL-LEVEL (so với GT mask) ──────────────────────────")
    log(f"  Dice  : {d_m:.4f} ± {d_s:.4f}")
    log(f"  IoU   : {i_m:.4f} ± {i_s:.4f}")
    log(f"  HD95  : {h_m:.2f}  ± {h_s:.2f}  px")

    li_m, li_s = ms("li_mae");  li_r, li_rs = ms("li_rmse")
    ma_m, ma_s = ms("ma_mae");  ma_r, ma_rs = ms("ma_rmse")
    log(f"\n── BIÊN LI/MA [mm] (so với Manual-A1) ────────────────────")
    log(f"  LI  MAE : {li_m:.4f} ± {li_s:.4f} mm    RMSE : {li_r:.4f} ± {li_rs:.4f} mm")
    log(f"  MA  MAE : {ma_m:.4f} ± {ma_s:.4f} mm    RMSE : {ma_r:.4f} ± {ma_rs:.4f} mm")

    imt_m, imt_s = ms("imt_mae");  imt_r, imt_rs = ms("imt_rmse")
    log(f"\n── IMT [mm] ───────────────────────────────────────────────")
    log(f"  MAE  : {imt_m:.4f} ± {imt_s:.4f} mm")
    log(f"  RMSE : {imt_r:.4f} ± {imt_rs:.4f} mm")

    full = [r for r in results if "imt_pred_arr" in r]
    if full:
        all_pred = np.concatenate([r["imt_pred_arr"] for r in full])
        all_gt   = np.concatenate([r["imt_gt_arr"]   for r in full])
        bias     = float(np.mean(all_pred - all_gt))
        loa      = 1.96 * float(np.std(all_pred - all_gt))
        r_val, p_val = pearsonr(all_pred, all_gt)
        log(f"\n  Bland-Altman bias    : {bias:+.4f} mm")
        log(f"  Limits of agreement  : [{bias-loa:.4f}, {bias+loa:.4f}] mm")
        log(f"  Pearson r            : {r_val:.4f}  (p={p_val:.2e})")

        gt_m,   gt_s   = ms("mean_imt_gt")
        pred_m, pred_s = ms("mean_imt_pred")
        log(f"\n  Mean IMT GT          : {gt_m:.4f} ± {gt_s:.4f} mm")
        log(f"  Mean IMT Predicted   : {pred_m:.4f} ± {pred_s:.4f} mm")

    imt_per = sorted([(r["stem"], r["imt_mae"]) for r in results if "imt_mae" in r],
                     key=lambda x: x[1])
    log(f"\n── TOP-5 TỐT NHẤT (IMT MAE) ───────────────────────────────")
    for s, v in imt_per[:5]:
        log(f"  {s}   {v:.4f} mm")
    log(f"\n── TOP-5 XẤU NHẤT (IMT MAE) ───────────────────────────────")
    for s, v in imt_per[-5:]:
        log(f"  {s}   {v:.4f} mm")

    log(f"\n{'='*65}")
    log(f"  Tổng ảnh đã đánh giá: {len(results)}")
    log(f"{'='*65}")

    save_results(results, lines)


def save_results(results, summary_lines: list):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) CSV — chỉ số từng ảnh
    csv_path = OUTPUT_DIR / f"per_image_{ts}.csv"
    scalar_keys = ["stem", "dice", "iou", "hd95",
                   "li_mae", "li_rmse", "ma_mae", "ma_rmse",
                   "imt_mae", "imt_rmse", "mean_imt_pred", "mean_imt_gt"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Saved] per-image CSV  → {csv_path}")

    # 2) TXT — bản tóm tắt
    txt_path = OUTPUT_DIR / f"summary_{ts}.txt"
    txt_path.write_text("\n".join(summary_lines) + "\n")
    print(f"[Saved] summary TXT    → {txt_path}")


if __name__ == "__main__":
    evaluate()
