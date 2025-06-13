# scripts/compute_metrics.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tesserocr import PyTessBaseAPI, OEM, PSM
from PIL import Image

# =====================
#    CONFIGURATION
# =====================
root = "results"  # root containing model subfolders
model_name = "pix2pix"  # folder under results/
data_dir = "data/full_grid"  # where metadata.json & original_*.png live
metadata_path = os.path.join(data_dir, "metadata.json")
recon_dir = os.path.join(root, model_name)
# =====================


def calc_psnr(orig, recon):
    return peak_signal_noise_ratio(orig, recon, data_range=1.0)


def calc_ssim(orig, recon):
    return structural_similarity(orig, recon, data_range=1.0, channel_axis=2)


def ocr_digit(patch_bin, api):
    pil = Image.fromarray(patch_bin)
    api.SetImage(pil)
    txt = api.GetUTF8Text().strip()
    return txt[0] if (txt and txt[0].isdigit()) else ""


# ── MAIN ──────────────────────────────────────────────────────────────
with open(metadata_path, "r") as f:
    metadata = json.load(f)

rows = []
with PyTessBaseAPI(oem=OEM.LSTM_ONLY, psm=PSM.SINGLE_WORD) as api:
    api.SetVariable("tessedit_char_whitelist", "0123456789")
    api.SetVariable("load_system_dawg", "0")
    api.SetVariable("load_freq_dawg", "0")

    for rec in tqdm(metadata, desc=f"Processing {model_name}", unit="plate"):
        idx = rec["index"]
        alpha = rec["alpha"]
        beta = rec["beta"]
        bboxes = rec["digit_bboxes"]
        plate_gt = rec["plate_number"]

        orig_bgr = cv2.imread(os.path.join(data_dir, f"original_{idx}.png"))
        recon_bgr = cv2.imread(os.path.join(recon_dir, f"reconstructed_{idx}.png"))
        if orig_bgr is None or recon_bgr is None:
            continue

        # to float-RGB [0,1]
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        recon_rgb = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # prepare binary for OCR
        gray_recon = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2GRAY)
        _, bin_recon = cv2.threshold(gray_recon, 150, 255, cv2.THRESH_BINARY)

        psnr_vals, ssim_vals, ocr_preds = [], [], []

        for x, y, w, h in bboxes:
            orig_patch = orig_rgb[y : y + h, x : x + w]
            recon_patch = recon_rgb[y : y + h, x : x + w]

            psnr_vals.append(calc_psnr(orig_patch, recon_patch))
            ssim_vals.append(calc_ssim(orig_patch, recon_patch))

            patch_bin = bin_recon[y : y + h, x : x + w]
            ocr_preds.append(ocr_digit(patch_bin, api))

        worst_psnr = float(np.min(psnr_vals)) if psnr_vals else np.nan
        worst_ssim = float(np.min(ssim_vals)) if ssim_vals else np.nan

        correct = sum(1 for p, g in zip(ocr_preds, plate_gt) if p == g)
        digit_acc = correct / len(plate_gt) if plate_gt else 0.0
        plate_pass = 1.0 if "".join(ocr_preds) == plate_gt else 0.0

        rows.append(
            {
                "model": model_name,
                "index": idx,
                "alpha": alpha,
                "beta": beta,
                "psnr_worst": worst_psnr,
                "ssim_worst": worst_ssim,
                "ocr_digit_acc": digit_acc,
                "ocr_plate_pass": plate_pass,
            }
        )

# build DataFrame, round & sort
df = pd.DataFrame(rows)
df["psnr_worst"] = df["psnr_worst"].round(3)
df["ssim_worst"] = df["ssim_worst"].round(5)
df["ocr_digit_acc"] = df["ocr_digit_acc"].round(2)
df = df.sort_values("index")

out_csv = os.path.join(root, f"{model_name}.csv")
df.to_csv(out_csv, index=False)
print(f"Wrote {len(df)} rows to {out_csv}")
