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
import argparse

# ===== argparse for override =====
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", default="A")
parser.add_argument("--models", type=str)
args = parser.parse_args()

# =====================================
# Configuration
# =====================================
DATA_DIR = "data/full_grid"
EXPERIMENT_NAME = args.experiment_name
RESULTS_ROOT = f"results/{EXPERIMENT_NAME}"

MODELS = [
    "unet_base",
    "unet_conditional",
    "restormer",
    "pix2pix",
    "diffusion_sr3",
]

# override if passed
if args.models:
    MODELS = [m.strip() for m in args.models.split(",") if m.strip()]


def calc_psnr(orig, recon):
    return peak_signal_noise_ratio(orig, recon, data_range=1.0)


def calc_ssim(orig, recon):
    return structural_similarity(orig, recon, data_range=1.0, channel_axis=2)


def ocr_digit(patch_bin, api):
    pil = Image.fromarray(patch_bin)
    api.SetImage(pil)
    txt = api.GetUTF8Text().strip()
    return txt[0] if (txt and txt[0].isdigit()) else ""


def main():
    with open(os.path.join(DATA_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)

    for model_name in MODELS:
        recon_dir = os.path.join(RESULTS_ROOT, model_name)
        rows = []

        with PyTessBaseAPI(oem=OEM.LSTM_ONLY, psm=PSM.SINGLE_WORD) as api:
            api.SetVariable("tessedit_char_whitelist", "0123456789")
            api.SetVariable("load_system_dawg", "0")
            api.SetVariable("load_freq_dawg", "0")

            for rec in tqdm(metadata, desc=f"Processing {model_name}", unit="plate"):
                idx = rec["index"]
                alpha = rec.get("alpha")
                beta = rec.get("beta")
                bboxes = rec["digit_bboxes"]
                plate_gt = rec.get("plate_number", "")

                orig_path = os.path.join(DATA_DIR, f"original_{idx}.png")
                recon_path = os.path.join(recon_dir, f"reconstructed_{idx}.png")
                orig_bgr = cv2.imread(orig_path)
                recon_bgr = cv2.imread(recon_path)
                if orig_bgr is None or recon_bgr is None:
                    continue

                # to float RGB [0,1]
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

                correct = sum(p == g for p, g in zip(ocr_preds, plate_gt))
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

        out_csv = os.path.join(RESULTS_ROOT, f"{model_name}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Wrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
