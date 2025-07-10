# scripts/compute_metrics.py

import os
import sys
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
parser.add_argument("--experiment-name", default="C")
parser.add_argument("--models", type=str)
args = parser.parse_args()

# =====================================
# Configuration
# =====================================
DATA_DIR = "data/full_grid"
EXPERIMENT = args.experiment_name
RESULTS_ROOT = f"results/{EXPERIMENT}"

MODELS = [
    "unet_base",
    "unet_conditional",
    "restormer",
    "pix2pix",
    "diffusion_sr3",
]

if args.models:
    MODELS = [m.strip() for m in args.models.split(",") if m.strip()]


def calc_psnr(orig, recon):
    return peak_signal_noise_ratio(orig, recon, data_range=1.0)


def calc_ssim(orig, recon):
    return structural_similarity(orig, recon, data_range=1.0, channel_axis=2)


class OcrEngine:
    UPSCALE_FACTOR = 2
    FALLBACK_DIGIT = "0"
    RECIPES = [
        # (threshold_mode, invert, page_seg_mode)
        ("fixed", False, PSM.SINGLE_WORD),
        ("otsu", False, PSM.SINGLE_WORD),
        ("adaptive", False, PSM.SINGLE_CHAR),
        ("fixed", True, PSM.SINGLE_CHAR),
        ("otsu", True, PSM.SINGLE_WORD),
    ]

    def __init__(self):
        # one Tesseract API instance per engine
        self.api = PyTessBaseAPI(oem=OEM.LSTM_ONLY, psm=PSM.SINGLE_WORD)
        # digit‐only configuration
        self.api.SetVariable("tessedit_char_whitelist", "0123456789")
        self.api.SetVariable("load_system_dawg", "0")
        self.api.SetVariable("load_freq_dawg", "0")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.api.End()

    def _preprocess(self, gray, mode, invert):
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        if mode == "fixed":
            _, bin_img = cv2.threshold(gray, 150, 255, flag)
        elif mode == "otsu":
            _, bin_img = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
        elif mode == "adaptive":
            bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, 11, 2)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return bin_img

    def _ocr_single(self, gray_patch):
        for mode, inv, psm in self.RECIPES:
            bin_p = self._preprocess(gray_patch, mode, inv)
            h, w = bin_p.shape
            up = cv2.resize(bin_p, (w * self.UPSCALE_FACTOR, h * self.UPSCALE_FACTOR), cv2.INTER_CUBIC)

            self.api.SetPageSegMode(psm)
            self.api.SetVariable("classify_bln_numeric_mode", "1")
            self.api.SetImage(Image.fromarray(up))
            self.api.Recognize()

            txt = self.api.GetUTF8Text() or ""
            for ch in txt:
                if ch.isdigit():
                    return ch, False

        # none succeeded → fallback
        return self.FALLBACK_DIGIT, True

    def _ocr_full(self, gray_plate):
        for mode, inv, psm in self.RECIPES:
            bin_p = self._preprocess(gray_plate, mode, inv)
            h, w = bin_p.shape
            up = cv2.resize(bin_p, (w * self.UPSCALE_FACTOR, h * self.UPSCALE_FACTOR), cv2.INTER_CUBIC)

            self.api.SetPageSegMode(psm)
            self.api.SetVariable("classify_bln_numeric_mode", "1")
            self.api.SetImage(Image.fromarray(up))
            self.api.Recognize()

            digits = "".join(ch for ch in (self.api.GetUTF8Text() or "") if ch.isdigit())
            if digits:
                return digits

        return ""

    def ocr_plate(self, gray_plate, bboxes):
        chars = []
        fb_mask = []
        # 1) patch-level
        for x, y, w, h in bboxes:
            ch, fb = self._ocr_single(gray_plate[y : y + h, x : x + w])
            chars.append(ch)
            fb_mask.append(fb)

        # 2) rescue missing with full‐plate
        if any(fb_mask):
            full = self._ocr_full(gray_plate)
            if len(full) >= len(chars):
                for i, fb in enumerate(fb_mask):
                    if fb:
                        chars[i] = full[i]

        return chars


def main():
    # load metadata
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    for model in MODELS:
        recon_dir = os.path.join(RESULTS_ROOT, model)
        rows = []

        with OcrEngine() as ocr:
            for rec in tqdm(metadata, desc=f"Metrics for {model}", unit="plate"):
                idx = rec["index"]
                bboxes = rec["digit_bboxes"]
                gt = rec.get("plate_number", "")

                # load images
                orig_path = os.path.join(DATA_DIR, f"original_{idx}.png")
                recon_path = os.path.join(recon_dir, f"reconstructed_{idx}.png")
                orig_bgr = cv2.imread(orig_path)
                recon_bgr = cv2.imread(recon_path)
                if orig_bgr is None or recon_bgr is None:
                    continue

                # PSNR/SSIM on patches
                orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                recon_rgb = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                gray_recon = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2GRAY)

                # Plate-level metrics
                psnr = calc_psnr(orig_rgb, recon_rgb)
                ssim = calc_ssim(orig_rgb, recon_rgb)

                psnrs = []
                ssims = []
                for x, y, w, h in bboxes:
                    orig_patch = orig_rgb[y : y + h, x : x + w]
                    recon_patch = recon_rgb[y : y + h, x : x + w]
                    psnrs.append(calc_psnr(orig_patch, recon_patch))
                    ssims.append(calc_ssim(orig_patch, recon_patch))
                worst_psnr = float(np.min(psnrs)) if psnrs else np.nan
                worst_ssim = float(np.min(ssims)) if ssims else np.nan

                # OCR pipeline
                chars = ocr.ocr_plate(gray_recon, bboxes)
                pred_str = "".join(chars)

                # OCR metrics
                correct = sum(p == g for p, g in zip(pred_str, gt))
                digit_acc = correct / len(gt) if gt else 0.0
                plate_pass = 1.0 if pred_str == gt else 0.0

                # grab alpha/beta from metadata
                alpha = rec.get("alpha")
                beta = rec.get("beta")

                rows.append(
                    {
                        "model": model,
                        "index": idx,
                        "alpha": alpha,
                        "beta": beta,
                        "psnr": round(psnr, 3),
                        "ssim": round(ssim, 5),
                        "psnr_worst": round(worst_psnr, 3),
                        "ssim_worst": round(worst_ssim, 5),
                        "ocr_digit": round(digit_acc, 2),
                        "ocr_plate": plate_pass,
                    }
                )

        # write CSV
        df = pd.DataFrame(rows).sort_values("index")
        out_csv = os.path.join(RESULTS_ROOT, f"{model}.csv")
        df.to_csv(out_csv, index=False)
        print(f"Wrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
