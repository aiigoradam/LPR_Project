import os
import json
import cv2
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tesserocr import PyTessBaseAPI, OEM, PSM
from PIL import Image

# ── CONFIG ─────────────────────────────────────────────────────────────
DIR = r"results\A\restormer"
METADATA_DIR = r"data\full_grid"
METADATA_PATH = os.path.join(METADATA_DIR, "metadata.json")
LIST_PATH = "list.txt"
N_WORKERS = 8
UPSCALE_FACTOR = 2
FALLBACK_DIGIT = "0"
# ── END CONFIG ─────────────────────────────────────────────────────────

# 1) Load plate list
plates_to_check = {}
with open(LIST_PATH, encoding="utf-8") as f:
    for line in f:
        if "|" not in line:
            continue
        idx_part, gt_part = [p.strip() for p in line.strip().split("|")]
        plates_to_check[int(idx_part.split()[1])] = gt_part.split("=", 1)[1]

# 2) Load metadata
with open(METADATA_PATH, encoding="utf-8") as f:
    all_records = json.load(f)

records = [
    dict(rec, plate_number=plates_to_check[rec["index"]]) for rec in all_records if rec["index"] in plates_to_check
]
if not records:
    raise RuntimeError("No matching plates found in metadata.json")


# 3) Pre-processing helpers
def preprocess(gray, mode, invert):
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if mode == "fixed":
        _, bin_img = cv2.threshold(gray, 150, 255, flag)
    elif mode == "otsu":
        _, bin_img = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    elif mode == "adaptive":
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, 11, 2)
    else:
        raise ValueError("bad mode")
    return bin_img


RECIPES = [
    ("fixed", False, PSM.SINGLE_WORD),
    ("otsu", False, PSM.SINGLE_WORD),
    ("adaptive", False, PSM.SINGLE_CHAR),
    ("fixed", True, PSM.SINGLE_CHAR),
    ("otsu", True, PSM.SINGLE_WORD),
]


# 4) OCR a single digit patch → (char, used_fallback)
def ocr_single_digit(api, gray_patch):
    for mode, inv, psm in RECIPES:
        # 1) binarise + upscale
        bin_p = preprocess(gray_patch, mode, inv)
        h, w = bin_p.shape
        up = cv2.resize(bin_p, (w * UPSCALE_FACTOR, h * UPSCALE_FACTOR), cv2.INTER_CUBIC)

        # 2) Tesseract config: always whitelist digits
        api.SetPageSegMode(psm)
        api.SetVariable("tessedit_char_whitelist", "0123456789")
        api.SetVariable("classify_bln_numeric_mode", "1")

        # 3) OCR attempt
        api.SetImage(Image.fromarray(up))
        api.Recognize()
        txt = api.GetUTF8Text() or ""
        for ch in txt:
            if ch.isdigit():
                return ch, False

    # fallback if no digit found
    return FALLBACK_DIGIT, True


# 5) OCR the full plate image → digit-string (may be empty)
def ocr_full_plate(api, gray_plate):
    for mode, inv, psm in RECIPES:
        bin_p = preprocess(gray_plate, mode, inv)
        h, w = bin_p.shape
        up = cv2.resize(bin_p, (w * UPSCALE_FACTOR, h * UPSCALE_FACTOR), cv2.INTER_CUBIC)

        api.SetPageSegMode(psm)
        api.SetVariable("tessedit_char_whitelist", "0123456789")
        api.SetVariable("classify_bln_numeric_mode", "1")

        api.SetImage(Image.fromarray(up))
        api.Recognize()
        digits = "".join(ch for ch in (api.GetUTF8Text() or "") if ch.isdigit())
        if digits:
            return digits

    return ""


# 6) Worker for a chunk
def process_chunk(chunk):
    with PyTessBaseAPI(oem=OEM.LSTM_ONLY, psm=PSM.SINGLE_WORD) as api:
        dig_ok = dig_tot = pl_ok = pl_tot = 0
        plates_true = 0
        out_lines = []

        for rec in chunk:
            idx = rec["index"]
            truth = rec["plate_number"]
            img = cv2.imread(os.path.join(DIR, f"reconstructed_{idx}.png"))
            if img is None:
                out_lines.append(f"Plate {idx:4d} | [MISSING IMAGE]")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Patch-level OCR
            chars = []
            fb_mask = []
            for i, (x, y, w, h) in enumerate(rec["digit_bboxes"]):
                ch, fb = ocr_single_digit(api, gray[y : y + h, x : x + w])
                chars.append(ch)
                fb_mask.append(fb)
                dig_tot += 1
                if ch == truth[i]:
                    dig_ok += 1

            # Full-plate rescue if any fallback
            if any(fb_mask):
                full = ocr_full_plate(api, gray)
                if len(full) >= len(chars):
                    for i, fb in enumerate(fb_mask):
                        if fb:
                            chars[i] = full[i]
                            fb_mask[i] = False

            plate_ocr = "".join(chars)
            if plate_ocr == truth:
                pl_ok += 1
            pl_tot += 1
            if not any(fb_mask):
                plates_true += 1

            out_lines.append(
                f"Plate {idx:4d} | GT={truth} | OCR={plate_ocr:<6} | " f"{'OK' if plate_ocr==truth else 'ERR'}"
            )

        return dig_ok, dig_tot, pl_ok, pl_tot, plates_true, out_lines


# 7) Multithreaded execution
chunk_sz = math.ceil(len(records) / N_WORKERS)
chunks = [records[i : i + chunk_sz] for i in range(0, len(records), chunk_sz)]

digit_ok = digit_tot = plate_ok = plate_tot = 0
total_true = 0
all_lines = []

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = [ex.submit(process_chunk, c) for c in chunks]
    for fut in as_completed(futures):
        d_ok, d_tot, p_ok, p_tot, pt, lines = fut.result()
        digit_ok += d_ok
        digit_tot += d_tot
        plate_ok += p_ok
        plate_tot += p_tot
        total_true += pt
        all_lines.extend(lines)

# 8) Reporting
for ln in sorted(all_lines):
    if "| ERR" in ln or "[MISSING IMAGE]" in ln:
        print(ln)

print("\nRESULTS")
print(f"  Digit-level accuracy: {digit_ok/digit_tot*100:.2f}% " f"({digit_ok}/{digit_tot})")
print(f"  Plate-level accuracy: {plate_ok/plate_tot*100:.2f}% " f"({plate_ok}/{plate_tot})")

total = len(records)
print(
    f"  Plates with all six digits from Tesseract (no fallbacks): "
    f"{total_true}/{total} ({total_true/total*100:.2f}%)"
)
