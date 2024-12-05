import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Directory with test data
data_dir = "data_test"
moderate_noise_threshold = 100
metric_choice = "average"  # Change to "worst" if you want to plot worst-case PSNR

# PSNR calculation using PyTorch
def calculate_psnr(outputs, targets):
    mse = F.mse_loss(outputs, targets)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

# Transform to convert PIL image to tensor in [0,1]
to_tensor = transforms.ToTensor()

# Gather all metadata files
metadata_files = [f for f in os.listdir(data_dir) if f.startswith('metadata_') and f.endswith('.json')]

psnr_dict_avg = {}
psnr_dict_worst = {}

for meta_file in tqdm(metadata_files, desc="Processing images", unit="image"):
    meta_path = os.path.join(data_dir, meta_file)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    alpha = metadata['alpha']
    beta = metadata['beta']
    noise_level = metadata['noise_level']
    digit_bboxes = metadata['digit_bboxes']

    # Consider only moderate noise
    if noise_level > moderate_noise_threshold:
        continue

    idx = metadata['idx']
    original_path = os.path.join(data_dir, f"original_{idx}.png")
    distorted_path = os.path.join(data_dir, f"distorted_{idx}.png")

    if not (os.path.exists(original_path) and os.path.exists(distorted_path)):
        continue

    # Load images as PyTorch tensors in [C,H,W] and [0,1] range
    original_img = to_tensor(Image.open(original_path).convert('RGB'))  # [C, H, W]
    distorted_img = to_tensor(Image.open(distorted_path).convert('RGB')) # [C, H, W]

    digit_psnrs = []
    for (x, y, w, h) in digit_bboxes:
        # Ensure bounding box is valid
        x, y = max(x, 0), max(y, 0)
        x2, y2 = min(x+w, original_img.shape[2]), min(y+h, original_img.shape[1])

        # Skip if no valid area
        if x2 <= x or y2 <= y:
            continue

        # Extract digit region
        original_digit = original_img[:, y:y2, x:x2]   # [C, h', w']
        distorted_digit = distorted_img[:, y:y2, x:x2] # [C, h', w']

        if original_digit.numel() == 0 or distorted_digit.numel() == 0:
            continue

        # Calculate PSNR for this digit
        psnr_val = calculate_psnr(distorted_digit.unsqueeze(0), original_digit.unsqueeze(0))
        digit_psnrs.append(psnr_val)

    if len(digit_psnrs) == 0:
        continue

    avg_digit_psnr = np.mean(digit_psnrs)
    worst_digit_psnr = min(digit_psnrs)

    # Store the results
    if (alpha, beta) not in psnr_dict_avg:
        psnr_dict_avg[(alpha, beta)] = []
        psnr_dict_worst[(alpha, beta)] = []
    psnr_dict_avg[(alpha, beta)].append(avg_digit_psnr)
    psnr_dict_worst[(alpha, beta)].append(worst_digit_psnr)

# Average over multiple images if any
for key in psnr_dict_avg:
    psnr_dict_avg[key] = np.mean(psnr_dict_avg[key])
    psnr_dict_worst[key] = np.mean(psnr_dict_worst[key])

alpha_values = sorted(list(set(a for (a, b) in psnr_dict_avg.keys())))
beta_values = sorted(list(set(b for (a, b) in psnr_dict_avg.keys())))

num_alphas = len(alpha_values)
num_betas = len(beta_values)

psnr_matrix_avg = np.full((num_betas, num_alphas), np.nan)
psnr_matrix_worst = np.full((num_betas, num_alphas), np.nan)

alpha_to_idx = {val: i for i, val in enumerate(alpha_values)}
beta_to_idx = {val: i for i, val in enumerate(beta_values)}

for (a, b), val in psnr_dict_avg.items():
    psnr_matrix_avg[beta_to_idx[b], alpha_to_idx[a]] = val

for (a, b), val in psnr_dict_worst.items():
    psnr_matrix_worst[beta_to_idx[b], alpha_to_idx[a]] = val

# Select matrix to plot
if metric_choice == "average":
    psnr_matrix = psnr_matrix_avg
    heatmap_title = "Average PSNR per Digit (Moderate Noise)"
elif metric_choice == "worst":
    psnr_matrix = psnr_matrix_worst
    heatmap_title = "Worst PSNR per Digit (Moderate Noise)"
else:
    raise ValueError("Invalid metric_choice. Use 'average' or 'worst'.")

plt.figure(figsize=(10, 8))
im = plt.imshow(psnr_matrix, origin='lower', aspect='auto', cmap="viridis")
plt.title(heatmap_title)

# Reduce axis clutter
alpha_tick_positions = range(0, num_alphas, 10)
alpha_tick_labels = [alpha_values[i] for i in alpha_tick_positions]
beta_tick_positions = range(0, num_betas, 10)
beta_tick_labels = [beta_values[i] for i in beta_tick_positions]

plt.xticks(alpha_tick_positions, alpha_tick_labels, rotation=45)
plt.yticks(beta_tick_positions, beta_tick_labels)

plt.colorbar(im, label='PSNR (dB)')

plt.tight_layout()
plt.show()
