##Heatmap of worst PSNR, SSIM, and OCR (parallel processing)
import os
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tqdm import tqdm
import numpy as np
import mlflow
import mlflow.pytorch
import cv2
from pytorch_msssim import ssim
import pytesseract
from joblib import Parallel, delayed

# --------------------
# Configuration
# --------------------
data_dir = "data_test"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MLflow model load
mlflow.set_experiment('Unet')
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name('Unet')
runs = client.search_runs(
    experiment_ids=experiment.experiment_id,
    order_by=["attributes.start_time DESC"],
    max_results=1
)
run_id = runs[0].info.run_id
model_uri = f"runs:/{run_id}/model"
model = mlflow.pytorch.load_model(model_uri)
model.eval().to(device)
print(f"Model loaded from run {run_id} in experiment '{experiment.name}' successfully.")

# --------------------
# Functions 
# --------------------

def calculate_psnr(outputs, targets):
    mse = F.mse_loss(outputs, targets)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

def ocr_single_digit(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    config = r'--oem 1 --psm 10 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(thresh, config=config).strip()
    if len(text) == 1 and text.isdigit():
        return text
    return '?'

def align_and_update_bboxes(original_np, generated_np, digit_bboxes):
    search_margin = 16

    def process_digit_bbox(bbox):
        x, y, w, h = bbox
        original_digit = original_np[y:y+h, x:x+w, :]
        original_digit_gray = cv2.cvtColor((original_digit * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Define search window
        search_x1 = max(0, x - search_margin)
        search_y1 = max(0, y - search_margin)
        search_x2 = min(generated_np.shape[1], x + w + search_margin)
        search_y2 = min(generated_np.shape[0], y + h + search_margin)
        search_region = generated_np[search_y1:search_y2, search_x1:search_x2, :]
        search_region_gray = cv2.cvtColor((search_region * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Template matching
        result = cv2.matchTemplate(search_region_gray, original_digit_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        best_x, best_y = max_loc[0] + search_x1, max_loc[1] + search_y1

        # Compute PSNR and SSIM
        aligned_digit = generated_np[best_y:best_y+h, best_x:best_x+w, :]
        original_digit_tensor = torch.from_numpy(original_digit.transpose(2,0,1)).unsqueeze(0)
        aligned_digit_tensor = torch.from_numpy(aligned_digit.transpose(2,0,1)).unsqueeze(0)

        psnr_val = calculate_psnr(aligned_digit_tensor, original_digit_tensor)
        ssim_val = ssim(aligned_digit_tensor, original_digit_tensor, data_range=1.0, size_average=True).item()

        return psnr_val, ssim_val, (best_x, best_y, w, h)

    results = Parallel(n_jobs=-1)(delayed(process_digit_bbox)(bbox) for bbox in digit_bboxes)
    psnr_values = [r[0] for r in results]
    ssim_values = [r[1] for r in results]
    updated_bboxes = [r[2] for r in results]

    return psnr_values, ssim_values, updated_bboxes

def compute_ocr_metrics(image_bgr, updated_bboxes, plate_number_gt, margin):
    def process_bbox(bbox):
        x, y, w, h = bbox
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image_bgr.shape[1], x + w + margin)
        y2 = min(image_bgr.shape[0], y + h + margin)
        digit_patch = image_bgr[y1:y2, x1:x2]
        recognized_digit = ocr_single_digit(digit_patch)
        return recognized_digit

    recognized_digits = Parallel(n_jobs=-1)(delayed(process_bbox)(bbox) for bbox in updated_bboxes)
    recognized_text = "".join(recognized_digits)
    gt = plate_number_gt
    correct_digits = sum(1 for a, b in zip(gt, recognized_text) if a == b)
    ocr_accuracy = correct_digits / len(gt) if len(gt) > 0 else 0.0
    ocr_binary = 1.0 if recognized_text == gt else 0.0
    return recognized_text, ocr_accuracy, ocr_binary

# --------------------------------------
# Compute metrics for each (alpha, beta)
# --------------------------------------
metadata_files = [f for f in os.listdir(data_dir) if f.startswith('metadata_') and f.endswith('.json')]

psnr_dict_worst = {}
ssim_dict_worst = {}
ocr_acc_dict_avg = {}
ocr_bin_dict_avg = {}

to_tensor = transforms.ToTensor()

for meta_file in tqdm(metadata_files, desc="Processing images", unit="image"):
    meta_path = os.path.join(data_dir, meta_file)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    alpha, beta = metadata['alpha'], metadata['beta']
    digit_bboxes = metadata['digit_bboxes']
    plate_number_gt = metadata['plate_number']

    idx = metadata['idx']
    original_path = os.path.join(data_dir, f"original_{idx}.png")
    distorted_path = os.path.join(data_dir, f"distorted_{idx}.png")

    if not (os.path.exists(original_path) and os.path.exists(distorted_path)):
        continue

    original_img = to_tensor(Image.open(original_path).convert('RGB')).unsqueeze(0).to(device)
    distorted_img = to_tensor(Image.open(distorted_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        generated_img = model(distorted_img)
        generated_img = torch.clamp(generated_img, 0.0, 1.0)

    original_np = original_img.squeeze(0).permute(1,2,0).cpu().numpy()
    generated_np = generated_img.squeeze(0).permute(1,2,0).cpu().numpy()

    # Parallelized CPU operations
    psnr_per_number, ssim_per_number, updated_bboxes = align_and_update_bboxes(original_np, generated_np, digit_bboxes)
    image_bgr = (generated_np * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
    recognized_text, ocr_accuracy, ocr_binary = compute_ocr_metrics(image_bgr, updated_bboxes, plate_number_gt, margin=2)

    # Take worst PSNR and SSIM
    worst_psnr = np.min(psnr_per_number) if psnr_per_number else 0.0
    worst_ssim = np.min(ssim_per_number) if ssim_per_number else 0.0

    if (alpha, beta) not in psnr_dict_worst:
        psnr_dict_worst[(alpha, beta)] = []
        ssim_dict_worst[(alpha, beta)] = []
        ocr_acc_dict_avg[(alpha, beta)] = []
        ocr_bin_dict_avg[(alpha, beta)] = []

    psnr_dict_worst[(alpha, beta)].append(worst_psnr)
    ssim_dict_worst[(alpha, beta)].append(worst_ssim)
    ocr_acc_dict_avg[(alpha, beta)].append(ocr_accuracy)
    ocr_bin_dict_avg[(alpha, beta)].append(ocr_binary)

alpha_values = sorted(set(a for (a, b) in psnr_dict_worst.keys()))
beta_values = sorted(set(b for (a, b) in psnr_dict_worst.keys()))
num_alphas, num_betas = len(alpha_values), len(beta_values)

def create_matrix_from_dict(data_dict):
    mat = np.full((num_betas, num_alphas), np.nan)
    alpha_to_idx = {val: i for i, val in enumerate(alpha_values)}
    beta_to_idx = {val: i for i, val in enumerate(beta_values)}
    for (a, b), val_list  in data_dict.items():
        val = np.min(val_list) if val_list else np.nan
        mat[beta_to_idx[b], alpha_to_idx[a]] = val
    return mat

psnr_matrix = create_matrix_from_dict(psnr_dict_worst)
ssim_matrix = create_matrix_from_dict(ssim_dict_worst)
ocr_acc_matrix = create_matrix_from_dict(ocr_acc_dict_avg)
ocr_bin_matrix = create_matrix_from_dict(ocr_bin_dict_avg)

# ----------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------- #

def show_image_details_for(alpha, beta):
    # Re-run the detailed view logic
    found_file = None
    for meta_file in metadata_files:
        meta_path = os.path.join(data_dir, meta_file)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        if metadata['alpha'] == alpha and metadata['beta'] == beta:
            found_file = metadata
            break

    if found_file is None:
        print("No images found for that angle.")
        return

    found_file['digit_bboxes'].sort(key=lambda bbox: bbox[0])
    idx = found_file['idx']
    plate_number_gt = found_file['plate_number']
    original_path = os.path.join(data_dir, f"original_{idx}.png")
    distorted_path = os.path.join(data_dir, f"distorted_{idx}.png")

    # Load images as tensors for generation
    original_img = to_tensor(Image.open(original_path).convert('RGB')).unsqueeze(0).to(device)
    distorted_img = to_tensor(Image.open(distorted_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        generated_tensor = model(distorted_img)
        generated_tensor = torch.clamp(generated_tensor, 0.0, 1.0)

    # Convert tensors to numpy arrays
    original_np = original_img.squeeze(0).permute(1,2,0).cpu().numpy()
    generated_np = generated_tensor.squeeze(0).permute(1,2,0).cpu().numpy()

    # Compute PSNR, SSIM, and bounding boxes
    psnr_vals, ssim_vals, updated_bboxes = align_and_update_bboxes(original_np, generated_np, found_file['digit_bboxes'])

    # Prepare images for display
    distorted_image_cv = cv2.imread(distorted_path)
    distorted_image_rgb = cv2.cvtColor(distorted_image_cv, cv2.COLOR_BGR2RGB)

    # Original image (with rectangles and text)
    original_image_cv = cv2.imread(original_path)
    for i, bbox in enumerate(found_file['digit_bboxes'], start=1):
        x, y, w, h = bbox
        cv2.rectangle(original_image_cv, (x, y), (x+w, y+h), (0,0,255),1)
        cv2.putText(original_image_cv, str(i), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,150,0),1)
    original_image_rgb = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)

    # Generated image (with rectangles and text)
    generated_bgr = (generated_np * 255).astype(np.uint8)
    generated_bgr = cv2.cvtColor(generated_bgr, cv2.COLOR_RGB2BGR)  
    generated_show = generated_bgr.copy()
    for i,bbox in enumerate(updated_bboxes, start=1):
        x,y,w,h = bbox
        cv2.rectangle(generated_show, (x,y),(x+w,y+h),(0,0,255),1)
        cv2.putText(generated_show,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,150,0),1)
    generated_image_rgb = cv2.cvtColor(generated_show, cv2.COLOR_BGR2RGB)

    recognized_text, ocr_accuracy, ocr_binary = compute_ocr_metrics(generated_bgr, updated_bboxes, plate_number_gt, margin=2)

    # Prepare table data
    table_data = [["Digit","PSNR(dB)","SSIM"]]
    for i,(p,s) in enumerate(zip(psnr_vals, ssim_vals), start=1):
        table_data.append([str(i), f"{p:.2f}", f"{s:.3f}"])
    transposed_table_data = list(zip(*table_data))

    # Create figure with 3 rows: Distorted, Original, Generated
    fig2 = plt.figure(figsize=(11,9))

    # Distorted image 
    plt.subplot(3,1,1)
    plt.imshow(distorted_image_rgb)
    plt.title(f'Distorted Image (Alpha={alpha}, Beta={beta})')
    plt.axis('off')

    # Original image 
    plt.subplot(3,1,2)
    plt.imshow(original_image_rgb)
    plt.title(f'Original Image')
    plt.axis('off')

    # Generated image 
    plt.subplot(3,1,3)
    plt.imshow(generated_image_rgb)
    plt.title(f'Generated Image\nGT: {plate_number_gt}, Rec: {recognized_text}, OCR Acc: {ocr_accuracy*100:.2f}%, Binary: {int(ocr_binary)}')
    plt.axis('off')

    # Add table below all images
    # Adjust the bbox so it appears below the last subplot
    table = plt.table(cellText=transposed_table_data,
                      cellLoc='center',
                      loc='center',
                      bbox=[0,-0.5, 1,0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    plt.tight_layout()
    plt.show()

def format_coord(x, y):
    col = int(round(x))
    row = int(round(y))
    if 0 <= row < num_betas and 0 <= col < num_alphas:
        alpha = alpha_values[col]
        beta = beta_values[row]
        psnr_value = psnr_matrix_clipped[row, col]
        return f"Alpha: {alpha:.0f}, Beta: {beta:.0f}, PSNR: {psnr_value:.2f} dB" if not np.isnan(psnr_value) else f"Alpha: {alpha:.0f}, Beta: {beta:.0f}, PSNR: N/A"
    return "Alpha: N/A, Beta: N/A"

psnr_matrix_clipped = np.clip(psnr_matrix, None, 20)

current_metric = 'PSNR'
fig, ax = plt.subplots(figsize=(11, 9))
plt.subplots_adjust(bottom=0.15)  # space for buttons

# Draw initial heatmap
im = ax.imshow(psnr_matrix_clipped, origin='lower', aspect='auto', cmap="viridis")
ax.set_title("Worst PSNR per Image (Minimum Digit PSNR)")
cbar = plt.colorbar(im, ax=ax, label='PSNR (dB)')
ax.set_xticks(range(0, num_alphas, 5))
ax.set_xticklabels(alpha_values[::5])
ax.set_yticks(range(0, num_betas, 5))
ax.set_yticklabels(beta_values[::5])
ax.set_xlabel("Alpha (degrees)")
ax.set_ylabel("Beta (degrees)")
ax.format_coord = format_coord  # Set the coordinate display format

# Define button positions
button_width = 0.1   # Button width
button_height = 0.05  # Button height
button_spacing = 0.02  # Space between buttons

# Compute x-coordinates for buttons
x_start = 0.2  # Starting x-position
y_position = 0.03
x_psnr = x_start
x_ssim = x_psnr + button_width + button_spacing
x_ocr_acc = x_ssim + button_width + button_spacing
x_ocr_bin = x_ocr_acc + button_width + button_spacing

# Add buttons
ax_psnr = plt.axes([x_psnr, y_position, button_width, button_height])
ax_ssim = plt.axes([x_ssim, y_position, button_width, button_height])
ax_ocr_acc = plt.axes([x_ocr_acc, y_position, button_width, button_height])
ax_ocr_bin = plt.axes([x_ocr_bin, y_position, button_width, button_height])

btn_psnr = Button(ax_psnr, 'PSNR')
btn_ssim = Button(ax_ssim, 'SSIM')
btn_ocr_acc = Button(ax_ocr_acc, 'OCR Acc')
btn_ocr_bin = Button(ax_ocr_bin, 'OCR Bin')

def update_heatmap(metric):
    global current_metric
    current_metric = metric
    ax.clear()
    
    if metric == 'PSNR':
        data = psnr_matrix_clipped
        title = "Worst PSNR per Image (Minimum Digit PSNR)"
        cbar_label = "PSNR (dB)"
    elif metric == 'SSIM':
        data = ssim_matrix
        title = "Worst SSIM per Image (Minimum Digit SSIM)"
        cbar_label = "SSIM"
    elif metric == 'OCR_Accuracy':
        data = ocr_acc_matrix
        title = "Average OCR Accuracy"
        cbar_label = "OCR Acc"
    else:
        data = ocr_bin_matrix
        title = "OCR Binary (1=All Correct)"
        cbar_label = "OCR Binary"

    # Update heatmap
    im = ax.imshow(data, origin='lower', aspect='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xticks(range(0, num_alphas, 5))
    ax.set_xticklabels(alpha_values[::5])
    ax.set_yticks(range(0, num_betas, 5))
    ax.set_yticklabels(beta_values[::5])
    ax.set_xlabel("Alpha (degrees)")
    ax.set_ylabel("Beta (degrees)")

    # Update colorbar
    cbar.mappable = im
    cbar.set_label(cbar_label)
    cbar.update_normal(im)

    fig.canvas.draw_idle()

def on_psnr_clicked(event):
    update_heatmap('PSNR')

def on_ssim_clicked(event):
    update_heatmap('SSIM')

def on_ocr_acc_clicked(event):
    update_heatmap('OCR_Accuracy')

def on_ocr_bin_clicked(event):
    update_heatmap('OCR_Binary')

btn_psnr.on_clicked(on_psnr_clicked)
btn_ssim.on_clicked(on_ssim_clicked)
btn_ocr_acc.on_clicked(on_ocr_acc_clicked)
btn_ocr_bin.on_clicked(on_ocr_bin_clicked)

# Connect the click event after setting up the entire figure
def on_click(event):
    if event.inaxes == ax:  # Ensure the click is within the heatmap axis
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        col = int(round(x))
        row = int(round(y))
        if 0 <= row < num_betas and 0 <= col < num_alphas:
            alpha = alpha_values[col]
            beta = beta_values[row]
            show_image_details_for(alpha, beta)

cid = fig.canvas.mpl_connect('button_press_event', on_click)  # Connect after all setups

plt.show()
