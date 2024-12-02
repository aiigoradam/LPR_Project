# evaluation.py

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_msssim import ssim
import mlflow
import mlflow.pytorch
import cv2

# Add the parent directory to the Python path if necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming your LicensePlateDataset and UNet are defined in the same script or properly imported
from models.unet import UNet
from lp_dataset import LicensePlateDataset

# =====================================
# Helper Functions
# =====================================

def calculate_psnr_torch(img1, img2):
    """
    Calculate PSNR between two images using PyTorch tensors.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0  # Assuming images are normalized between 0 and 1
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim_torch(img1, img2):
    """
    Calculate SSIM between two images using pytorch_msssim.
    """
    ssim_value = ssim(img1, img2, data_range=1.0, size_average=True)
    return ssim_value.item()

def perform_ocr(image_np):
    """
    Perform OCR using Tesseract on a NumPy image array.
    """
    # Convert to PIL Image
    pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
    # OCR configuration
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    # Clean the text
    text = ''.join(filter(str.isdigit, text))
    return text

def compute_ocr_accuracy(gt_text, pred_text):
    """
    Compute OCR accuracy as the proportion of correctly recognized digits.
    """
    min_len = min(len(gt_text), len(pred_text))
    correct = sum(1 for a, b in zip(gt_text[:min_len], pred_text[:min_len]) if a == b)
    accuracy = correct / len(gt_text) if len(gt_text) > 0 else 0
    return accuracy

def plot_heatmap(data_matrix, alpha_values, beta_values, title, xlabel='Alpha', ylabel='Beta', cmap='viridis'):
    """
    Plot a heatmap using Seaborn.
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(data_matrix, xticklabels=alpha_values, yticklabels=beta_values, annot=True, fmt='.2f', cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# =====================================
# Dataset Class
# =====================================

class LicensePlateDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.indices = self._get_indices()
    
    def _get_indices(self):
        # Get the list of indices based on existing files
        files = os.listdir(self.image_dir)
        indices = sorted(set(int(f.split('_')[1].split('.')[0]) for f in files if f.endswith('.png')))
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        # Construct file paths
        original_path = os.path.join(self.image_dir, f"original_{actual_idx}.png")
        distorted_path = os.path.join(self.image_dir, f"distorted_{actual_idx}.png")
        metadata_path = os.path.join(self.image_dir, f"metadata_{actual_idx}.json")

        # Load images
        original = Image.open(original_path).convert('RGB')
        distorted = Image.open(distorted_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            original_tensor = self.transform(original)
            distorted_tensor = self.transform(distorted)
        else:
            original_tensor = original
            distorted_tensor = distorted
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            'original': original_tensor,
            'distorted': distorted_tensor,
            'metadata': metadata
        }

# =====================================
# Main Evaluation Function
# =====================================

def main():
    # Set device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set the MLflow experiment and load the model
    mlflow.set_experiment('Unet_Final')

    # Get the last run ID from the experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name('Unet_Final')
    runs = client.search_runs(experiment_ids=experiment.experiment_id, order_by=["attributes.start_time DESC"])
    run_id = runs[0].info.run_id  # Get the last run

    # Load the model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    model.to(device)
    print("Model loaded successfully.")

    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL Image to Tensor and scales pixel values to [0, 1]
    ])

    # Create the test dataset and DataLoader
    test_dataset = LicensePlateDataset(image_dir='data_test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test dataset loaded with {len(test_dataset)} samples.")

    # Initialize data structures for metrics
    metrics_data = []

    # Define noise level threshold for categorization
    moderate_noise_threshold = 100  # Adjust this threshold based on your data distribution

    # Iterate over the test set
    for sample in test_loader:
        distorted_images = sample['distorted'].to(device)  # Shape: [1, C, H, W]
        original_images = sample['original'].to(device)
        metadata = sample['metadata']

        # Get metadata values
        idx = metadata['idx']
        alpha = metadata['alpha']
        beta = metadata['beta']
        noise_level = metadata['noise_level']
        digit_bboxes = metadata['digit_bboxes']
        plate_number = metadata['plate_number']

        # Generate enhanced image
        with torch.no_grad():
            outputs = model(distorted_images)

        # Convert tensors to NumPy arrays for processing
        distorted_image_np = distorted_images.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Shape: H x W x C
        original_image_np = original_images.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_image_np = outputs.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Ensure images are in [0, 1] range
        distorted_image_np = np.clip(distorted_image_np, 0, 1)
        original_image_np = np.clip(original_image_np, 0, 1)
        enhanced_image_np = np.clip(enhanced_image_np, 0, 1)

        # Calculate per-digit PSNR and SSIM
        psnr_values = []
        ssim_values = []

        for bbox in digit_bboxes:
            x, y, w, h = map(int, bbox)
            x2, y2 = x + w, y + h
            # Ensure the bounding box is within image bounds
            x, y = max(x, 0), max(y, 0)
            x2, y2 = min(x2, original_image_np.shape[1]), min(y2, original_image_np.shape[0])

            # Extract digit regions
            original_digit_np = original_image_np[y:y2, x:x2, :]
            enhanced_digit_np = enhanced_image_np[y:y2, x:x2, :]

            # Convert to tensors
            original_digit_tensor = torch.from_numpy(original_digit_np.transpose(2, 0, 1)).unsqueeze(0).to(device)  # Shape: [1, C, H, W]
            enhanced_digit_tensor = torch.from_numpy(enhanced_digit_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

            # Compute PSNR
            psnr_value = calculate_psnr_torch(original_digit_tensor, enhanced_digit_tensor)
            psnr_values.append(psnr_value)

            # Compute SSIM
            ssim_value = calculate_ssim_torch(original_digit_tensor, enhanced_digit_tensor)
            ssim_values.append(ssim_value)

        # Use the worst PSNR and SSIM per license plate
        worst_psnr = min(psnr_values)
        worst_ssim = min(ssim_values)

        # Perform OCR on the enhanced image
        ocr_text = perform_ocr(enhanced_image_np)
        # Compute OCR accuracy
        ocr_accuracy = compute_ocr_accuracy(plate_number, ocr_text)

        # Classify noise level
        noise_category = 'Moderate' if noise_level <= moderate_noise_threshold else 'High'

        # Collect the data
        metrics_data.append({
            'alpha': alpha,
            'beta': beta,
            'noise_level': noise_level,
            'noise_category': noise_category,
            'worst_psnr': worst_psnr,
            'worst_ssim': worst_ssim,
            'ocr_accuracy': ocr_accuracy
        })

    print("Evaluation on test set completed.")

    # Prepare data for heatmaps
    # Collect unique alpha and beta values
    alpha_values = sorted(set(item['alpha'] for item in metrics_data))
    beta_values = sorted(set(item['beta'] for item in metrics_data))

    # Create mappings from alpha and beta to indices
    alpha_to_idx = {alpha: idx for idx, alpha in enumerate(alpha_values)}
    beta_to_idx = {beta: idx for idx, beta in enumerate(beta_values)}

    # Initialize matrices
    num_alphas = len(alpha_values)
    num_betas = len(beta_values)

    # For moderate noise
    moderate_psnr_matrix = np.full((num_betas, num_alphas), np.nan)
    moderate_ssim_matrix = np.full((num_betas, num_alphas), np.nan)
    moderate_ocr_matrix = np.full((num_betas, num_alphas), np.nan)

    # For high noise
    high_psnr_matrix = np.full((num_betas, num_alphas), np.nan)
    high_ssim_matrix = np.full((num_betas, num_alphas), np.nan)
    high_ocr_matrix = np.full((num_betas, num_alphas), np.nan)

    # Populate the matrices
    for item in metrics_data:
        alpha = item['alpha']
        beta = item['beta']
        alpha_idx = alpha_to_idx[alpha]
        beta_idx = beta_to_idx[beta]
        noise_category = item['noise_category']

        if noise_category == 'Moderate':
            # PSNR
            if np.isnan(moderate_psnr_matrix[beta_idx, alpha_idx]):
                moderate_psnr_matrix[beta_idx, alpha_idx] = item['worst_psnr']
            else:
                moderate_psnr_matrix[beta_idx, alpha_idx] = np.mean([moderate_psnr_matrix[beta_idx, alpha_idx], item['worst_psnr']])
            # SSIM
            if np.isnan(moderate_ssim_matrix[beta_idx, alpha_idx]):
                moderate_ssim_matrix[beta_idx, alpha_idx] = item['worst_ssim']
            else:
                moderate_ssim_matrix[beta_idx, alpha_idx] = np.mean([moderate_ssim_matrix[beta_idx, alpha_idx], item['worst_ssim']])
            # OCR Accuracy
            if np.isnan(moderate_ocr_matrix[beta_idx, alpha_idx]):
                moderate_ocr_matrix[beta_idx, alpha_idx] = item['ocr_accuracy']
            else:
                moderate_ocr_matrix[beta_idx, alpha_idx] = np.mean([moderate_ocr_matrix[beta_idx, alpha_idx], item['ocr_accuracy']])
        elif noise_category == 'High':
            # PSNR
            if np.isnan(high_psnr_matrix[beta_idx, alpha_idx]):
                high_psnr_matrix[beta_idx, alpha_idx] = item['worst_psnr']
            else:
                high_psnr_matrix[beta_idx, alpha_idx] = np.mean([high_psnr_matrix[beta_idx, alpha_idx], item['worst_psnr']])
            # SSIM
            if np.isnan(high_ssim_matrix[beta_idx, alpha_idx]):
                high_ssim_matrix[beta_idx, alpha_idx] = item['worst_ssim']
            else:
                high_ssim_matrix[beta_idx, alpha_idx] = np.mean([high_ssim_matrix[beta_idx, alpha_idx], item['worst_ssim']])
            # OCR Accuracy
            if np.isnan(high_ocr_matrix[beta_idx, alpha_idx]):
                high_ocr_matrix[beta_idx, alpha_idx] = item['ocr_accuracy']
            else:
                high_ocr_matrix[beta_idx, alpha_idx] = np.mean([high_ocr_matrix[beta_idx, alpha_idx], item['ocr_accuracy']])

    # Generate heatmaps
    # PSNR Heatmaps
    plot_heatmap(moderate_psnr_matrix, alpha_values, beta_values, 'Worst PSNR per Digit (Moderate Noise)', cmap='coolwarm')
    plot_heatmap(high_psnr_matrix, alpha_values, beta_values, 'Worst PSNR per Digit (High Noise)', cmap='coolwarm')

    # SSIM Heatmaps
    plot_heatmap(moderate_ssim_matrix, alpha_values, beta_values, 'Worst SSIM per Digit (Moderate Noise)', cmap='coolwarm')
    plot_heatmap(high_ssim_matrix, alpha_values, beta_values, 'Worst SSIM per Digit (High Noise)', cmap='coolwarm')

    # OCR Accuracy Heatmaps
    plot_heatmap(moderate_ocr_matrix, alpha_values, beta_values, 'OCR Accuracy per License Plate (Moderate Noise)', cmap='coolwarm')
    plot_heatmap(high_ocr_matrix, alpha_values, beta_values, 'OCR Accuracy per License Plate (High Noise)', cmap='coolwarm')

    print("Heatmaps generated successfully.")

if __name__ == '__main__':
    main()
