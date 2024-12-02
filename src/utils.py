# utils.py

import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from pytorch_msssim import ssim
import os

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Calculate PSNR
def calculate_psnr(outputs, targets):
    mse = F.mse_loss(outputs, targets)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

# Save sample images to MLflow
def save_sample_images(model, distorted_images, original_images, epoch, mlflow, max_images=8):
    model.eval()
    with torch.no_grad():
        outputs = model(distorted_images)

        # Limit to max_images
        distorted_images = distorted_images[:max_images]
        original_images = original_images[:max_images]
        outputs = outputs[:max_images]

        # Convert images to make_grid format and save
        distorted_grid = make_grid(distorted_images, nrow=max_images, normalize=True, scale_each=True)
        original_grid = make_grid(original_images, nrow=max_images, normalize=True, scale_each=True)
        output_grid = make_grid(outputs, nrow=max_images, normalize=True, scale_each=True)
        
        # Concatenate the grids into a single grid (dim=1 for vertical concatenation)
        combined_grid = torch.cat([distorted_grid, original_grid, output_grid], dim=1)
        
        # Save the combined grid as an image artifact
        combined_grid_cpu = combined_grid.cpu()
        img = transforms.ToPILImage()(combined_grid_cpu)
        img_path = f"combined_epoch_{epoch:02}.png"
        img.save(img_path)
        mlflow.log_artifact(img_path)
        os.remove(img_path)  # Clean up after logging
    model.train()

# Evaluate model on validation or test set
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    running_ssim = 0.0
    running_psnr = 0.0

    with torch.no_grad():
        for batch in dataloader:
            distorted_images = batch['distorted'].to(device)
            original_images = batch['original'].to(device)

            outputs = model(distorted_images)
            loss = criterion(outputs, original_images).item()
            mse_value = loss
            ssim_value = ssim(outputs, original_images, data_range=1.0, size_average=True).item()
            psnr_value = calculate_psnr(outputs, original_images)

            batch_size_actual = distorted_images.size(0)
            running_loss += loss * batch_size_actual
            running_mse += mse_value * batch_size_actual
            running_ssim += ssim_value * batch_size_actual
            running_psnr += psnr_value * batch_size_actual

    num_samples = len(dataloader.dataset)
    val_loss = running_loss / num_samples
    val_mse = running_mse / num_samples
    val_ssim = running_ssim / num_samples
    val_psnr = running_psnr / num_samples

    return val_loss, val_mse, val_ssim, val_psnr

