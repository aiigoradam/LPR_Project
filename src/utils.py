# utils/utils.py

import os
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from pytorch_msssim import ssim


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def log_samples(
    model: torch.nn.Module,
    distorted: torch.Tensor,
    original: torch.Tensor,
    epoch_or_step: int,
    mlflow,
    max_images: int = 8,
):
    """
    Build a 3-row grid (distorted | original | recon) and log to MLflow.
    """
    model.eval()
    reconstructed = model(distorted)

    # pull the first up to max_images
    d = distorted[:max_images]
    o = original[:max_images]
    r = reconstructed[:max_images]

    # make each row a grid (
    row_d = make_grid(d, nrow=max_images, normalize=True, scale_each=True)
    row_o = make_grid(o, nrow=max_images, normalize=True, scale_each=True)
    row_r = make_grid(r, nrow=max_images, normalize=True, scale_each=True)

    # stack rows vertically
    combined = torch.cat([row_d, row_o, row_r], dim=1).cpu()
    img = transforms.ToPILImage()(combined)

    fname = f"diffusion_step_{epoch_or_step:06}.png"
    img.save(fname)
    mlflow.log_artifact(fname)
    img.close()
    os.remove(fname)
    model.train()


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model whose inputs/outputs are scaled to [0, 1].

    Returns (mse, ssim, psnr) averaged over the whole dataset.
    """
    model.eval()
    run_loss = run_mse = run_ssim = run_psnr = 0.0

    for batch in dataloader:
        distorted = batch["distorted"].to(device)
        original = batch["original"].to(device)
        predicted = model(distorted)

        loss_b = criterion(predicted, original).item()
        mse_b = F.mse_loss(predicted, original, reduction="mean").item()
        ssim_b = ssim(predicted, original, data_range=1.0, size_average=True).item()
        psnr_b = 10 * math.log10(1.0 / mse_b)

        bs = distorted.size(0)
        run_loss += loss_b * bs
        run_mse += mse_b * bs
        run_ssim += ssim_b * bs
        run_psnr += psnr_b * bs

    n = len(dataloader.dataset)
    return (run_loss / n, run_mse / n, run_ssim / n, run_psnr / n)
