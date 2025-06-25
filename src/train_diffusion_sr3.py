# src/train_diffusion_sr3.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import argparse
import math
import mlflow
import torch
import torch.optim as optim
import torch.nn.functional as F
from mlflow.models.signature import infer_signature
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from types import SimpleNamespace
from lp_dataset import LicensePlateDataset
from models.diffusion_sr3 import UNetDiffusion, Diffusion
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import set_seed
from pytorch_msssim import ssim


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", default="Test")
parser.add_argument("--data-dir", default=os.path.join("data", "A"))
args = parser.parse_args()
experiment_name = args.experiment_name
data_dir = args.data_dir

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
config = {
    "run_name": "diffusion_sr3",
    "seed": 42,
    "max_steps": 10000,
    "batch_size": 32,
    "filters": 32,
    "time_dim": 320,
    "learning_rate": 1e-4,
    "gn_groups": 8,
    "grad_clip": 1.0,
    "save_samples_every": 1000,
    "T": 100,
    "sample_steps": 10,
}
cfg = SimpleNamespace(**config)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

@torch.no_grad()
def log_samples(
    distorted: torch.Tensor,
    original: torch.Tensor,
    reconstructions: torch.Tensor,
    epoch_or_step: int,
    mlflow,
    max_images: int = 8,
):
    """
    Build a 3-row grid (distorted | original | recon) and log to MLflow.
    """
    # pull the first up to max_images
    d = distorted[:max_images]
    o = original[:max_images]
    r = reconstructions[:max_images]

    # make each row a grid
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


@torch.no_grad()
def evaluate(model, diffusion, dataloader, device):
    """
    Evaluate model; inputs/outputs are scaled to [-1,1].

    Returns (mse, ssim, psnr) averaged over the whole dataset.
    """
    model.eval()
    run_mse = run_ssim = run_psnr = 0.0

    for batch in dataloader:
        xd = batch["distorted"].to(device)
        x0 = batch["original"].to(device)
        rec = diffusion.sample(model, xd, steps=cfg.sample_steps)

        rec_u = ((rec + 1.0) * 0.5).clamp(0.0, 1.0)  # unormalize [-1,1] -> [0,1]
        x0_u = ((x0 + 1.0) * 0.5).clamp(0.0, 1.0)  # unormalize [-1,1] -> [0,1]

        mse_b = F.mse_loss(rec_u, x0_u, reduction="mean").item()
        ssim_b = ssim(rec_u, x0_u, data_range=1.0, size_average=True).item()
        psnr_b = 10 * math.log10(1.0 / mse_b)

        bs = xd.size(0)
        run_mse += mse_b * bs
        run_ssim += ssim_b * bs
        run_psnr += psnr_b * bs

    n = len(dataloader.dataset)
    return (run_mse / n, run_ssim / n, run_psnr / n)


# ----------------------------------------------------------------------
# Training routine
# ----------------------------------------------------------------------
def train(
    cfg,
    device,
    model,
    diffusion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    sample_distorted,
    sample_original,
):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params(vars(cfg))
        mlflow.log_artifact(__file__, artifact_path="code")
        mlflow.log_artifact("models/diffusion_sr3.py", artifact_path="code")
        mlflow.log_artifact("src/utils.py", artifact_path="code")
        print("MLflow run:", run.info.run_id)

        best_val_ssim = -float("inf")
        best_val_psnr = -float("inf")
        best_state = None

        # --- Training Loop ---
        step_iter = iter(train_loader)
        pbar = tqdm(range(1, cfg.max_steps + 1), desc="train-steps")
        for step in pbar:
            try:
                batch = next(step_iter)
            except StopIteration:
                step_iter = iter(train_loader)
                batch = next(step_iter)

            x0 = batch["original"].to(device)
            xd = batch["distorted"].to(device)

            loss = diffusion.p_losses(model, x0, xd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics({"loss": loss.item(), "learning_rate": current_lr}, step=step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # --- Validation ---
            if step % cfg.save_samples_every == 0:
                model.eval()

                rec = diffusion.sample(model, sample_distorted, steps=cfg.sample_steps)
                log_samples(sample_distorted, sample_original, rec, step, mlflow)
                val_mse, val_ssim, val_psnr = evaluate(model, diffusion, val_loader, device)

                mlflow.log_metrics({"val_mse": val_mse, "val_ssim": val_ssim, "val_psnr": val_psnr}, step=step)

                # Log metrics
                metrics = {
                    "val_mse": round(val_mse, 5),
                    "val_ssim": round(val_ssim, 4),
                    "val_psnr": round(val_psnr, 2),
                }
                mlflow.log_metrics(metrics, step=step)

                improved = False
                if val_ssim > best_val_ssim:
                    best_val_ssim = val_ssim
                    improved = True
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    improved = True
                if improved:
                    best_state = copy.deepcopy(model.state_dict())

                model.train()

        # --- Post‑training Evaluation ---
        print(f"Val SSIM: {best_val_ssim:.4f}, " f"Val PSNR: {best_val_psnr:.2f}")

        print(f"\nLoading best model state for final evaluation.")
        model.load_state_dict(best_state)

        # Log final model
        model.eval()
        with torch.no_grad():
            x_example = torch.randn_like(sample_distorted, device=device)
            d_example = sample_distorted
            t_example = torch.zeros(x_example.size(0), dtype=torch.long, device=device)
            out_example = model(x_example, d_example, t_example)
            input_example = {"x_t": x_example.cpu().numpy(), "d": d_example.cpu().numpy(), "t": t_example.cpu().numpy()}
            output_example = out_example.cpu().numpy()
            signature = infer_signature(input_example, output_example)
            mlflow.pytorch.log_model(
                pytorch_model=model, name="model", conda_env="environment.yml", signature=signature
            )

        # --- Test Evaluation ---
        print("Evaluating best model on the test set...")
        test_mse, test_ssim, test_psnr = evaluate(model, diffusion, test_loader, device)
        print(f"Test Results: MSE: {test_mse:.5f}, SSIM: {test_ssim:.4f}, PSNR: {test_psnr:.2f}")
        mlflow.log_metrics(
            {"test_mse": round(test_mse, 5), "test_ssim": round(test_ssim, 4), "test_psnr": round(test_psnr, 2)}
        )

    print("Training function finished.")


# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------
def main():
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = LicensePlateDataset(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = LicensePlateDataset(os.path.join(data_dir, "val"), transform=transform)
    test_dataset = LicensePlateDataset(os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    model = UNetDiffusion(base=cfg.filters, time_dim=cfg.time_dim, gn_groups=cfg.gn_groups).to(device)
    diffusion = Diffusion(T=cfg.T, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps, eta_min=1e-6)

    sample_batch = next(iter(val_loader))
    sample_distorted = sample_batch["distorted"].to(device)
    sample_original = sample_batch["original"].to(device)

    print("Starting training process...")
    train(
        cfg,
        device,
        model,
        diffusion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        test_loader,
        sample_distorted,
        sample_original,
    )


if __name__ == "__main__":
    main()
