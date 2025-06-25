# src/train_unet.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from mlflow.models.signature import infer_signature
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from types import SimpleNamespace
from tqdm import tqdm
from lp_dataset import LicensePlateDataset
from models.unet import UNet
from utils import evaluate, log_samples, set_seed


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
    "run_name": "unet_base",
    "seed": 42,
    "num_epochs": 40,
    "batch_size": 32,
    "filters": 32,
    "learning_rate": 0.001,
    "weight_decay": 5e-5,
    "save_samples_freq": 5,
}

cfg = SimpleNamespace(**config)


# ----------------------------------------------------------------------
# Training Function
# ----------------------------------------------------------------------
def train(
    cfg,
    device,
    model,
    criterion,
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
        mlflow.log_artifact("models/unet.py", artifact_path="code")
        mlflow.log_artifact("src/utils.py", artifact_path="code")
        print(f"MLflow Run ID: {run.info.run_id}")

        best_val_ssim = -float("inf")
        best_val_loss = float("inf")
        best_val_psnr = -float("inf")
        best_epoch = None
        best_state = None

        # --- Training Loop ---
        for epoch in range(1, cfg.num_epochs + 1):
            model.train()
            running_loss = 0.0

            # --- Training ---
            with tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}", unit="batch") as bar:
                for batch in bar:
                    distorted_images = batch["distorted"].to(device)
                    original_images = batch["original"].to(device)

                    optimizer.zero_grad()
                    outputs = model(distorted_images)
                    loss = criterion(outputs, original_images)
                    loss.backward()
                    optimizer.step()

                    bs = distorted_images.size(0)
                    running_loss += loss.item() * bs
                    bar.set_postfix(loss=f"{loss.item():.5f}")

            num_train_samples = len(train_loader.dataset)
            train_loss = running_loss / num_train_samples

            # --- Validation ---
            val_loss, val_mse, val_ssim, val_psnr = evaluate(model, val_loader, criterion, device)

            current_lr = optimizer.param_groups[0]["lr"]

            scheduler.step()

            # Log metrics
            metrics = {
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_mse": round(val_mse, 5),
                "val_ssim": round(val_ssim, 4),
                "val_psnr": round(val_psnr, 2),
                "learning_rate": current_lr,
            }
            mlflow.log_metrics(metrics, step=epoch)

            print(
                f"{'':17s}"
                f"Train Loss: {train_loss:.5f}, "
                f"Val Loss: {val_loss:.5f}, "
                f"Val MSE: {val_mse:.5f}, "
                f"Val SSIM: {val_ssim:.4f}, "
                f"Val PSNR: {val_psnr:.2f}, "
                f"LR: {current_lr:.6f}"
            )

            # Save sample images periodically
            if epoch % cfg.save_samples_freq == 0:
                log_samples(model, sample_distorted, sample_original, epoch, mlflow)

            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim
                best_val_loss = val_loss
                best_val_psnr = val_psnr
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        # --- Post‑training Evaluation ---
        print(
            f"Best checkpoint at epoch {best_epoch}: "
            f"Val Loss: {best_val_loss:.4f}, "
            f"Val SSIM: {best_val_ssim:.4f}, "
            f"Val PSNR: {best_val_psnr:.2f}"
        )

        print(f"\nLoading best model state for final evaluation.")
        model.load_state_dict(best_state)

        # Log final model
        model.eval()
        with torch.no_grad():
            input_example = sample_distorted
            output_example = model(input_example)
            signature = infer_signature(input_example.cpu().numpy(), output_example.cpu().numpy())
            mlflow.pytorch.log_model(
                pytorch_model=model, name="model", conda_env="environment.yml", signature=signature
            )
            print("Best model logged to MLflow.")

        # --- Test Evaluation ---
        print("Evaluating best model on the test set...")
        _, test_mse, test_ssim, test_psnr = evaluate(model, test_loader, criterion, device)

        print(f"Test Results: MSE: {test_mse:.5f}, SSIM: {test_ssim:.4f}, PSNR: {test_psnr:.2f}")

        # Log final test metrics
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

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = LicensePlateDataset(image_source=os.path.join(data_dir, "train"), transform=transform)
    val_dataset = LicensePlateDataset(image_source=os.path.join(data_dir, "val"), transform=transform)
    test_dataset = LicensePlateDataset(image_source=os.path.join(data_dir, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    model = UNet(in_channels=3, out_channels=3, base=cfg.filters).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    sample_batch = next(iter(val_loader))
    sample_distorted = sample_batch["distorted"].to(device)
    sample_original = sample_batch["original"].to(device)

    print("Starting training process...")
    train(
        cfg,
        device,
        model,
        criterion,
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
