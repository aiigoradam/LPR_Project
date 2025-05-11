# src/train_unet.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.models.signature import infer_signature
from torch.utils.data import DataLoader
from torchvision import transforms
from types import SimpleNamespace
from tqdm import tqdm
from lp_dataset import LicensePlateDataset
from models.unet import UNet
from utils import evaluate_model, save_sample_images, set_seed

# This is to disable the console control handler in Windows
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

# =====================================
# Configuration Dictionary
# =====================================

config = {
    "experiment_name": "Unet",
    "run_name": "run_03",
    "seed": 42,
    "num_epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "AdamW",
    "criterion": "MSELoss",
    "scheduler": "ReduceLROnPlateau",
    "weight_decay": 5e-5,
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "save_samples_freq": 3,
}

cfg = SimpleNamespace(**config)

# =====================================
# Training Function
# =====================================


def train_model(
    cfg,
    device,
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    sample_distorted_images,
    sample_original_images,
):
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        print(f"MLflow Run ID: {run.info.run_id}")

        params = {
            "seed": cfg.seed,
            "num_epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "optimizer": cfg.optimizer,
            "criterion": cfg.criterion,
            "scheduler": cfg.scheduler,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
        }

        mlflow.log_params(params)

        best_val_ssim = -float("inf")
        best_val_loss = float("inf")
        best_epoch = None
        best_model_state = None

        # --- Training Loop ---
        try:
            for epoch in range(cfg.num_epochs):
                model.train()
                running_loss = 0.0

                # --- Training ---
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", unit="batch") as bar:
                    for batch in bar:
                        distorted_images = batch["distorted"].to(device)
                        original_images = batch["original"].to(device)

                        optimizer.zero_grad()
                        outputs = model(distorted_images)
                        loss = criterion(outputs, original_images)
                        loss.backward()
                        optimizer.step()

                        batch_size_actual = distorted_images.size(0)
                        running_loss += loss.item() * batch_size_actual
                        bar.set_postfix(loss=f"{loss.item():.5f}")

                num_train_samples = len(train_loader.dataset)
                train_loss = running_loss / num_train_samples

                # --- Validation ---
                val_loss, val_mse, val_ssim, val_psnr = evaluate_model(model, val_loader, criterion, device)

                current_lr = optimizer.param_groups[0]["lr"]
                if scheduler:
                    scheduler.step(val_loss)

                # Log metrics
                metrics = {
                    "train_loss": round(train_loss, 5),
                    "val_loss": round(val_loss, 4),
                    "val_mse": round(val_mse, 4),
                    "val_ssim": round(val_ssim, 4),
                    "val_psnr": round(val_psnr, 2),
                    "learning_rate": current_lr,
                }
                mlflow.log_metrics(metrics, step=epoch)

                print(
                    f"{'':17s}Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.4f}, "
                    f"Val MSE: {val_mse:.5f}, Val SSIM: {val_ssim:.4f}, Val PSNR: {val_psnr:.2f}, "
                    f"LR: {current_lr:.6f}"
                )

                # Save sample images periodically
                if epoch % cfg.save_samples_freq == 0 and sample_distorted_images is not None:
                    save_sample_images(model, sample_distorted_images, sample_original_images, epoch, mlflow)

                # Save the best model based on validation loss and SSIM
                if val_ssim > best_val_ssim:
                    best_val_ssim = val_ssim
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_state = model.state_dict()

        except KeyboardInterrupt:
            print("Training interrupted by user.")

        # --- Post‑training Evaluation ---
        if best_model_state is not None:
            print(
                f"Best checkpoint at epoch {best_epoch+1}: "
                f"Val Loss: {best_val_loss:.4f}, "
                f"Val SSIM: {best_val_ssim:.4f}"
            )
        else:
            print("No best model was saved.")

        print(f"\nLoading best model state for final evaluation.")
        model.load_state_dict(best_model_state)

        # Log final model artifact
        model.eval()
        with torch.no_grad():
            input_example = sample_distorted_images
            output_example = model(input_example)
            signature = infer_signature(input_example.cpu().numpy(), output_example.cpu().numpy())
            mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model", signature=signature)
            print("Best model logged to MLflow.")

        # --- Test Evaluation ---
        print("Evaluating best model on the test set...")
        test_loss, test_mse, test_ssim, test_psnr = evaluate_model(model, test_loader, criterion, device)

        print(
            f"Test Results: Loss: {test_loss:.4f}, MSE: {test_mse:.5f}, "
            f"SSIM: {test_ssim:.4f}, PSNR: {test_psnr:.2f}"
        )

        # Log final test metrics
        mlflow.log_metrics(
            {
                "test_loss": round(test_loss, 4),
                "test_mse": round(test_mse, 4),
                "test_ssim": round(test_ssim, 4),
                "test_psnr": round(test_psnr, 2),
            }
        )

    print("Training function finished.")


# =====================================
# Main Function
# =====================================


def main():
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = LicensePlateDataset(image_source="data/train.h5", transform=transform)
    val_dataset = LicensePlateDataset(image_source="data/val.h5", transform=transform)
    test_dataset = LicensePlateDataset(image_source="data/test.h5", transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )

    print(f"Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    model = UNet(in_channels=3, out_channels=3, features=32).to(device)

    if cfg.criterion == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported criterion: {cfg.criterion}")

    if cfg.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    scheduler = None
    if cfg.scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
        )
    elif cfg.scheduler is not None:
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")

    print("Loading sample batch for visualization...")
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        print("Skipping training due to missing sample data.")
        return

    sample_distorted = batch["distorted"].to(device)
    sample_original = batch["original"].to(device)

    print("Starting training process...")
    train_model(
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

    print("Main function finished.")


if __name__ == "__main__":
    main()
