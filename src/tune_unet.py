# src/ tune_unet.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import threading
import optuna
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.models.signature import infer_signature
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from types import SimpleNamespace
from lp_dataset import LicensePlateDataset
from models.unet import UNet
from utils import evaluate_model, save_sample_images, set_seed

# =====================================
# Configuration
# =====================================

config = {
    # --- experiment / search ---
    "experiment_name": "Unet_Tuning",
    "n_trials": 20,
    "seed": 42,
    # --- training hyper‑params ---
    "num_epochs": 40,
    "batch_sizes": [16, 32],
    "learning_rate_range": (1e-4, 1e-3),
    "weight_decay_range": (5e-6, 5e-5),
    # --- optimiser / criterion / scheduler ---
    "optimizer": "AdamW",
    "criterion": "MSELoss",
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    # --- misc ---
    "save_samples_freq": 3,
    "storage_url": f"sqlite:///{os.path.abspath('optuna_study_MSEloss.db')}",
    "pruner": optuna.pruners.MedianPruner(),
}

cfg = SimpleNamespace(**config)


def handle_pruning(trial, epoch):
    print(f"Pruning trial {trial.number} at epoch {epoch}.")
    mlflow.set_tag("status", "pruned")
    mlflow.end_run(status="KILLED")
    raise optuna.exceptions.TrialPruned()


# =====================================
# Objective Function for Optuna
# =====================================
def objective(trial):
    # --- hyper-params from Optuna ---
    lr = trial.suggest_float("learning_rate", *cfg.learning_rate_range, log=True)
    wd = trial.suggest_float("weight_decay", *cfg.weight_decay_range, log=True)
    bs = trial.suggest_categorical("batch_size", cfg.batch_sizes)

    # --- data loaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )

    # --- model, criterion, optimizer, scheduler ---
    model = UNet(3, 3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
    )

    # --- fixed sample for logging ---
    sample = next(iter(val_loader))
    samp_dist = sample["distorted"].to(device)
    samp_orig = sample["original"].to(device)

    # --- prepare MLflow run ---
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=f"trial_{trial.number:02d}"):

        mlflow.log_params(
            {
                "learning_rate": lr,
                "weight_decay" : wd,
                "batch_size"   : bs,
                "optimizer"    : cfg.optimizer,
                "criterion"    : cfg.criterion,
                "scheduler"    : cfg.scheduler,
                "seed"         : cfg.seed,
                "num_epochs"   : cfg.num_epochs,
                "train_size"   : len(train_loader.dataset),
                "val_size"     : len(val_loader.dataset),
                "test_size"    : len(test_loader.dataset),
            }
        )
        trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)

        # --- metric tracking ---
        best_val_ssim          = -float("inf")
        best_val_loss  = float("inf")
        best_epoch             = None
        best_model_state       = None

        # --- epoch loop ---
        for epoch in range(cfg.num_epochs):
            model.train()
            running_loss = 0.0

            try:
                with tqdm(
                    train_loader, desc=f"Trial {trial.number:02d}  Epoch {epoch+1}/{cfg.num_epochs}", unit="batch"
                ) as bar:
                    for batch in bar:
                        x = batch["distorted"].to(device)
                        y = batch["original"].to(device)

                        optimizer.zero_grad()
                        out = model(x)
                        loss = criterion(out, y)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * x.size(0)
                        bar.set_postfix(loss=f"{loss.item():.5f}")
            except KeyboardInterrupt:
                print(f"Trial {trial.number:02d} interrupted at epoch {epoch+1}.")
                stop_after_trial.set()
                break

            # --- metrics & scheduler ---
            train_loss = running_loss / len(train_loader.dataset)
            val_loss, val_mse, val_ssim, val_psnr = evaluate_model(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics(
                {
                    "train_loss"   : round(train_loss, 5),
                    "val_loss"     : round(val_loss, 4),
                    "val_mse"      : round(val_mse, 4),
                    "val_ssim"     : round(val_ssim, 4),
                    "val_psnr"     : round(val_psnr, 2),
                    "learning_rate": lr_now,
                },
                step=epoch,
            )

            # --- sample images, pruning ---
            if epoch % cfg.save_samples_freq == 0:
                save_sample_images(model, samp_dist, samp_orig, epoch, mlflow)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                handle_pruning(trial, epoch)

            # --- checkpointing ---
            if val_ssim > best_val_ssim:
                best_val_ssim         = val_ssim
                best_val_loss = val_loss
                best_epoch            = epoch
                best_model_state      = model.state_dict()

            if stop_after_trial.is_set():
                break

        # --- end of trial: single summary print + final logging ---
        if best_model_state is not None:
            print(
                f"Trial {trial.number:02d} best checkpoint: "
                f"epoch {best_epoch+1}, val_loss={best_val_loss:.4f}, "
                f"val_ssim={best_val_ssim:.4f}"
            )

            # load & log that model artifact
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                out = model(samp_dist)
            sig = infer_signature(samp_dist.cpu().numpy(), out.cpu().numpy())
            mlflow.pytorch.log_model(model, artifact_path="model", signature=sig)

            # test‐set metrics
            t_loss, t_mse, t_ssim, t_psnr = evaluate_model(model, test_loader, criterion, device)
            mlflow.log_metrics(
                {
                    "test_loss": round(t_loss, 4),
                    "test_mse": round(t_mse, 4),
                    "test_ssim": round(t_ssim, 4),
                    "test_psnr": round(t_psnr, 2),
                }
            )
        else:
            print(f"Trial {trial.number:02d} found no valid checkpoint.")

        return best_val_loss


# =====================================
# Main Function
# =====================================


def main():
    global device
    global stop_after_trial
    global train_dataset, val_dataset, test_dataset

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = LicensePlateDataset(image_source="data/train.h5", transform=transform)
    val_dataset = LicensePlateDataset(image_source="data/val.h5", transform=transform)
    test_dataset = LicensePlateDataset(image_source="data/test.h5", transform=transform)

    print(f"Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    stop_after_trial = threading.Event()

    # Set the MLflow experiment
    mlflow.set_experiment(cfg.experiment_name)

    # Define or resume the Optuna study
    study = optuna.create_study(
        study_name=cfg.experiment_name,
        storage=cfg.storage_url,
        direction="minimize",
        pruner=cfg.pruner,
        load_if_exists=True,
    )

    for _ in range(cfg.n_trials):
        if stop_after_trial.is_set():
            print("Stop requested. Stopping the study after the current trial.")
            break

        study.optimize(objective, n_trials=1)

    print("Main function finished.")


if __name__ == "__main__":
    main()
