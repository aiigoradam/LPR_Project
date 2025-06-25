# src/train_pix2pix.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import argparse
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from types import SimpleNamespace
from tqdm import tqdm
from lp_dataset import LicensePlateDataset
from models.pix2pix import UNetGenerator, PatchGANDiscriminator
from utils import evaluate, log_samples, set_seed
from mlflow.models.signature import infer_signature


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
    "run_name": "pix2pix",
    "seed": 42,
    "num_epochs": 60,
    "batch_size": 32,
    "filters": 32,
    "learning_rate": 2e-4,
    "beta1": 0.5,
    "beta2": 0.999,
    "lambda_l1": 100.0,
    "save_samples_freq": 5,
}

cfg = SimpleNamespace(**config)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def lambda_rule(epoch):
    n_epochs = cfg.num_epochs
    if epoch < n_epochs // 2:
        return 1.0
    return float(n_epochs - epoch) / float(n_epochs - n_epochs // 2)


# ----------------------------------------------------------------------
# Training Function
# ----------------------------------------------------------------------
def train(
    cfg,
    device,
    generator,
    discriminator,
    g_optimizer,
    d_optimizer,
    g_scheduler,
    d_scheduler,
    train_loader,
    val_loader,
    test_loader,
    sample_distorted,
    sample_original,
):
    mlflow.set_experiment(experiment_name)

    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    l1_loss = nn.L1Loss().to(device)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params(vars(cfg))
        mlflow.log_artifact(__file__, artifact_path="code")
        mlflow.log_artifact("models/pix2pix.py", artifact_path="code")
        mlflow.log_artifact("src/utils.py", artifact_path="code")
        print(f"MLflow Run ID: {run.info.run_id}")

        params = {
            "seed": cfg.seed,
            "num_epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "filters": cfg.filters,
            "learning_rate": cfg.learning_rate,
            "beta1": cfg.beta1,
            "beta2": cfg.beta2,
            "lambda_l1": cfg.lambda_l1,
        }
        mlflow.log_params(params)

        best_val_ssim = -float("inf")
        best_val_loss = float("inf")
        best_val_psnr = -float("inf")
        best_epoch = None
        best_gen_state = None
        best_disc_state = None

        for epoch in range(1, cfg.num_epochs + 1):
            generator.train()
            discriminator.train()

            running_g_loss = 0.0
            running_d_loss = 0.0

            with tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}", unit="batch") as bar:
                for batch in bar:
                    distorted = batch["distorted"].to(device)
                    original = batch["original"].to(device)
                    bs = distorted.size(0)

                    # ----------------------------------
                    #  Train Discriminator
                    # ----------------------------------
                    discriminator.zero_grad()

                    # Real pair
                    real_pred = discriminator(distorted, original)
                    valid = torch.ones_like(real_pred, device=device)
                    d_real_loss = adversarial_loss(real_pred, valid)

                    # Fake pair (detach so G is not updated here)
                    gen_output = generator(distorted)
                    fake_pred = discriminator(distorted, gen_output.detach())
                    fake = torch.zeros_like(fake_pred, device=device)
                    d_fake_loss = adversarial_loss(fake_pred, fake)

                    d_loss = 0.5 * (d_real_loss + d_fake_loss)
                    d_loss.backward()
                    d_optimizer.step()

                    # ----------------------------------
                    #  Train Generator
                    # ----------------------------------
                    generator.zero_grad()
                    fake_pred_for_g = discriminator(distorted, gen_output)
                    valid_for_g = torch.ones_like(fake_pred_for_g, device=device)
                    g_adv = adversarial_loss(fake_pred_for_g, valid_for_g)
                    g_l1 = l1_loss(gen_output, original) * cfg.lambda_l1
                    g_loss = g_adv + g_l1
                    g_loss.backward()
                    g_optimizer.step()

                    running_d_loss += d_loss.item() * bs
                    running_g_loss += g_loss.item() * bs

                    bar.set_postfix(d_loss=f"{d_loss.item():.5f}", g_loss=f"{g_loss.item():.5f}")

            num_train = len(train_loader.dataset)
            d_loss = running_d_loss / num_train
            g_loss = running_g_loss / num_train

            # --------------
            #  Validation
            # --------------
            generator.eval()
            val_loss, val_mse, val_ssim, val_psnr = evaluate(generator, val_loader, l1_loss, device)

            g_scheduler.step()
            d_scheduler.step()

            current_lr = g_optimizer.param_groups[0]["lr"]

            metrics = {
                "epoch_g_loss": round(g_loss, 5),
                "epoch_d_loss": round(d_loss, 5),
                "val_l1_loss": round(val_loss, 5),
                "val_mse": round(val_mse, 5),
                "val_ssim": round(val_ssim, 4),
                "val_psnr": round(val_psnr, 2),
                "learning_rate": current_lr,
            }
            mlflow.log_metrics(metrics, step=epoch)

            print(
                f"{'':17s}"
                f"G Loss: {g_loss:.5f}, "
                f"D Loss: {d_loss:.5f}, "
                f"Val Loss: {val_loss:.5f}, "
                f"Val MSE: {val_mse:.5f}, "
                f"Val SSIM: {val_ssim:.4f}, "
                f"Val PSNR: {val_psnr:.2f}, "
                f"LR: {current_lr:.6f}"
            )

            if epoch % cfg.save_samples_freq == 0:
                log_samples(generator, sample_distorted, sample_original, epoch, mlflow)

            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim
                best_val_loss = val_loss
                best_val_psnr = val_psnr
                best_epoch = epoch
                best_gen_state  = copy.deepcopy(generator. state_dict())
                best_disc_state = copy.deepcopy(discriminator.state_dict())

        # ---- Post-training Evaluation ----
        print(
            f"Best checkpoint at epoch {best_epoch}: "
            f"Val L1: {best_val_loss:.5f}, "
            f"Val SSIM: {best_val_ssim:.4f}, "
            f"Val PSNR: {best_val_psnr:.2f}"
        )

        print("Loading best generator/discriminator for final evaluation.")
        generator.load_state_dict(best_gen_state)
        discriminator.load_state_dict(best_disc_state)

        # Log final model
        generator.eval()
        with torch.no_grad():
            input_example = sample_distorted
            output_example = generator(input_example)
            signature = infer_signature(input_example.cpu().numpy(), output_example.cpu().numpy())
            mlflow.pytorch.log_model(
                pytorch_model=generator, name="model", conda_env="environment.yml", signature=signature
            )
            print("Best generator logged to MLflow.")

        # ---- Test Evaluation ----
        print("Evaluating best generator on the test set...")
        _, test_mse, test_ssim, test_psnr = evaluate(generator, test_loader, l1_loss, device)
        print(f"Test Results: MSE: {test_mse:.5f}, SSIM: {test_ssim:.4f}, PSNR: {test_psnr:.2f}")
        mlflow.log_metrics(
            {
                "test_mse": round(test_mse, 5),
                "test_ssim": round(test_ssim, 4),
                "test_psnr": round(test_psnr, 2),
            }
        )

    print("Training finished.")


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

    # Initialize models
    generator = UNetGenerator(in_ch=3, out_ch=3, base=cfg.filters).to(device)
    discriminator = PatchGANDiscriminator(in_ch=3, base=cfg.filters).to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

    # Schedulers (on L1 validation loss)
    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_rule)
    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda_rule)

    # Sample batch for logging
    val_batch = next(iter(val_loader))
    sample_distorted = val_batch["distorted"].to(device)
    sample_original = val_batch["original"].to(device)

    print("Starting Pix2Pix training...")
    train(
        cfg,
        device,
        generator,
        discriminator,
        g_optimizer,
        d_optimizer,
        g_scheduler,
        d_scheduler,
        train_loader,
        val_loader,
        test_loader,
        sample_distorted,
        sample_original,
    )


if __name__ == "__main__":
    main()
