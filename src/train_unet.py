import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_msssim import SSIM
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from tqdm import tqdm
import optuna
import mlflow
from mlflow.models.signature import infer_signature

# Add the parent directory of src (LPR_Project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet
from license_plate_dataset import LicensePlateDataset

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Compute combined loss and individual metrics
def compute_loss(outputs, targets, mse_loss_fn, ssim_loss_fn, p):
    mse_loss = mse_loss_fn(outputs, targets)
    ssim_value = ssim_loss_fn(outputs, targets)
    ssim_loss = 1 - ssim_value  # Since SSIM is between 0 and 1
    loss = p * mse_loss + (1 - p) * ssim_loss
    return loss, mse_loss.item(), ssim_value.item()

# Calculate PSNR
def calculate_psnr(outputs, targets):
    mse = F.mse_loss(outputs, targets)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

# Save sample images to MLflow
def save_sample_images(model, distorted_images, original_images, epoch, max_images=8):
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
        img_path = f"combined_epoch_{epoch}.png"
        img.save(img_path)
        mlflow.log_artifact(img_path)
        os.remove(img_path)  # Clean up after logging
    model.train()

# Evaluate model on validation or test set
def evaluate_model(model, dataloader, mse_loss_fn, ssim_loss_fn, ssim_mse_weight):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    with torch.no_grad():
        for batch in dataloader:
            distorted_images = batch['distorted'].to(device)
            original_images = batch['original'].to(device)

            outputs = model(distorted_images)

            loss, mse_value, ssim_value = compute_loss(outputs, original_images, mse_loss_fn, ssim_loss_fn, ssim_mse_weight)
            psnr_value = calculate_psnr(outputs, original_images)

            batch_size = distorted_images.size(0)
            total_loss += loss.item() * batch_size
            total_mse += mse_value * batch_size
            total_ssim += ssim_value * batch_size
            total_psnr += psnr_value * batch_size

    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    avg_mse = total_mse / num_samples
    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples

    return avg_loss, avg_mse, avg_ssim, avg_psnr

# Objective function for Optuna
def objective(trial, train_dataset, val_dataset, train_size, val_size, test_size, seed):
    # Hyperparameters to tune
    num_epochs = 3 
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ssim_mse_weight = trial.suggest_float('ssim_mse_weight', 0.0, 1.0)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss functions, and optimizer
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion_mse = nn.MSELoss()
    criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Preload a sample batch from val_loader for logging and visualization
    sample_batch = next(iter(val_loader))
    sample_distorted_images = sample_batch['distorted'].to(device)
    sample_original_images = sample_batch['original'].to(device)
    
    # Start MLflow run for this trial using trial ID
    with mlflow.start_run(run_name=f"Trial_{trial.number}"):
        mlflow.log_params({
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'ssim_mse_weight': ssim_mse_weight,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'seed': seed
        })

        # Save the MLflow run ID in the trial's user attributes
        run_id = mlflow.active_run().info.run_id
        trial.set_user_attr("mlflow_run_id", run_id)

        # Best validation loss for this trial
        best_val_loss = float('inf')
        best_model_state = None  # To store the best model's state_dict

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_mse = 0.0
            running_ssim = 0.0
            running_psnr = 0.0
            
            with tqdm(train_loader, desc=f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
                for batch in tepoch:
                    distorted_images = batch['distorted'].to(device)
                    original_images = batch['original'].to(device)

                    optimizer.zero_grad()
                    outputs = model(distorted_images)

                    loss, mse_value, ssim_value = compute_loss(outputs, original_images, criterion_mse, criterion_ssim, ssim_mse_weight)
                    psnr_value = calculate_psnr(outputs, original_images)

                    loss.backward()
                    optimizer.step()

                    batch_size_actual = distorted_images.size(0)
                    running_loss += loss.item() * batch_size_actual
                    running_mse += mse_value * batch_size_actual
                    running_ssim += ssim_value * batch_size_actual
                    running_psnr += psnr_value * batch_size_actual

                    # Update tqdm description
                    tepoch.set_postfix(
                        loss=f"{loss.item():.3f}", mse=f"{mse_value:.3f}", ssim=f"{ssim_value:.3f}", psnr=f"{psnr_value:.3f}"
                    )

            # Average metrics for the epoch
            num_train_samples = len(train_loader.dataset)
            train_loss = running_loss / num_train_samples
            train_mse = running_mse / num_train_samples
            train_ssim = running_ssim / num_train_samples
            train_psnr = running_psnr / num_train_samples
            
            # Validation
            val_loss, val_mse, val_ssim, val_psnr = evaluate_model(
                model, val_loader, criterion_mse, criterion_ssim, ssim_mse_weight
            )

            # Log training and validation metrics to MLflow
            mlflow.log_metrics({
                'train_loss': round(train_loss, 4),
                'val_loss': round(val_loss, 4),
                'train_mse': round(train_mse, 4),
                'val_mse': round(val_mse, 4),
                'train_ssim': round(train_ssim, 4),
                'val_ssim': round(val_ssim, 4),
                'train_psnr': round(train_psnr, 2),
                'val_psnr': round(val_psnr, 2)
            }, step=epoch)

            # Save sample images every 2 epochs
            if epoch % 2 == 0:
                save_sample_images(model, sample_distorted_images, sample_original_images, epoch)

            # Report validation loss to Optuna and prune if needed
            trial.report(val_loss, epoch)
            if trial.should_prune():
                # Log a message to indicate pruning in the console
                print(f"Trial {trial.number} pruned at epoch {epoch} with val_loss {val_loss:.4f}")

                # Tag the MLflow run to indicate pruning
                mlflow.set_tag("status", "pruned")
                mlflow.log_metric("pruned_epoch", epoch)  # Log the epoch at which it was pruned
                
                # End the MLflow run explicitly with a "KILLED" status to mark it as incomplete
                mlflow.end_run(status="KILLED")
                
                # Raise the Optuna exception to stop the trial
                raise optuna.exceptions.TrialPruned()

            # Save best model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()  # Save the model's state_dict

        # After training completes, log the best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

            input_example = sample_distorted_images
                 
            # Perform inference to get output example for signature
            model.eval()
            with torch.no_grad():
                output_example = model(sample_distorted_images)

            # Infer signature
            signature = infer_signature(
                input_example.cpu().numpy(),
                output_example.cpu().numpy()
            )

            # Log the model
            mlflow.pytorch.log_model(
                model, 
                artifact_path="model",
                signature=signature
            )

        return best_val_loss  # Return the best validation loss after all epochs


def main():
    # Experiment name
    experiment_name = "Unet_Optimization"
    seed = 42
    set_seed(seed)
    
    # Device configuration
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
        
    # Define image transformations
    transform = transforms.Compose([transforms.ToTensor()])

    # Load and split the dataset with fixed splits
    full_dataset = LicensePlateDataset(image_dir='data', transform=transform)
    num_samples = int(len(full_dataset)/10)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size

    # Linear split for reproducibility
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, num_samples))

    # Create subset datasets for each split
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
   
    # Print split sizes for confirmation
    print(f"Total samples: {num_samples}, Training samples: {train_size}, Validation samples: {val_size}, Test samples: {test_size}")
    
    # Set the MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Define or resume the Optuna study
    storage_url = f"sqlite:///{os.path.abspath('optuna_study.db')}"  
    study = optuna.create_study(
        study_name=experiment_name,
        storage=storage_url,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )
    
    # Resume study from where it left off
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, train_size, val_size, test_size, seed),
        n_trials=3  # Adjust n_trials as needed
    )

    # Identify and log the best trial results
    best_trial = study.best_trial
    print(f"Best trial number: {best_trial.number}")
    print(f"Best validation loss: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
        
    # Get the MLflow run ID from the best trial
    best_run_id = best_trial.user_attrs.get("mlflow_run_id", None)
    if best_run_id is None:
        print("No MLflow run ID found for the best trial.")
    else:
        # Tag the best trial in MLflow
        with mlflow.start_run(run_id=best_run_id):
            mlflow.set_tag("best_model", "True")
            # Log final test metrics to MLflow under the best run

            # Load and test best model
            test_loader = DataLoader(test_dataset, batch_size=best_trial.params['batch_size'], shuffle=False, num_workers=4)
            best_model = mlflow.pytorch.load_model(f"runs:/{best_run_id}/model")
            best_model.to(device)
            best_model.eval()

            criterion_mse = nn.MSELoss()
            criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
            
            test_loss, test_mse, test_ssim, test_psnr = evaluate_model(
                best_model, test_loader, criterion_mse, criterion_ssim, best_trial.params['ssim_mse_weight']
            )

            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test MSE: {test_mse:.4f}")
            print(f"Test SSIM: {test_ssim:.4f}")
            print(f"Test PSNR: {test_psnr:.4f}")

            mlflow.log_metrics({
                'test_loss': test_loss,
                'test_mse': test_mse,
                'test_ssim': test_ssim,
                'test_psnr': test_psnr
            })

if __name__ == "__main__":
    main()

