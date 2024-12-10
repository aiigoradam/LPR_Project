# train_unet_single.py
import os
import sys
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import mlflow
from mlflow.models.signature import infer_signature
from pytorch_msssim import ssim

# Add the parent directory of src (LPR_Project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet
from lp_dataset import LicensePlateDataset

from utils import set_seed, calculate_psnr, save_sample_images, evaluate_model

# =====================================
# Configuration Dictionary
# =====================================

config = {
    'experiment_name': 'Unet_Final',
    'run_name': 'run_01',
    'seed': 42,
    'num_epochs': 40,
    'train_size': 8192,
    'val_size': 2048,
    'batch_size': 4,  # Fixed batch size 
    'learning_rate': 0.001,  # Fixed learning rate  0.0001 to 0.01 (best 0.001, 0.0005!)
    'weight_decay': 5e-5,   # Fixed weight decay   0.00001 to 0.0001  (best 0.00001, 0.00005)
    'transform': transforms.Compose([transforms.ToTensor()]),
}

# =====================================
# Main Function
# =====================================

def main():
    set_seed(config['seed'])
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
        
    # Define image transformations
    transform = config['transform']

    # Load and split the dataset with fixed splits
    full_dataset = LicensePlateDataset(image_dir='data', transform=transform)
    train_size = config['train_size']
    val_size = config['val_size']

    # Linear split for reproducibility
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))

    # Create subset datasets for each split
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
   
    # Print split sizes for confirmation
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Set the MLflow experiment
    mlflow.set_experiment(config['experiment_name'])

    # DataLoaders
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model and loss function
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.MSELoss()
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Preload a sample batch from val_loader for logging and visualization
    sample_batch = next(iter(val_loader))
    sample_distorted_images = sample_batch['distorted'].to(device)
    sample_original_images = sample_batch['original'].to(device)
    
    # Start MLflow run
    with mlflow.start_run(run_name=config['run_name']):
        mlflow.log_params({
            'learning_rate': config['learning_rate'],
            'batch_size': batch_size,
            'weight_decay': config['weight_decay'],
            'train_size': config['train_size'],
            'val_size': config['val_size'],
            'seed': config['seed']
        })

        # Best validation loss for this run
        best_val_loss = float('inf')
        best_model_state = None  # To store the best model's state_dict

        num_epochs = config['num_epochs']

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_mse = 0.0
            running_ssim = 0.0
            running_psnr = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch: 
                for batch in tepoch:
                    distorted_images = batch['distorted'].to(device)
                    original_images = batch['original'].to(device)

                    optimizer.zero_grad()
                    outputs = model(distorted_images)

                    loss = criterion(outputs, original_images)
                    mse_value = loss.item()
                    ssim_value = ssim(outputs, original_images, data_range=1.0, size_average=True).item()
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
                        loss=f"{loss.item():.5f}", mse=f"{mse_value:.5f}", ssim=f"{ssim_value:.3f}", psnr=f"{psnr_value:.2f}"
                    )

            # Average metrics for the epoch
            num_train_samples = len(train_loader.dataset)
            train_loss = running_loss / num_train_samples
            train_mse = running_mse / num_train_samples
            train_ssim = running_ssim / num_train_samples
            train_psnr = running_psnr / num_train_samples
            
            # Validation
            val_loss, val_mse, val_ssim, val_psnr = evaluate_model(model, val_loader, criterion, device)
            
            # Step the scheduler
            scheduler.step(val_loss)
            
            # Log and print the current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # Log training and validation metrics to MLflow
            mlflow.log_metrics({
                'train_loss':   round(train_loss, 5),
                'train_mse':    round(train_mse, 5), 
                'train_ssim':   round(train_ssim, 4),
                'train_psnr':   round(train_psnr, 2),
                'val_loss':     round(val_loss, 4),
                'val_mse':      round(val_mse, 4),
                'val_ssim':     round(val_ssim, 4),
                'val_psnr':     round(val_psnr, 2),
                'current_learning_rate': current_lr
            }, step=epoch)
            
            print(
                f"          Epoch {epoch+1}/{num_epochs}:      Val Loss: {val_loss:.4f}, "
                f"Val MSE: {val_mse:.5f}, Val SSIM: {val_ssim:.4f}, Val PSNR: {val_psnr:.2f}, "
                f"Current lr: {current_lr:.6f}"
            )

            # Save sample images every 2 epochs
            if epoch % 2 == 0:
                save_sample_images(model, sample_distorted_images, sample_original_images, epoch, mlflow)

            # Save best model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
        
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

    print("Training completed.")

if __name__ == "__main__":
    main()
