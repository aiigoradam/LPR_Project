import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add the parent directory of src (LPR_Project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet
from license_plate_dataset import LicensePlateDataset

# Add the set_seed function
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_loss(outputs, targets, mse, ssim, p):
    mse_loss = mse(outputs, targets)
    ssim_loss = 1 - ssim(outputs, targets)
    
    # Combine MSE and SSIM losses
    loss = p * mse_loss + (1 - p) * ssim_loss
    return loss, mse_loss.item(), ssim_loss.item()

def evaluate_model(model, dataloader, mse_loss, ssim_loss, device, alpha, phase):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            distorted_images = batch['distorted'].to(device)
            original_images = batch['original'].to(device)

            outputs = model(distorted_images)

            # Compute combined loss
            loss = compute_loss(outputs, original_images, mse_loss, ssim_loss, alpha)

            epoch_loss += loss.item() * distorted_images.size(0)

    epoch_loss /= len(dataloader.dataset)
    print(f"{phase} Loss: {epoch_loss:.4f}")
    return epoch_loss

def plot_losses(train_losses, val_losses, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

def main():
    # Device configuration
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Set seed for reproducibility
    set_seed(42)

    # Hyperparameters
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 16
    ssim_mse_weight = 0.5  # Weight for balancing MSE and SSIM (0 for only SSIM, 1 for only MSE)
    patience = 5  # Early stopping patience

    # Image transformations
    transform = transforms.Compose([transforms.ToTensor()]) # No normalization needed for SSIM

    # Load the full dataset
    full_dataset = LicensePlateDataset(image_dir='data', transform=transform)

    # Determine the total number of samples and calculate split sizes
    num_samples = len(full_dataset)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)

    # Split indices linearly
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, num_samples))

    # Create subset datasets for each split
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Print split sizes for confirmation
    print(f"Total samples: {num_samples}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")

    # Initialize the model
    model = UNet(in_channels=3, out_channels=3).to(device)
    
    # Initialize both SSIM and MSE losses
    criterion_mse = nn.MSELoss()
    criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    
    # Initialize the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Initialize early stopping parameters
    best_val_loss = float('inf')  # Best validation loss encountered
    epochs_no_improve = 0  # Number of epochs with no improvement
    early_stop = False  # Early stopping flag
    
    # Lists to store losses
    train_losses = []
    val_losses = []
    
    # Ensure the models directory exists
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            distorted_images = batch['distorted'].to(device)
            original_images = batch['original'].to(device)

            optimizer.zero_grad()

            generated_images = model(distorted_images)
           
            # Compute combined loss
            loss = compute_loss(generated_images, original_images, criterion_mse, criterion_ssim, ssim_mse_weight)
            
            loss.backward() 
            optimizer.step()

            running_loss += loss.item() * distorted_images.size(0) 

        # Calculate average training loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Validate the model
        val_loss = evaluate_model(model, val_loader, criterion_mse, criterion_ssim, device, ssim_mse_weight, phase='Validation')
        val_losses.append(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the model checkpoint (best model so far) 
            checkpoint_path = os.path.join(models_dir, 'best_model_checkpoint.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation loss improved. Saving best model at epoch {epoch+1}.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        # Check for early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            early_stop = True

     # Plot training and validation losses
    plot_losses(train_losses, val_losses)

    # Load the best model before testing
    model.load_state_dict(torch.load(checkpoint_path), weights_only=True)

    # After training is done, evaluate on the test set
    test_loss = evaluate_model(model, test_loader, criterion_mse, criterion_ssim, device, ssim_mse_weight, phase='Test')
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
