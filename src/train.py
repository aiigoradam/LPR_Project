import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import SSIM
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm  
import sys
import os
import matplotlib.pyplot as plt

# Add the parent directory of src (LPR_Project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet
from license_plate_dataset import LicensePlateDataset

def compute_loss(outputs, targets, mse_loss, ssim_loss, alpha):
    mse = mse_loss(outputs, targets)
    ssim = ssim_loss(outputs, targets)
    
    # Combine MSE and SSIM losses
    loss = alpha * mse + (1 - alpha) * (1 - ssim)
    return loss

def evaluate_model(model, dataloader, mse_loss, ssim_loss, device, alpha, phase):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{phase}", leave=False):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    num_samples = 5000
    num_epochs = 50
    learning_rate = 0.001
    batch_size = 16
    alpha = 0  # Weight for balancing MSE and SSIM (0 for only SSIM, 1 for only MSE)
    patience = 5  # Early stopping patience

    # Image transformations
    transform = transforms.Compose([transforms.ToTensor()])

    # Create datasets
    train_dataset = LicensePlateDataset(image_dir='data', split='train', num_samples=num_samples, transform=transform)  
    val_dataset = LicensePlateDataset(image_dir='data', split='val', num_samples=num_samples, transform=transform)
    test_dataset = LicensePlateDataset(image_dir='data', split='test', num_samples=num_samples, transform=transform)  

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)  

    # Initialize the model
    model = UNet(in_channels=3, out_channels=3).to(device)
    
    # Initialize both SSIM and MSE losses
    mse_loss = nn.MSELoss()
    ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)
    
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

            outputs = model(distorted_images)
            
            # Compute combined loss
            loss = compute_loss(outputs, original_images, mse_loss, ssim_loss, alpha)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * distorted_images.size(0)

        # Calculate average training loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Validate the model
        val_loss = evaluate_model(model, val_loader, mse_loss, ssim_loss, device, alpha, phase='Validation')
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
    test_loss = evaluate_model(model, test_loader, mse_loss, ssim_loss, device, alpha, phase='Test')
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
