import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm  # For progress bars

import sys
import os

# Add the parent directory of src (LPR_Project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet  # Now this import should work

from license_plate_dataset import LicensePlateDataset  # Your dataset class

def main():
    # Device configuration (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    num_samples = 1000
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 16

    # Image transformations
    transform = transforms.Compose([transforms.ToTensor()])

    # Create datasets
    train_dataset = LicensePlateDataset(image_dir='data', split='train', num_samples=num_samples, transform=transform)  
    val_dataset = LicensePlateDataset(image_dir='data', split='val', num_samples=num_samples, transform=transform)
    test_dataset = LicensePlateDataset(image_dir='data', split='test', num_samples=num_samples, transform=transform)  

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)  

    # Initialize the model, loss function, and optimizer
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            distorted_images = batch['distorted'].to(device)
            original_images = batch['original'].to(device)

            optimizer.zero_grad()

            outputs = model(distorted_images)
            loss = criterion(outputs, original_images)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * distorted_images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Validate the model
        validate(model, val_loader, criterion, device)

        # Save the model checkpoint
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

    # After training is done, evaluate on the test set
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            distorted_images = batch['distorted'].to(device)
            original_images = batch['original'].to(device)

            outputs = model(distorted_images)
            loss = criterion(outputs, original_images)

            val_loss += loss.item() * distorted_images.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}")

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            distorted_images = batch['distorted'].to(device)
            original_images = batch['original'].to(device)

            outputs = model(distorted_images)
            loss = criterion(outputs, original_images)

            test_loss += loss.item() * distorted_images.size(0)

    test_loss /= len(test_loader.dataset)
    return test_loss

if __name__ == "__main__":
    main()
