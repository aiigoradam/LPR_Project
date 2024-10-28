import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import random
from pytorch_msssim import SSIM, ssim  # Import for SSIM

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pix2pix import UNetGenerator, PatchGANDiscriminator
from license_plate_dataset import LicensePlateDataset

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_generator_loss(generator, discriminator, distorted_images, real_images, criterion_GAN, criterion_ssim, lambda_ssim, device):
    fake_images = generator(distorted_images)  # Generate fake images
    # Discriminator's output on fake images
    pred_fake = discriminator(distorted_images, fake_images)  # Classifies fake as "real" or "fake"
    
    # Adversarial loss (MSE loss): Generator wants pred_fake to be "real" (ones)
    valid = torch.ones_like(pred_fake).to(device)  
    loss_GAN = criterion_GAN(pred_fake, valid)  # Adversarial (GAN) loss
    
    # SSIM loss between generated and real images
    loss_ssim = 1 - criterion_ssim(fake_images, real_images)  # SSIM: higher is better, so we invert
    
    # Total generator loss
    loss_G = loss_GAN + lambda_ssim * loss_ssim
    return loss_G, loss_GAN.item(), loss_ssim.item(), fake_images

def compute_discriminator_loss(discriminator, distorted_images, real_images, fake_images, criterion_GAN, device):
    # Real images
    pred_real = discriminator(distorted_images, real_images)  # Classify real pair
    valid = torch.ones_like(pred_real).to(device)
    loss_real = criterion_GAN(pred_real, valid)  # Discriminator should classify real pairs as "real"
    
    # Fake images
    pred_fake = discriminator(distorted_images, fake_images.detach())  # Classify fake pair
    fake = torch.zeros_like(pred_fake).to(device)
    loss_fake = criterion_GAN(pred_fake, fake)  # Discriminator should classify fake pairs as "fake"
    
    # Total discriminator loss
    loss_D = (loss_real + loss_fake) * 0.5
    return loss_D, loss_real.item(), loss_fake.item()

def evaluate_model(generator, dataloader, criterion_ssim, device, phase):
    generator.eval()
    epoch_loss = 0.0
    epoch_ssim = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{phase} Evaluation", leave=False):
            distorted_images = batch['distorted'].to(device)
            real_images = batch['original'].to(device)
            fake_images = generator(distorted_images)
            # Compute SSIM loss
            loss_ssim = 1 - criterion_ssim(fake_images, real_images)  # SSIM loss for structural similarity
            epoch_loss += loss_ssim.item() * distorted_images.size(0)
            # Compute SSIM metric
            batch_ssim = ssim(fake_images, real_images, data_range=1.0, size_average=True)
            epoch_ssim += batch_ssim.item() * distorted_images.size(0)
    epoch_loss /= len(dataloader.dataset)
    epoch_ssim /= len(dataloader.dataset)
    print(f"{phase} SSIM Loss: {epoch_loss:.4f}, SSIM Metric: {epoch_ssim:.4f}")
    generator.train()
    return epoch_loss, epoch_ssim

def plot_losses(train_losses, val_losses, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Metric')
    plt.title('Training Loss and Validation Metric over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Metric')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

def save_sample_images(generator, dataloader, device, epoch, output_dir='output_samples'):
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        distorted_images = batch['distorted'].to(device)
        real_images = batch['original'].to(device)
        fake_images = generator(distorted_images)
        # Denormalize images if necessary
        sample_images = torch.cat((distorted_images, real_images, fake_images), dim=0)
        save_image(sample_images.cpu(), os.path.join(output_dir, f'epoch_{epoch}.png'), nrow=distorted_images.size(0), normalize=True)
    generator.train()

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set seed for reproducibility
    set_seed(42)

    # Hyperparameters
    num_samples = 5000
    num_epochs = 100
    learning_rate_G = 0.0003
    learning_rate_D = 0.0001
    batch_size = 16
    lambda_ssim = 25  # Weight for SSIM in the generator loss
    patience = 10
    min_epochs = 20

    # Image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Create datasets
    train_dataset = LicensePlateDataset(image_dir='data', split='train', num_samples=num_samples, transform=transform)
    val_dataset = LicensePlateDataset(image_dir='data', split='val', num_samples=num_samples, transform=transform)
    test_dataset = LicensePlateDataset(image_dir='data', split='test', num_samples=num_samples, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # Instantiate models
    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    discriminator = PatchGANDiscriminator(in_channels=3).to(device)

    # Loss functions
    criterion_GAN = nn.MSELoss()  # MSE for adversarial loss
    criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3)  # SSIM for structural similarity

    # Optimizers with Adam
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))

    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)

    # Initialize early stopping parameters
    best_val_metric = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Lists to store losses and metrics
    train_losses = []
    val_metrics = []

    # Ensure directories exist
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    output_samples_dir = 'output_samples'
    os.makedirs(output_samples_dir, exist_ok=True)
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        generator.train()
        discriminator.train()
        running_loss_G = 0.0
        running_loss_D = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            distorted_images = batch['distorted'].to(device)
            real_images = batch['original'].to(device)

            # Train Generator
            optimizer_G.zero_grad()
            loss_G, loss_GAN_item, loss_ssim_item, fake_images = compute_generator_loss(
                generator, discriminator, distorted_images, real_images,
                criterion_GAN, criterion_ssim, lambda_ssim, device)
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            loss_D, loss_real_item, loss_fake_item = compute_discriminator_loss(
                discriminator, distorted_images, real_images, fake_images,
                criterion_GAN, device)
            loss_D.backward()
            optimizer_D.step()

            running_loss_G += loss_G.item() * distorted_images.size(0)
            running_loss_D += loss_D.item() * distorted_images.size(0)

        epoch_loss_G = running_loss_G / len(train_loader.dataset)
        epoch_loss_D = running_loss_D / len(train_loader.dataset)
        scheduler_G.step()
        scheduler_D.step()
        train_losses.append(epoch_loss_G)
        print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {epoch_loss_G:.4f}, Discriminator Loss: {epoch_loss_D:.4f}")

        # Validate the model
        val_loss, val_ssim = evaluate_model(generator, val_loader, criterion_ssim, device, phase='Validation')
        combined_metric = val_loss - val_ssim  # Lower is better, since SSIM loss is 1 - SSIM
        val_metrics.append(combined_metric)

        # Early stopping logic
        if combined_metric < best_val_metric:
            best_val_metric = combined_metric
            epochs_no_improve = 0
            checkpoint_path = os.path.join(models_dir, 'best_generator_checkpoint.pth')
            torch.save(generator.state_dict(), checkpoint_path)
            print(f"Validation metric improved. Saving best generator model at epoch {epoch+1}.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation metric for {epochs_no_improve} epochs.")

        if epoch + 1 >= min_epochs and epochs_no_improve >= patience:
            print("Early stopping triggered.")
            early_stop = True

        save_sample_images(generator, val_loader, device, epoch+1, output_dir=output_samples_dir)

    # Plot training loss and validation metric
    plot_losses(train_losses, val_metrics, output_dir=plots_dir)

    # Load best generator before testing
    generator.load_state_dict(torch.load(checkpoint_path), only_weights=True)
    test_loss, test_ssim = evaluate_model(generator, test_loader, criterion_ssim, device, phase='Test')
    print(f"Test SSIM Loss: {test_loss:.4f}, SSIM Metric: {test_ssim:.4f}")

    # Save final sample images
    save_sample_images(generator, test_loader, device, 'test', output_dir=output_samples_dir)

if __name__ == "__main__":
    main()