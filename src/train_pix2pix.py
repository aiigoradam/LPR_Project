import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys
import random
from pytorch_msssim import SSIM  # Import for SSIM

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.pix2pix import UNetGenerator, PatchGANDiscriminator
from lp_dataset import LicensePlateDataset

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def compute_ssim_loss(fake_images, real_images, criterion_ssim):
    # Convert images from [-1, 1] to [0, 1] range for SSIM calculation
    fake_images_ssim = (fake_images + 1) / 2
    real_images_ssim = (real_images + 1) / 2
    
    # Compute SSIM loss (1 - SSIM to make it a minimization objective)
    loss_ssim = 1 - criterion_ssim(fake_images_ssim, real_images_ssim)
    
    return loss_ssim       

def train_pix2pix():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Set seed for reproducibility
    set_seed(42)

    # Hyperparameters
    num_samples = 5000
    num_epochs = 100
    learning_rate = 0.0002
    batch_size = 16
    lambda_ssim = 0.5  # Weight for SSIM in the generator loss (1 - lambda_ssim is weight for L1 loss)
    lambda_L1 = 100  # Weight for L1 loss
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers

    # Image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
    ])

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

    # Instantiate models
    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    discriminator = PatchGANDiscriminator(in_channels=3).to(device)

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3)  

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    
    # Scheduler for learning rate decay
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=5, threshold=0.01, min_lr=1e-6)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=5, threshold=0.01, min_lr=1e-6)

    # Directories
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    output_samples_dir = 'output_samples'
    os.makedirs(output_samples_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch"):
            distorted_images = batch['distorted'].to(device)
            real_images = batch['original'].to(device)
            
            # Generator forward pass: Generate fake images
            fake_images = generator(distorted_images)

            # Discriminator's prediction for fake images
            pred_fake = discriminator(fake_images, distorted_images)  

            # Adversarial ground truths 
            valid = torch.ones_like(pred_fake).to(device)            
            fake = torch.zeros_like(pred_fake).to(device)

            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            
            # Adversarial loss
            loss_GAN = criterion_GAN(pred_fake, valid)

            # Combined SSIM and L1 loss
            loss_L1 = criterion_L1(fake_images, real_images)
            loss_ssim = compute_ssim_loss(fake_images, real_images, criterion_ssim)
            total_L1_SSIM = lambda_ssim * loss_L1 + (1 - lambda_ssim) * loss_ssim

            # Total generator loss
            loss_G = loss_GAN + lambda_L1 * total_L1_SSIM

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(real_images, distorted_images)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_images.detach(), distorted_images)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total discriminator loss
            loss_D = (loss_real + loss_fake) * 0.5

            loss_D.backward()
            optimizer_D.step()

        # Adjust learning rates
        scheduler_G.step(loss_G) 
        scheduler_D.step(loss_D)
            
        print(f"Epoch [{epoch}/{num_epochs}]  Loss_D: {loss_D.item():.4f}  Loss_G: {loss_G.item():.4f}  Loss_L1_SSIM: {total_L1_SSIM.item():.4f}")
        
        # Print the current learning rates
        print(f"Generator LR: {scheduler_G.optimizer.param_groups[0]['lr']} Discriminator LR: {scheduler_D.optimizer.param_groups[0]['lr']}")

        # Reset learning rate every number of epochs
        if epoch % 25 == 0:
            scheduler_G.optimizer.param_groups[0]['lr'] = learning_rate
            scheduler_D.optimizer.param_groups[0]['lr'] = learning_rate
            print(f"Learning rates reset to {learning_rate} at epoch {epoch}")

        # Save sample images
        if epoch % 2 == 0:
            save_sample_images(generator, val_loader, device, epoch, output_dir=output_samples_dir)

        # Save model checkpoints
        if epoch % 20 == 0:
            torch.save(generator.state_dict(), os.path.join(models_dir, f'generator_epoch_{epoch}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(models_dir, f'discriminator_epoch_{epoch}.pth'))

    # After training, evaluate on the test set
    evaluate(generator, test_loader, device)

def save_sample_images(generator, dataloader, device, epoch, output_dir='output_samples'):
    generator.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        distorted_images = batch['distorted'].to(device)
        real_images = batch['original'].to(device)
        fake_images = generator(distorted_images)
        # Concatenate images: distorted | real | fake
        images = torch.cat((distorted_images, real_images, fake_images), dim=0)
        images = (images + 1) / 2  # Denormalize to [0,1]
        save_image(images, os.path.join(output_dir, f'epoch_{epoch}.png'), nrow=distorted_images.size(0))
    generator.train()

def evaluate(generator, dataloader, device):
    generator.eval()
    total_L1_loss = 0.0
    criterion_L1 = nn.L1Loss()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            distorted_images = batch['distorted'].to(device)
            real_images = batch['original'].to(device)
            fake_images = generator(distorted_images)
            loss_L1 = criterion_L1(fake_images, real_images)
            total_L1_loss += loss_L1.item() * distorted_images.size(0)
    average_L1_loss = total_L1_loss / len(dataloader.dataset)
    print(f"Average L1 Loss on Test Set: {average_L1_loss:.4f}")

if __name__ == "__main__":
    train_pix2pix()
