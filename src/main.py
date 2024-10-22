import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm  # For progress bars
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

import sys
import os

# Add the parent directory of src (LPR_Project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import UNet  # Now this import should work

 # Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Assuming you already have the LicensePlateDataset class and trained model

def plot_single_test_output(model, original_path, distorted_path, device, transform=None):
    model.eval()

    # Load the original and distorted images
    original = Image.open(original_path).convert('RGB')
    distorted = Image.open(distorted_path).convert('RGB')

    transform = transforms.ToTensor()  # convert PIL image to Tensor
    distorted_tensor = transform(distorted).unsqueeze(0).to(device)
    
    # Get the model output (predicted image)
    with torch.no_grad():
        predicted_tensor = model(distorted_tensor).cpu().squeeze(0)  # Remove batch dimension

    # Ensure the predicted tensor is clamped between [0, 1]
    predicted_tensor = torch.clamp(predicted_tensor, 0, 1)

    # Convert tensors to PIL images for visualization
    transform_to_pil = transforms.ToPILImage()
    predicted_image_pil = transform_to_pil(predicted_tensor)

    # Plot the original, distorted, and predicted images
    plt.figure(figsize=(12, 4))
    # Distorted Image
    plt.subplot(3, 1, 1)
    plt.title('Distorted Image')
    plt.imshow(distorted)
    plt.axis('off')

    # Predicted Image (model output)
    plt.subplot(3, 1, 2)
    plt.title('Predicted Image')
    plt.imshow(predicted_image_pil)
    plt.axis('off')

    # Original Image (Ground truth)
    plt.subplot(3, 1, 3)
    plt.title('Original Image')
    plt.imshow(original)
    plt.axis('off')

    plt.show()

# Assuming you've already trained your UNet model and saved it
# Load the model (e.g., 'checkpoint_epoch_20.pth')
model = UNet(in_channels=3, out_channels=3).to(device)

# Load the state dictionary (model weights) safely
state_dict = torch.load('checkpoint_epoch_20.pth', map_location=device, weights_only=True)

# Load the weights into the model
model.load_state_dict(state_dict)

# Paths to the specific images
original_image_path = 'data/original_1.png'
distorted_image_path = 'data/distorted_1.png'

# Plot the output for the selected image pair
plot_single_test_output(model, original_image_path, distorted_image_path, device)
