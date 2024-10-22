import json
import os
from torch.utils.data import Dataset
from PIL import Image

class LicensePlateDataset(Dataset):
    def __init__(self, image_dir, split='train', num_samples=3, transform=None, train_pct=0.8, val_pct=0.1, test_pct=0.1):
        """
        Custom dataset for loading original and distorted license plate images by predefined filenames and indices.

        Args:
            image_dir (str): Path to the directory containing the images and metadata.
            split (str): 'train', 'val', or 'test'. Determines which split to use.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_samples (int): Total number of samples (image pairs) available.
            train_pct (float): Percentage of data to be used for training.
            val_pct (float): Percentage of data to be used for validation.
            test_pct (float): Percentage of data to be used for testing.
        """
        self.image_dir = image_dir
        self.transform = transform  # Store the transform

        # Calculate split indices
        train_end = int(train_pct * num_samples)
        val_end = train_end + int(val_pct * num_samples)

        # Assign indices based on the split
        match split:
            case 'train':
                self.indices = range(0, train_end)
            case 'val':
                self.indices = range(train_end, val_end)
            case 'test':
                self.indices = range(val_end, num_samples)
            case _:
                raise ValueError("split must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset based on the index."""
        actual_idx = self.indices[idx]

        # Construct the file names based on the index
        original_path = os.path.join(self.image_dir, f"original_{actual_idx}.png")
        distorted_path = os.path.join(self.image_dir, f"distorted_{actual_idx}.png")
        metadata_path = os.path.join(self.image_dir, f"metadata_{actual_idx}.json")

        # Load the images
        original = Image.open(original_path).convert('RGB')
        distorted = Image.open(distorted_path).convert('RGB')
        
        # Apply the transform
        original = self.transform(original)
        distorted = self.transform(distorted)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return {'original': original, 'distorted': distorted, 'metadata': metadata}
