# src/license_plate_dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image
import json

class LicensePlateDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.indices = self._get_indices()
    
    def _get_indices(self):
        # Get the list of indices based on existing files
        files = os.listdir(self.image_dir)
        indices = sorted(set(int(f.split('_')[1].split('.')[0]) for f in files if f.endswith('.png')))
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        # Construct file paths
        original_path = os.path.join(self.image_dir, f"original_{actual_idx}.png")
        distorted_path = os.path.join(self.image_dir, f"distorted_{actual_idx}.png")
        metadata_path = os.path.join(self.image_dir, f"metadata_{actual_idx}.json")

        # Load images
        original = Image.open(original_path).convert('RGB')
        distorted = Image.open(distorted_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            original = self.transform(original)
            distorted = self.transform(distorted)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {'original': original, 'distorted': distorted, 'metadata': metadata}
