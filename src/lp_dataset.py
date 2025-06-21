# src/license_plate_dataset.py

import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LicensePlateDataset(Dataset):
    def __init__(self, image_source, transform):
        # Validate inputs
        if not os.path.isdir(image_source):
            raise ValueError(f"No such directory: {image_source}")
        manifest = os.path.join(image_source, "metadata.json")
        if not os.path.isfile(manifest):
            raise ValueError(f"Missing metadata.json in {image_source}")

        # Load and sort metadata
        with open(manifest, "r") as f:
            records = json.load(f)
        self.metadata = sorted(records, key=lambda r: r["index"])
        N = len(self.metadata)

        # Peek at first image to get shape
        first_idx = self.metadata[0]["index"]
        sample = Image.open(os.path.join(image_source, f"original_{first_idx}.png")).convert("RGB")
        H, W = sample.size[1], sample.size[0]

        # Allocate numpy arrays (N × H × W × 3) for fast bulk load
        orig_np = np.empty((N, H, W, 3), dtype=np.uint8)
        dist_np = np.empty((N, H, W, 3), dtype=np.uint8)

        for i, rec in enumerate(self.metadata):
            idx = rec["index"]
            o = Image.open(os.path.join(image_source, f"original_{idx}.png")).convert("RGB")
            d = Image.open(os.path.join(image_source, f"distorted_{idx}.png")).convert("RGB")
            orig_np[i] = np.array(o, dtype=np.uint8)
            dist_np[i] = np.array(d, dtype=np.uint8)

        # Store arrays and transform
        self.orig_np = orig_np
        self.dist_np = dist_np
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Fetch raw uint8 H×W×3 arrays
        orig_arr = self.orig_np[idx]
        dist_arr = self.dist_np[idx]

        # Apply the user-provided transform pipeline (expects a PIL image or numpy array H×W×3)
        orig = self.transform(orig_arr)
        dist = self.transform(dist_arr)

        # Pull out metadata
        rec = self.metadata[idx]
        meta = {
            "plate_number": rec["plate_number"],
            "alpha": rec["alpha"],
            "beta": rec["beta"],
            "digit_bboxes": rec["digit_bboxes"],
        }

        return {"original": orig, "distorted": dist, "metadata": meta}
