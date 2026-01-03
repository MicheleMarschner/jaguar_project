import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class JaguarDataset(Dataset):
    def __init__(self, base_root, transform=None, processing_fn=None, samples_list=None):
        """
        base_dir: Path to the folder containing 'samples.json', 'data/', 'masks/', etc.
        transform: standard torchvision transforms.
        processing_fn: A function that takes (image, mask) and returns image.
        """
        self.base_root = Path(base_root)
        self.transform = transform
        self.processing_fn = processing_fn

        if samples_list:
            self.samples = samples_list
        else:
            # Only read from disk if we didn't pass a list
            with open(self.base_root / "samples.json", "r") as f:
                data = json.load(f)
                self.samples = data["samples"] if isinstance(data, dict) else data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 1. Load Crop and Mask using paths from the JSON
        # We join with export_root because paths in JSON are relative (e.g. "crops/1.png")
        img_path = self.base_root / s["filepath"]
        crop_path = self.base_root / s["crop_path"]
        mask_path = self.base_root / s["mask_path"]

        img = Image.open(crop_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 2. Expandable "Hook": Apply manipulation if provided
        if self.processing_fn:
            img = self.processing_fn(img, mask)

        # 3. Final Transform
        if self.transform:
            img = self.transform(img)

        # Return image and the 'rel_filepath' as the stable ID
        return {
            "img": img, 
            "id": s.get("id", ""),          
            "filepath": s["filepath"]   # !!TODO: check: Stable path for rebinding
        }