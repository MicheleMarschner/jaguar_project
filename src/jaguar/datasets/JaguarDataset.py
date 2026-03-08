import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import json
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Callable, Optional, Any, Dict, List, Union

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

from jaguar.config import PROJECT_ROOT, PATHS, DATA_ROOT

class JaguarDataset(Dataset):
    def __init__(
        self,
        base_root: Union[str, Path],
        data_root: Optional[Path] = None,
        transform: Optional[Callable] = None,
        processing_fn: Optional[Callable[[Image.Image, Dict[str, Any]], Image.Image]] = None,
        is_test: bool = False,
        mode: str = "train", # "train" or "val"
        split_parquet: Optional[Union[str, Path]] = None,
        include_duplicates: bool = True,
        filepath_key: str = "filepath",
        filename_key: str = "filename",
        samples_list: Optional[List[Dict[str, Any]]] = None,
    ):
        self.base_root = Path(base_root)
        self.data_root = Path(data_root).resolve() if data_root is not None else None
        self.transform = transform
        self.processing_fn = processing_fn
        self.is_test = is_test
        self.filepath_key = filepath_key
        self.filename_key = filename_key
        self.epoch = 0
        self.mode = mode
        
        if self.data_root is None:
            raise ValueError("data_root must be provided")
        
        if samples_list is not None:
            self.samples = samples_list

        elif split_parquet is not None:
            df = pd.read_parquet(split_parquet)

            if mode not in {"train", "val", "full"}:
                raise ValueError(f"Unsupported mode '{mode}'. Expected 'train' or 'val'.")

            if mode == "full":
                mask = pd.Series(True, index=df.index)
            else:
                mask = df["split_final"] == mode

            if mode != "full" and not include_duplicates:
                keep = df["keep_curated"]
                if keep.dtype != bool:
                    keep = keep.astype(str).str.lower().map({"true": True, "false": False})
                mask &= keep.fillna(False)

            df = df[mask].copy()

            self.samples = []
            for _, row in df.iterrows():
                self.samples.append({
                    self.filepath_key: row.get("filepath_rel", row["filename"]),
                    self.filename_key: row["filename"],
                    "ground_truth": {"label": row["identity_id"]},
                })

        elif split_parquet is None and not self.is_test:
            # Fallback to original JSON loading logic
            with open(self.base_root / "samples.json", "r") as f:
                data = json.load(f)

            raw_samples = data["samples"] if isinstance(data, dict) and "samples" in data else data

            self.samples = []
            for s in raw_samples:
                self.samples.append({
                    self.filepath_key: s.get("filename") or Path(s["filepath"]).name,
                    "ground_truth": {
                        "label": s["ground_truth"]["label"]
                    }
                })

        elif self.is_test:
            test_dir = self.data_root / "raw/jaguar-re-id/test/test"
            if not test_dir.exists():
                raise FileNotFoundError(f"Test directory not found: {test_dir}")
            self.samples = []
            # Collect all image files
            for img_path in sorted(test_dir.glob("*.png")):
                filename = img_path.name  # e.g. "000123.png"
                self.samples.append({
                    self.filepath_key: filename, # Only store "train_XXXX.png"
                    "ground_truth": {"label": ""}
                })
            print(f"[JaguarDataset] Loaded {len(self.samples)} test images.")
        # Setup Labels
        if self.is_test:
            self.labels = [""] * len(self.samples)
        else:
            self.labels = [str(s["ground_truth"]["label"]) for s in self.samples]
            unique_labels = sorted(list(set(self.labels)))
            self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
            self.labels_idx = [self.label_to_idx[l] for l in self.labels]

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _resolve_path(self, p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute() and pp.exists():
            return pp
        # If relative or absolute doesn't exist, use data_root (data/round_1/...)
        if self.data_root is not None and self.mode in ("train", "val"):
            return self.data_root / "raw/jaguar-re-id/train/train" / pp.name
        elif self.data_root is not None and self.mode=="test":
            return self.data_root / "raw/jaguar-re-id/test/test" / pp.name
        return pp
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = dict(self.samples[idx])
        s["_epoch"] = self.epoch
        s.setdefault("filename", s.get(self.filename_key) or Path(s[self.filepath_key]).name)

        img_path = self._resolve_path(s[self.filepath_key])
        img = Image.open(img_path).convert("RGBA") # do we need the alpha channel? 

        if self.processing_fn is not None:
            img = self.processing_fn(img, s)
        
        # Default mode uses background if no processing function is given 
        if img.mode == "RGBA":
            r, g, b, _a = img.split()
            img = Image.merge("RGB", (r, g, b))

        if self.transform is not None:
            img = self.transform(img)
        
        if not isinstance(img, torch.Tensor):
            # v2 approach: convert to image tensor and rescale to [0, 1]
            img = transforms.functional.to_image(img)
            img = transforms.functional.to_dtype(img, torch.float32, scale=True)

        out = {
            "img": img,
            "filepath": str(img_path),
            "filename": s["filename"],
            "idx": idx,
        }

        if not self.is_test:
            out["label"] = self.labels[idx]         # original string
            out["label_idx"] = self.labels_idx[idx] # numeric index

        return out
    


# ---------------------------------------------------------
# 1. SPECIALIZED DATASET (Inherits from your JaguarDataset)
# ---------------------------------------------------------
class MaskAwareJaguarDataset(JaguarDataset):
    def __init__(self, 
                 jaguar_model,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # 1. Access the inner wrapper configuration
        # We assume jaguar_model has 'backbone_wrapper'
        wrapper = jaguar_model.backbone_wrapper
        config = wrapper.MODEL_REGISTRY[wrapper.name] # Or wrapper.MODEL_REGISTRY[wrapper.model_name], wrapper.get_config()
        
        self.input_size = config.get("input_size", 256)
        self.mean = config.get("mean", [0.485, 0.456, 0.406])
        self.std = config.get("std", [0.229, 0.224, 0.225])
        
        # 2. Resize RGBA (Image + Mask together)
        self.resize_rgba = transforms.Resize(
            (self.input_size, self.input_size), 
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        
        # 3. Normalize RGB (After masking)
        self.to_tensor_norm = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.alpha_threshold = 128
        self.mask_fill_color = (128, 128, 128)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        img_path = self._resolve_path(s[self.filepath_key])
        
        # Load RGBA
        try:
            rgba = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None

        # split BEFORE resize to avoid RGBA premultiplication / blackening
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")

        rgb = rgb.resize((self.input_size, self.input_size), Image.BILINEAR)
        alpha = alpha.resize((self.input_size, self.input_size), Image.NEAREST)

        rgb_np = np.array(rgb)
        a_np = np.array(alpha)

        fg_mask = a_np > self.alpha_threshold

        img_orig = rgb_np.copy()

        img_bg_masked = rgb_np.copy()
        img_bg_masked[~fg_mask] = self.mask_fill_color   # keep jaguar

        img_fg_masked = rgb_np.copy()
        img_fg_masked[fg_mask] = self.mask_fill_color    # keep background
            
        
        # Normalize
        def process(arr):
            return self.to_tensor_norm(Image.fromarray(arr))

        return {
            "t_orig": process(img_orig),
            "t_bg_masked": process(img_bg_masked),
            "t_fg_masked": process(img_fg_masked),
            "label_idx": self.labels_idx[idx] if not self.is_test else -1,
            "filepath": str(img_path),
            "id": str(s.get("ground_truth", {}).get("label", "unknown"))
        }
        
if __name__ == "__main__":
    # Define your paths based on your folder structure
    base_path = Path(f"{PROJECT_ROOT}/experiments/round_1")
    data_path = DATA_ROOT
    parquet_path = base_path / "splits" / "jaguar_burst__str_closed_set__pol_drop_duplicates__k1" / "full_split.parquet"

    # Check if files exist before running
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
        exit(1)

    print(f"--- Dataset Test Start ---\n")

    # Test Training Split (Filtered: Curated Only)
    print("Test 1: Training mode (include_duplicates=False)")
    train_ds = JaguarDataset(
        base_root=base_path,
        data_root=data_path,
        mode="train",
        split_parquet=parquet_path,
        include_duplicates=False
    )
    print(f"Total curated train samples: {len(train_ds)}")
    print(f"Total unique identities: {len(train_ds.label_to_idx)}")
    
    if len(train_ds) > 0:
        sample = train_ds[0]
        print(f"Sample index 0 Filename: {sample['filename']}")
        print(f"Sample index 0 Label: {sample['label']} (Mapped ID: {sample['label_idx']})")
        print(f"Full Resolved Path: {sample['filepath']}")
        print(f"Image Tensor Shape: {sample['img'].shape}")
    print("-" * 40)

    # Test Validation Split
    print("Test 2: Validation mode")
    val_ds = JaguarDataset(
        base_root=base_path,
        data_root=data_path,
        mode="val",
        split_parquet=parquet_path
    )
    print(f"Total validation samples: {len(val_ds)}")
    if len(val_ds) > 0:
        print(f"Validation sample index 0: {val_ds[0]['filename']}")
    print("-" * 40)

    # Test Training Split (Unfiltered: Includes Duplicates)
    print("Test 3: Training mode (include_duplicates=True)")
    train_ds_all = JaguarDataset(
        base_root=base_path,
        data_root=data_path,
        mode="train",
        split_parquet=parquet_path,
        include_duplicates=True
    )
    print(f"Total train samples (curated + duplicates): {len(train_ds_all)}")
    
    # Quick math check to see if duplicates were actually filtered in Test 1
    diff = len(train_ds_all) - len(train_ds)
    print(f"Detected {diff} duplicate samples filtered out in Test 1.")

    print(f"\n--- Dataset Test Complete ---")


