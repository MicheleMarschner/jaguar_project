import torch
import json
from pathlib import Path
import numpy as np
from typing import Callable, Optional, Any, Dict, List, Union

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

class JaguarDataset(Dataset):
    def __init__(
        self,
        base_root: Union[str, Path],
        data_root: Path | None = None,
        transform: Optional[Callable] = None,
        processing_fn: Optional[Callable[[Image.Image, Dict[str, Any]], Image.Image]] = None,
        is_test: bool = False,
        samples_list: Optional[List[Dict[str, Any]]] = None,
        filepath_key: str = "filepath",
        filename_key: str = "filename",
    ):
        self.base_root = Path(base_root)
        self.data_root = Path(data_root).resolve() if data_root is not None else None
        self.transform = transform
        self.processing_fn = processing_fn
        self.is_test = is_test
        self.filepath_key = filepath_key
        self.filename_key = filename_key
        self.epoch = 0

        if samples_list is not None:
            self.samples = samples_list
        else:
            with open(self.base_root / "samples.json", "r") as f:
                data = json.load(f)
            self.samples = data["samples"] if isinstance(data, dict) and "samples" in data else data

        if self.is_test:
            self.labels = [""] * len(self.samples)
        else:
            self.labels = [str(s["ground_truth"]["label"]) for s in self.samples]

        self.idx_by_id: Dict[str, List[int]] = {}
        # if not self.is_test:
        #     for i, sid in enumerate(self.labels):
        #         self.idx_by_id.setdefault(sid, []).append(i)
        if not self.is_test:
            # Map label string → index
            unique_labels = sorted(list(set([str(s["ground_truth"]["label"]) for s in self.samples])))
            self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
            # Also store numeric labels
            self.labels_idx = [self.label_to_idx[str(s["ground_truth"]["label"])] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _resolve_path(self, p: str) -> Path:
        pp = Path(p)

        # if absolute but doesn't exist on this machine: try rebasing to data_root by filename
        if pp.is_absolute():
            if pp.exists():
                return pp
            if self.data_root is not None:
                # last-resort: filename under train/test folders
                return self.data_root / "raw/jaguar-re-id/train/train" / pp.name
            return pp

        # relative: prefer data_root
        if self.data_root is not None:
            return self.data_root / pp

        return self.base_root / pp


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = dict(self.samples[idx])
        s["_epoch"] = self.epoch
        s.setdefault("filename", s.get(self.filename_key) or Path(s[self.filepath_key]).name)

        img_path = self._resolve_path(s[self.filepath_key])
        img = Image.open(img_path).convert("RGBA") #.convert("RGBA") do we need the alpha channel?

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
            return None # Skip or handle error
        
        # Resize
        rgba = self.resize_rgba(rgba)

        rgba_np = np.array(rgba)          # (H,W,4)
        rgb0 = rgba_np[..., :3]
        a_np = rgba_np[..., 3]

        bg0 = rgb0[a_np == 0]
        print("[DBG] file:", Path(img_path).name,
            "| alpha0%:", float((a_np == 0).mean()),
            "| bg_rgb_max:", bg0.max(axis=0) if bg0.size else None,
            "| bg_rgb_mean:", bg0.mean(axis=0) if bg0.size else None)

        # Now create rgb_np (equivalent RGB image)
        r, g, b, a = rgba.split()
        rgb_np = np.array(Image.merge("RGB", (r, g, b)))

        bg2 = rgb_np[a_np == 0]
        print("[DBG] after merge | bg_rgb_max:", bg2.max(axis=0) if bg2.size else None)

        # Split Mask
        r, g, b, a = rgba.split()
        fg_mask = np.array(a) > self.alpha_threshold
        
        # Create Variants
        rgb_np = np.array(Image.merge("RGB", (r, g, b)))
        
        img_orig = rgb_np.copy()
        
        img_bg_masked = rgb_np.copy()
        img_bg_masked[~fg_mask] = self.mask_fill_color # Keep Jaguar
        
        img_fg_masked = rgb_np.copy()
        img_fg_masked[fg_mask] = self.mask_fill_color # Keep Background
        
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


