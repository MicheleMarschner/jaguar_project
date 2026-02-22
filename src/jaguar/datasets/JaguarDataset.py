import torch
import json
from pathlib import Path
from typing import Callable, Optional, Any, Dict, List, Union

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

class JaguarDataset(Dataset):
    def __init__(
        self,
        base_root: Union[str, Path],
        transform: Optional[Callable] = None,
        processing_fn: Optional[Callable[[Image.Image, Dict[str, Any]], Image.Image]] = None,
        is_test: bool = False,
        samples_list: Optional[List[Dict[str, Any]]] = None,
        filepath_key: str = "filepath",
        filename_key: str = "filename",
    ):
        self.base_root = Path(base_root)
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
        if not self.is_test:
            for i, sid in enumerate(self.labels):
                self.idx_by_id.setdefault(sid, []).append(i)

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _resolve_path(self, p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (self.base_root / pp)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = dict(self.samples[idx])
        s["_epoch"] = self.epoch
        s.setdefault("filename", s.get(self.filename_key) or Path(s[self.filepath_key]).name)

        img_path = self._resolve_path(s[self.filepath_key])
        img = Image.open(img_path).convert("RGBA") #.convert("RGBA") do we need the alpha channel?

        if self.processing_fn is not None:
            img = self.processing_fn(img, s)

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
            out["label"] = self.labels[idx]   

        return out