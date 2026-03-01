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
        
        # Remove alpha channel for later tranformation and modeling (models expect 3 channels)
        if img.mode == "RGBA":
            img = img.convert("RGB")

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