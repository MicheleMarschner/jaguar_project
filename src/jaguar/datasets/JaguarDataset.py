import json
from pathlib import Path
from typing import Callable, Optional, Any, Dict, List, Union

from PIL import Image
from torch.utils.data import Dataset


class JaguarDataset(Dataset):
    """
    Reads a FiftyOne-style export manifest (samples.json) and loads images from base_root.

    - sample["filepath"] is expected to be a relative path like "data/train_0001.png"
      (portable export case) OR an absolute path (manifest-only case).
    - If your JSON includes an identity field (e.g., "ground_truth" or "id"), we index it.
    """

    def __init__(
        self,
        base_root: Union[str, Path],
        transform: Optional[Callable] = None,
        processing_fn: Optional[Callable[[Image.Image, Dict[str, Any]], Image.Image]] = None,
        is_test: bool =False,
        samples_list: Optional[List[Dict[str, Any]]] = None,
        label: str = "ground_truth",     # change if your samples store identity under a different key
        filepath_key: str = "filepath",
        filename_key: str = "filename",   # if you stored original filename explicitly in the sample
    ):
        self.base_root = Path(base_root)
        self.transform = transform
        self.processing_fn = processing_fn

        self.id_key = label
        self.filepath_key = filepath_key
        self.filename_key = filename_key
        self.is_test = is_test
        self.epoch = 0

        if samples_list is not None:
            self.samples = samples_list
        else:
            with open(self.base_root / "samples.json", "r") as f:
                data = json.load(f)
            self.samples = data["samples"] if isinstance(data, dict) and "samples" in data else data

        # Build fast lookup maps
        self._idx_by_id: Dict[str, List[int]] = {}
        for i, s in enumerate(self.samples):
            sid = s.get(self.id_key, None)
            if sid is None:
                continue
            sid = str(sid)
            self._idx_by_id.setdefault(sid, []).append(i)

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, p: str) -> Path:
        """
        Resolve sample filepath:
        - if absolute: use as-is
        - if relative: join with base_root
        """
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return self.base_root / pp
        
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = dict(self.samples[idx])   # copy
        s["_epoch"] = self.epoch

        img_path = self._resolve_path(s[self.filepath_key])
        img = Image.open(img_path).convert("RGBA")

        # Hook: apply custom processing BEFORE torchvision transforms
        # processing_fn should return a PIL.Image (or something your transform accepts)
        if self.processing_fn is not None:
            img = self.processing_fn(img, s)

        if self.transform is not None:
            img = self.transform(img)

        if self.is_test:
            return {
                "img": img, 
                "filepath": str(s.get(self.filepath_key, ""))
            }

        return {
            "img": img,
            "label": str(s.get(self.id_key, "")),
            "filepath": str(s.get(self.filepath_key, "")),
            "idx": idx,
        }