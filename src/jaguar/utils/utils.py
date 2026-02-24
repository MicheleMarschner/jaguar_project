import os
from datetime import datetime
import torch
import numpy as np
from dataclasses import fields
from pathlib import Path
import wandb
import random

from typing import Sequence

from jaguar.config import IMGNET_MEAN, IMGNET_STD, PATHS, Paths, IN_COLAB


def ensure_dir(p: Path) -> None:
    """
    Ensures that the specified directories exist or creates it and any 
    missing parent directories.
    """
    p.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Paths = PATHS) -> None:
    """Create all directories in PATHS (dataclass fields that are Path)."""
    for f in fields(paths):
        val = getattr(paths, f.name)
        if isinstance(val, Path):
            ensure_dir(val)


def get_timestamp():
    """
    Return a human-readable timestamp string (YYYY-MM-DD_HH-MM) for naming 
    files or runs.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def set_seeds(seed: int=51, deterministic: bool=True) -> None:
    """Sets seeds for complete reproducibility across all libraries and operations"""

    # Python hashing (affects iteration order in some cases)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # CUDA deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if deterministic:
            # cuDNN determinism
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # CUDA matmul determinism (PyTorch recommends setting this env var)
            # Only needed for some CUDA versions/ops; harmless otherwise.
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    if deterministic:
        # Force deterministic algorithms when available
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(str(e))


    print(f"All random seeds set to {seed} for reproducibility")


def init_wandb(config):
    wandb.login(key=os.environ["WANDB_API_KEY"])

    if IN_COLAB:
        mode = "online"
    else: 
        mode = "offline"


    run = wandb.init(
        project="jaguar_project",
        #group=group,
        #name=name,
        mode=mode,                 
        config=config,
        reinit=True,               
        settings=wandb.Settings(start_method="thread"),
    )
    return run


def normalize_query_indices(query_indices: Sequence[int], dataset_len: int) -> np.ndarray:
    """
    Strict sequence-only API.
    - Single sample must be [i], NOT i.
    """
    if query_indices is None:
        raise ValueError("query_indices must be a list/array. For one sample use [i].")

    q = np.asarray(query_indices, dtype=np.int64).reshape(-1)

    if q.size == 0:
        raise ValueError("query_indices is empty. For one sample use [i].")

    if q.min() < 0 or q.max() >= dataset_len:
        raise IndexError(f"query_indices out of range [0, {dataset_len-1}].")

    return q


def denormalize_image(x, mean=IMGNET_MEAN, std=IMGNET_STD):
    """
    x: torch.Tensor [3,H,W] normalized
    returns: np.ndarray [H,W,3] in [0,1]
    """
    if isinstance(x, np.ndarray):
        # assume already CHW or HWC
        if x.ndim == 3 and x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))
        x = x.astype(np.float32)
        return np.clip(x, 0, 1)

    x = x.detach().cpu().float().clone()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    x = x * std_t + mean_t
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def tensor_img_to_hwc01(x):
    """
    Converts image tensor/array to HWC float [0,1] for matplotlib.imshow.
    Supports:
      - torch.Tensor [C,H,W] or [H,W,C]
      - np.ndarray   [C,H,W] or [H,W,C]
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[0] in (1, 3, 4):  # CHW -> HWC
        x = np.transpose(x, (1, 2, 0))

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    # If values are in [0,255], scale down
    if np.nanmax(x) > 1.0:
        x = x / 255.0

    return np.clip(x, 0.0, 1.0)


def json_default(obj):
    """Make numpy/path types JSON serializable."""
    try:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
