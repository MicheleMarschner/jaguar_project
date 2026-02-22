import os
from datetime import datetime
import torch
import numpy as np
from dataclasses import fields
from pathlib import Path
import wandb
import random

from jaguar.config import PATHS, Paths, IN_COLAB

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