from datetime import datetime
import torch
import numpy as np

def ensure_dir(paths):
    """
    Ensures that the specified directories exist or creates it and any 
    missing parent directories.
    """
    for p in paths:
      p.mkdir(parents=True, exist_ok=True)


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