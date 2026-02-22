from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch

def get_device(prefer_name: str | None = None):
    """
    Select a GPU intelligently:
    - Print all available CUDA devices
    - Skip incompatible devices
    - Prefer a specific GPU name if provided (e.g. "RTX")
    - Otherwise pick the best (highest capability)
    """
    if not torch.cuda.is_available():
        print("[Config] CUDA not available, using CPU")
        return torch.device("cpu")

    num_gpus = torch.cuda.device_count()
    print(f"[Config] Found {num_gpus} CUDA device(s):")

    compatible_devices = []

    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"  - GPU {i}: {name} (compute capability {cap[0]}.{cap[1]})")

        # PyTorch >= 2.2 typically supports >= sm_70
        if cap[0] >= 7:  # Skip old GPUs like Quadro P400 (6.1)
            compatible_devices.append((i, name, cap))

    if not compatible_devices:
        print("[Config] No compatible CUDA GPUs found, falling back to CPU")
        return torch.device("cpu")

    # If user prefers a specific GPU (e.g. "RTX", "3090", etc.)
    if prefer_name is not None:
        for idx, name, cap in compatible_devices:
            if prefer_name.lower() in name.lower():
                torch.cuda.set_device(idx)
                print(f"[Config] Using preferred GPU {idx}: {name}")
                return torch.device(f"cuda:{idx}")

    # Otherwise pick the best GPU (highest compute capability)
    best_gpu = max(compatible_devices, key=lambda x: x[2])
    best_idx, best_name, best_cap = best_gpu

    torch.cuda.set_device(best_idx)
    print(f"[Config] Using best compatible GPU {best_idx}: {best_name}")
    return torch.device(f"cuda:{best_idx}")

def is_colab() -> bool:
    # Most reliable: module only present in Colab
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

@dataclass(frozen=True)
class Paths:
    data_train: Path
    data_test: Path
    data: Path
    data_export: Path
    results: Path
    runs: Path
    configs_file : Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IN_COLAB = is_colab()

if IN_COLAB:
    PATHS = Paths(
        data_train=Path("/data/train"),
        data_test=Path("/data/test"),
        data=Path("/data"),
        data_export=Path("/data/fiftyone/jaguar_export"),
        results=Path("/results"),
        runs=Path("/experiments"),
        configs_file=Path("/configs")
    )
    
else:
    PATHS = Paths(
        data_train=PROJECT_ROOT / "data/raw/jaguar-re-id/train/train",
        data_test=PROJECT_ROOT / "data/raw/jaguar-re-id/test/test",
        data_export=PROJECT_ROOT / "data/fiftyone/jaguar_export",
        data=PROJECT_ROOT / "data",
        results=PROJECT_ROOT / "results",
        runs=PROJECT_ROOT / "experiments",
        configs_file=PROJECT_ROOT / "configs",
    )

DEVICE = get_device(prefer_name="RTX")  
NUM_WORKERS = 0
SEED = 51
NUM_WORKERS = 0
SEED = 51
