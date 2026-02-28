from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
import os

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
    return "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ
    
def is_kaggle() -> bool:
    if Path("/kaggle/input").exists():
        return True
    return False

def find_project_root(start: Path) -> Path:
    """
    Robust project root detection: walk upward until we find pyproject.toml or configs/.
    Falls back to parents[2] to keep your current behavior if markers are missing.
    """
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / "configs").exists():
            return p
    return start.parents[2]


@dataclass(frozen=True)
class Paths:
    data_train: Path
    data_test: Path
    data: Path
    data_export: Path
    results: Path
    runs: Path
    configs : Path
    checkpoints: Path


# --- Roots (stable scheme) ---
ROUND = os.environ.get("JAGUAR_ROUND", "round_1")
# Code root (auto)
PROJECT_ROOT = find_project_root(Path(__file__).parent)

# Data root (portable): set this in Kaggle + local
# Kaggle example:
#   os.environ["JAGUAR_DATA_ROOT"] = "/kaggle/input/datasets/mmarschn/jaguar-data/jaguar_data"
DATA_ROOT_ENV = os.environ.get("JAGUAR_DATA_ROOT")
if DATA_ROOT_ENV is None:
    # Local fallback: keep your old layout as a default *only* if env var not set
    DATA_ROOT = PROJECT_ROOT / f"data/{ROUND}"
else:
    DATA_ROOT = Path(DATA_ROOT_ENV)

# Work root (where outputs go). On Kaggle, always writable.
WORK_ROOT_ENV = os.environ.get("JAGUAR_WORK_ROOT")
if WORK_ROOT_ENV is not None:
    WORK_ROOT = Path(WORK_ROOT_ENV)
else:
    WORK_ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else PROJECT_ROOT

# Optional: separate persistent checkpoints input (read-only). If not set, just write to WORK_ROOT.
CHECKPOINTS_ROOT_ENV = os.environ.get("JAGUAR_CHECKPOINTS_ROOT")
CHECKPOINTS_ROOT = Path(CHECKPOINTS_ROOT_ENV) if CHECKPOINTS_ROOT_ENV else (WORK_ROOT / "checkpoints")


# --- Derive your existing PATHS exactly as your code expects ---
PATHS = Paths(
    data_train=DATA_ROOT / "raw/jaguar-re-id/train/train",
    data_test=DATA_ROOT / "raw/jaguar-re-id/test/test",
    data=DATA_ROOT / "raw",
    data_export=DATA_ROOT / "fiftyone",

    # outputs always go to WORK_ROOT (Kaggle-safe)
    results=WORK_ROOT / "results" / ROUND,
    runs=WORK_ROOT / "experiments" / ROUND,
    checkpoints=CHECKPOINTS_ROOT / ROUND,

    # configs live with code
    configs=PROJECT_ROOT / "configs",
)

DEVICE = get_device(prefer_name="RTX")  
NUM_WORKERS = 0
SEED = 51
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]
