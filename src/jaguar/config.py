# src/jaguar/config.py
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
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            print("[Config] CUDA not available, using MPS")
            return torch.device("mps")
        else: 
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
    configs: Path
    checkpoints: Path


@dataclass(frozen=True)
class ArtifactStore:
    """
    Cache pattern:
      - read_roots: optional persisted caches (read-only is fine)
      - write_root: always writable (current run)
    """
    read_roots: tuple[Path, ...]
    write_root: Path


# NOTE: Project root is where your repo (configs/, src/, pyproject.toml, …) lives.
# We detect it robustly so it behaves the same even if nesting changes.
PROJECT_ROOT = find_project_root(Path(__file__).parent)

IN_COLAB = is_colab()
IN_KAGGLE = is_kaggle()
ROUND = "round_2"

DATA_ROOT = Path(
    os.environ.get("JAGUAR_DATA_ROOT", str(PROJECT_ROOT / f"data/{ROUND}"))
).resolve()


if IN_KAGGLE and "JAGUAR_WORK_ROOT" not in os.environ:
    WORK_ROOT = Path("/kaggle/working").resolve()
else:
    WORK_ROOT = Path(os.environ.get("JAGUAR_WORK_ROOT", str(PROJECT_ROOT))).resolve()


PATHS = Paths(
    data_train=DATA_ROOT / "raw/jaguar-re-id/train/train",
    data_test=DATA_ROOT / "raw/jaguar-re-id/test/test",
    data=DATA_ROOT / "raw",
    data_export=DATA_ROOT / "fiftyone",
    results=WORK_ROOT / "results" / ROUND,
    runs=WORK_ROOT / "experiments" / ROUND,
    configs=PROJECT_ROOT / "configs",
    checkpoints=WORK_ROOT / "checkpoints" / ROUND,
)

# NOTE: caching pattern (read-if-exists else compute+write)
# - Local: read_roots empty, write_root is under WORK_ROOT (often same as PROJECT_ROOT)
# - Kaggle: optional read cache dataset mounted at /kaggle/input/jaguar-artifacts, write_root under /kaggle/working
_experiments_cache = Path("/kaggle/input/datasets/mmarschn/jaguar-code/experiments/round_1")   
_results_cache     = Path("/kaggle/input/datasets/mmarschn/jaguar-code/results/round_1") 
_data_cache = Path("/kaggle/input/datasets/mmarschn/jaguar-data/jaguar_data")   

_experiments_read_roots = []
_results_read_roots = []
_data_read_roots = []

if IN_KAGGLE:
    if _experiments_cache.exists():
        _experiments_read_roots.append(_experiments_cache)
    if _results_cache.exists():
        _results_read_roots.append(_results_cache)
    if _data_cache.exists():
        _data_read_roots.append(_data_cache)


_experiments_read_roots.append(PATHS.runs)
_results_read_roots.append(PATHS.results)
_data_read_roots.append(PATHS.runs / "data")

EXPERIMENTS_STORE = ArtifactStore(
    read_roots=tuple(_experiments_read_roots),
    write_root=PATHS.runs,
)

RESULTS_STORE = ArtifactStore(
    read_roots=tuple(_results_read_roots),
    write_root=PATHS.results,
)

_data_write_root = DATA_ROOT if not IN_KAGGLE else (PATHS.runs / "data")
DATA_STORE = ArtifactStore(
    read_roots=tuple(_data_read_roots),
    write_root=_data_write_root,
)


DEVICE = device = get_device(prefer_name="RTX")
NUM_WORKERS = 0
SEED = 51
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]