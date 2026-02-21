from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch

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
    )
    
else:
    PATHS = Paths(
        data_train=PROJECT_ROOT / "data/raw/jaguar-re-id/train/train",
        data_test=PROJECT_ROOT / "data/raw/jaguar-re-id/test/test",
        data_export=PROJECT_ROOT / "data/fiftyone/jaguar_export",
        data=PROJECT_ROOT / "data",
        results=PROJECT_ROOT / "results",
        runs=PROJECT_ROOT / "experiments",
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
SEED = 51
