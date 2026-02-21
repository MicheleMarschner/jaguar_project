from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
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
    data: Path
    results: Path
    runs: Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_COLAB = is_colab()

if IN_COLAB:
    PATHS = Paths(
        data=Path("/data"),
        results=Path("/results"),
        runs=Path("/experiments"),
    )
else:
    PATHS = Paths(
        data=PROJECT_ROOT / "data",
        results=PROJECT_ROOT / "results",
        runs=PROJECT_ROOT / "experiments",
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
SEED = 51
