from pathlib import Path
import torch
import os
from dataclasses import dataclass
import nltk

@dataclass(frozen=True)
class Paths:
    """Holds the main filesystem paths used by the project"""
    data: Path
    results: Path
    runs: Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (…/BeyondLiteral_Idiomaticity_Detection)

IN_COLAB = True

if IN_COLAB:
    PATHS = Paths(
        data=Path("/data"),
        results=Path("/results"),
        exp=Path("/experiments"),
    )
else:
    PATHS = Paths(
        data=PROJECT_ROOT / "data",
        results=PROJECT_ROOT / "results",
        exp=PROJECT_ROOT / "experiments",
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0     # easier to ensure reproducibility
SEED = 51

