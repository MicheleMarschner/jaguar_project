import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0     # easier to ensure reproducibility
SEED = 51

