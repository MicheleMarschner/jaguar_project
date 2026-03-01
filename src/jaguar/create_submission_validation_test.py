import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path

from jaguar.config import PATHS, DEVICE, DATA_ROOT, PROJECT_ROOT
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.datasets.JaguarDataset import JaguarDataset
import torch.nn.functional as F

# --------------------------------------------------
# Minimal identity-balanced mAP function
# --------------------------------------------------
def compute_ib_map_from_embeddings(labels, embeddings):
    N = len(labels)
    identity_map = {}

    sim_matrix = embeddings @ embeddings.T
    sim_matrix = sim_matrix / (np.linalg.norm(embeddings, axis=1, keepdims=True) *
                               np.linalg.norm(embeddings, axis=1, keepdims=True).T)
    sim_matrix = np.clip(sim_matrix, 0, 1)

    for i in range(N):
        q_label = labels[i]
        scores = sim_matrix[i]
        scores[i] = -1.0
        idx_rank = np.argsort(-scores)
        ranked_labels = [labels[j] for j in idx_rank]
        rels = np.array([1 if l==q_label else 0 for l in ranked_labels])
        num_rel = rels.sum()
        if num_rel == 0:
            continue
        precision_at_k = np.cumsum(rels) / (np.arange(len(rels)) + 1)
        ap = (precision_at_k * rels).sum() / num_rel
        identity_map.setdefault(q_label, []).append(ap)

    identity_map = {k: np.mean(v) for k,v in identity_map.items()}
    ib_map = np.mean(list(identity_map.values()))
    return ib_map, identity_map

# --------------------------------------------------
# ReIDEvalBundle (copy your class or import it)
# --------------------------------------------------
from jaguar.evaluation.metrics import ReIDEvalBundle


def compute_validation_map():

    # ------------------------
    # CONFIG
    # ------------------------
    BACKBONE_NAME = "MiewID"
    CHECKPOINT_PATH = "/home/vanessa/Documents/repos/jaguar_project/miewid/jaguar_reid_v1_epoch_20.pth"
    base_path = Path(f"{PROJECT_ROOT}/experiments/round_1")
    PARQUET_PATH = base_path / "splits" / "jaguar_burst__str_closed_set__pol_drop_duplicates__k1" / "full_split.parquet"
    EMB_DIM = 512
    NUM_CLASSES = 31
    BATCH_SIZE = 32

    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = JaguarIDModel(
        backbone_name=BACKBONE_NAME,
        num_classes=NUM_CLASSES,
        head_type="arcface",
        device=DEVICE,
        emb_dim=EMB_DIM
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # ------------------------
    # LOAD VALIDATION DATASET
    # ------------------------
    val_ds = JaguarDataset(
        base_root=PATHS.data_export / "init",
        data_root=DATA_ROOT,
        mode="val",
        is_test=False,
        transform=model.backbone_wrapper.transform,
        split_parquet=PARQUET_PATH
    )

    print(f"[Info] Loaded {len(val_ds)} validation images.")

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ------------------------
    # EXTRACT EMBEDDINGS
    # ------------------------
    all_embeddings = []
    all_labels = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            imgs = batch["img"].to(DEVICE)
            labels = batch["label_idx"]

            emb = model.get_embeddings(imgs)
            emb = F.normalize(emb, dim=1)

            all_embeddings.append(emb.cpu())
            all_labels.extend(labels)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = [val_ds.labels[i] for i in range(len(val_ds))]  # string labels

    print(f"Embeddings shape: {embeddings.shape}, #labels: {len(labels)}")

    # --- Compute identity-balanced mAP (simple function) ---
    ib_map_simple, identity_map_simple = compute_ib_map_from_embeddings(labels, embeddings)
    print(f"\nSimple IB-mAP: {ib_map_simple:.4f}")
        
    # --- Compute mAP using ReIDEvalBundle ---
    bundle = ReIDEvalBundle(embeddings=embeddings, labels=labels, model=None, device=DEVICE)
    ib_map_bundle = bundle.identity_balanced_map()
    map_bundle = bundle.mAP()
    print(f"ReIDEvalBundle IB-mAP: {ib_map_bundle:.4f}")
    print(f"ReIDEvalBundle standard mAP: {map_bundle:.4f}")


if __name__ == "__main__":
    compute_validation_map()