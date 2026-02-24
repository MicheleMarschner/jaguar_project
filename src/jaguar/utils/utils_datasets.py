import fiftyone as fo
from jaguar.config import PATHS
from jaguar.utils.utils import ensure_dir
import numpy as np
import random 
import pandas as pd

from torch.utils.data import Subset, WeightedRandomSampler
from torch.utils.data import Sampler, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from collections import defaultdict, Counter

from jaguar.datasets.FiftyOneDataset import FODataset
from jaguar.datasets.JaguarDataset import JaguarDataset 

class BalancedBatchSampler(Sampler):
    """
    Returns batches of size (P * K) where:
    P = number of identities per batch
    K = number of images per identity
    """
    def __init__(self, labels, batch_size, samples_per_class=4):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        
        # Create a mapping of {identity_label: [list_of_indices]}
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
            
        self.labels_list = list(self.label_to_indices.keys())
        self.num_identities_per_batch = batch_size // samples_per_class

    def __iter__(self):
        num_batches = len(self.labels) // self.batch_size
        
        for _ in range(num_batches):
            batch = []
            # Select P random identities
            selected_ids = random.sample(self.labels_list, self.num_identities_per_batch)
            
            for identity in selected_ids:
                indices = self.label_to_indices[identity]
                # Pick K random images for this identity
                # Use choices (with replacement) if identity has fewer than K images
                replace = len(indices) < self.samples_per_class
                selected_indices = np.random.choice(indices, self.samples_per_class, replace=replace)
                batch.extend(selected_indices.tolist())
            
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return len(self.labels) // self.batch_size
    

def split_train_val_indices_from_labels(labels, val_split: float, seed: int):
    """
    - stratify labels with frequency > 1
    - labels occurring once are forced into train
    """
    # ut singletons in training set so we don't lose classes
    train_idx = np.concatenate([train_idx_part, singleton_indices])

    labels = [str(x) for x in labels]
    indices = np.arange(len(labels))

    # Handle identities with only 1 image (Singletons)
    counts = Counter(labels)
    stratifiable_mask = np.array([counts[lbl] > 1 for lbl in labels])

    strat_indices = indices[stratifiable_mask]
    strat_labels = [labels[i] for i in strat_indices]
    singleton_indices = indices[~stratifiable_mask]

    if len(strat_indices) == 0:
        raise ValueError("No stratifiable labels found (all labels occur only once).")

    train_idx_part, val_idx = train_test_split(
        strat_indices,
        test_size=val_split,
        random_state=seed,
        stratify=strat_labels,
    )

    # keep singleton-label units in train so classes are not lost
    train_idx = np.concatenate([train_idx_part, singleton_indices])

    return train_idx, val_idx


def get_stratified_train_val_split(dataset, val_split, seed):
    # Extract labels for every image
    image_labels = [str(s.get("ground_truth").get("label")) for s in dataset.samples]

    # Reuse shared core
    train_idx, val_idx = split_train_val_indices_from_labels(
        labels=image_labels,
        val_split=val_split,
        seed=seed,
    )

    print(f"[Data] Split Complete: {len(train_idx)} train, {len(val_idx)} val.")
    print(f"[Data] Identities in train: {len(set([image_labels[i] for i in train_idx]))}")
    
    return train_idx, val_idx, image_labels


def get_group_aware_stratified_train_val_split(
    df: pd.DataFrame,
    val_split: float,
    seed: int,
    identity_col: str = "identity_id",
    burst_group_col: str = "burst_group_id",
    filepath_col: str = "filepath",
    verbose: bool = True,
):
    """
    Burst-aware train/val split
    Primary constraint: Keep all rows belonging to the same burst group in the same split.
    Secondary objective: Stratify approximately by identity at group level
    """
    out = df.copy().reset_index(drop=True)

    if identity_col not in out.columns:
        raise ValueError(f"Missing required column: {identity_col}")

    # ------------------------------------------------------------
    # Build split unit per row:
    #    - rows with burst_group_id share one group
    #    - rows without burst_group_id become singleton groups
    # ------------------------------------------------------------
    group_ids = []
    has_burst_col = burst_group_col in out.columns
    has_fp_col = filepath_col in out.columns

    for i, row in out.iterrows():
        bg = row.get(burst_group_col, np.nan) if has_burst_col else np.nan

        if pd.notna(bg):
            gid = f"burst::{bg}"
        else:
            if has_fp_col and pd.notna(row.get(filepath_col, np.nan)):
                gid = f"single::{row[filepath_col]}"
            else:
                gid = f"single_row::{i}"
        group_ids.append(gid)

    out["_split_group_id"] = group_ids

    # ------------------------------------------------------------
    # Build one label per group (majority identity if mixed)
    # ------------------------------------------------------------
    group_rows = []
    for gid, g in out.groupby("_split_group_id", sort=False):
        ids = g[identity_col].dropna().astype(str).tolist()
        if len(ids) == 0:
            raise ValueError(f"Group {gid} has no valid identity labels.")

        id_counts = Counter(ids)
        group_identity, group_identity_count = id_counts.most_common(1)[0]

        if len(id_counts) > 1 and verbose:
            print(f"[WARN] Mixed identities in group {gid}: {dict(id_counts)} -> using majority '{group_identity}'")

        group_rows.append(
            {
                "_split_group_id": gid,
                "group_identity": group_identity,
                "group_size": int(len(g)),
                "n_unique_identities_in_group": int(len(id_counts)),
                "group_identity_majority_count": int(group_identity_count),
            }
        )

    groups_df = pd.DataFrame(group_rows)


    train_group_idx, val_group_idx = split_train_val_indices_from_labels(
        labels=groups_df["group_identity"].tolist(),
        val_split=val_split,
        seed=seed,
    )

    train_group_ids = set(groups_df.iloc[train_group_idx]["_split_group_id"].tolist())
    val_group_ids = set(groups_df.iloc[val_group_idx]["_split_group_id"].tolist())

    # ------------------------------------------------------------
    # Map group assignments back to row/image indices
    # ------------------------------------------------------------
    train_mask = out["_split_group_id"].isin(train_group_ids).to_numpy()
    val_mask = out["_split_group_id"].isin(val_group_ids).to_numpy()

    if np.any(train_mask & val_mask):
        raise RuntimeError("Group leakage detected: some rows assigned to both train and val.")
    if not np.all(train_mask | val_mask):
        raise RuntimeError("Some rows were not assigned to train or val.")

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    # ------------------------------------------------------------
    # Diagnostics + group integrity check
    # ------------------------------------------------------------
    if verbose:
        print(f"[BurstAwareSplit] Rows: train={len(train_idx)} | val={len(val_idx)}")
        print(f"[BurstAwareSplit] Groups: train={len(train_group_ids)} | val={len(val_group_ids)}")

        row_labels = out[identity_col].astype(str).tolist()
        train_ids = set(row_labels[i] for i in train_idx)
        val_ids = set(row_labels[i] for i in val_idx)
        print(
            f"[BurstAwareSplit] Identities: train={len(train_ids)} | "
            f"val={len(val_ids)} | overlap={len(train_ids & val_ids)} (closed-set overlap allowed)"
        )

        tmp = out.copy()
        tmp["__split"] = None
        tmp.loc[train_idx, "__split"] = "train"
        tmp.loc[val_idx, "__split"] = "val"

        per_group_split_nunique = tmp.groupby("_split_group_id")["__split"].nunique()
        if (per_group_split_nunique > 1).any():
            raise RuntimeError("Burst/group integrity violated: a group spans multiple splits.")

        print("[BurstAwareSplit] [OK] Burst groups kept intact.")

    return train_idx, val_idx, out, groups_df


def load_jaguar_from_FO_export(
    manifest_dir,
    dataset_name="jaguar_stage0",
    transform=None,
    processing_fn=None,
    overwrite_db=False,
):
    """
    - If dataset exists in FiftyOne DB: load it
    - Else: import from manifest_dir/samples.json into DB via load_manifest()
    Returns: (fo_wrapper, fo_dataset, torch_dataset)
    """
    manifest_dir = Path(manifest_dir)

    if dataset_name in fo.list_datasets() and not overwrite_db:
        fo_ds = FODataset(dataset_name, overwrite=False)
    else:
        fo_ds = FODataset.load_manifest(
            export_dir=manifest_dir,
            dataset_name=dataset_name,
            overwrite_db=overwrite_db,
        )

    # Torch dataset reads the same samples.json and uses absolute paths inside it
    torch_ds = JaguarDataset(
        base_root=manifest_dir,
        filepath_key="filepath",
        transform=transform,
        processing_fn=processing_fn,
    )

    return fo_ds, torch_ds 


def load_or_extract_embeddings(model_wrapper, torch_ds, split="training"):
    """
    Returns embeddings as np.ndarray [N,D].
    Loads from disk if available; otherwise extracts and saves.
    """
    folder = PATHS.data / "embeddings"
    ensure_dir(folder)

    filename = f"embeddings_{model_wrapper.name}_{split}.npy"
    path = folder / filename

    if path.exists():
        emb = np.load(path)
        print(f"[Info] Loaded embeddings from {path}, shape={emb.shape}")
        return emb

    print(f"[Info] Embeddings not found at {path}. Extracting...")
    imgs_all = [torch_ds[k]["img"] for k in range(len(torch_ds))]  # PIL images
    emb = model_wrapper.extract_embeddings(imgs_all)               # np.ndarray [N,D]
    np.save(path, emb)
    print(f"[Info] Saved embeddings to {path}")
    return emb