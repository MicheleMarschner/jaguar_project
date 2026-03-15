try:
    import fiftyone as fo
    HAS_FIFTYONE = True
except ImportError:
    fo = None
    HAS_FIFTYONE = False

import numpy as np
import random 
import pandas as pd
import torch 
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode
from functools import partial
from torch.utils.data import Sampler, Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from collections import defaultdict, Counter
 
from jaguar.config import DATA_ROOT, PATHS, IMGNET_MEAN, IMGNET_STD
from jaguar.preprocessing.preprocessing_background import PROCESSORS
from jaguar.datasets.FiftyOneDataset import FODataset, ManifestDataset
from jaguar.datasets.JaguarDataset import JaguarDataset 
from jaguar.utils.utils import resolve_path

# --------------------Progressive resizing utilities--------------------
def round_to_patch(size, patch):
    if patch is None:
        return int(size)
    return int(size // patch * patch)

def build_progressive_sizes(base_size, patch=None):
    sizes = [
        int(base_size * 0.6),
        int(base_size * 0.85),
        base_size
    ]
    return [round_to_patch(s, patch) for s in sizes]

def auto_generate_pr_sizes(model):
    base_size = model.backbone_wrapper.input_size
    if not model.backbone_wrapper.supports_progressive_resizing:
        print(
            f"[ProgressiveResizing] Disabled for backbone "
            f"{model.backbone_wrapper.name} (fixed input size)"
        )
        return [base_size]
    sizes = build_progressive_sizes(base_size)
    print(f"[ProgressiveResizing] Auto-generated sizes: {sizes}")
    return sizes

def get_resize_for_epoch(epoch, sizes, stage_epochs):
    cumulative = 0
    for size, stage_len in zip(sizes, stage_epochs):
        cumulative += stage_len
        if epoch <= cumulative:
            return size
    return sizes[-1]


"""
# --- Helper for different transforms on Subsets if full_ds is used ---
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None, processing_fn=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.subset[index]

        if self.transform:
            # Assuming the dataset returns a dict with "img"
            sample["img"] = self.transform(sample["img"])
        return sample
        
    def __len__(self):
        return len(self.subset)
"""

def get_resize_for_epoch(epoch: int, sizes: list[int], stage_epochs: list[int]) -> int:
    cum = 0
    for size, n_epochs in zip(sizes, stage_epochs):
        cum += n_epochs
        if epoch <= cum:
            return size
    return sizes[-1]

def get_transforms(config, model_wrapper, is_training=True, input_size_override=None):
    # Extract model-specific requirements from the wrapper's registry
    registry_entry = model_wrapper.registry_entry
    input_size = input_size_override or registry_entry["input_size"]
    # Default to BICUBIC if not specified
    interpolation = InterpolationMode.BICUBIC

    aug_cfg = config.get("augmentation", {})
    transform_list = []

    # Add Training Augmentations
    if is_training and aug_cfg.get("apply_augmentations", False):

        # RandomResizedCrop instead of Resize
        if aug_cfg.get("random_resized_crop", True):
            transform_list.append(
                transforms.RandomResizedCrop(
                    input_size,
                    scale=(0.6, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=interpolation,
                )
            )
        else:
            # Start with Resize by default 
            transform_list.append(
                transforms.Resize((input_size, input_size), interpolation=interpolation)
            )

        if aug_cfg.get("horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())

        if aug_cfg.get("affine_degrees", 0) > 0:
            transform_list.append(
                transforms.RandomAffine(
                    degrees=aug_cfg.get("affine_degrees", 0),
                    translate=tuple(aug_cfg.get("affine_translate", [0, 0])),
                    scale=tuple(aug_cfg.get("affine_scale", [1, 1])),
                )
            )

        if aug_cfg.get("gaussian_blur", False):
            transform_list.append(
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            )

        if (
            aug_cfg.get("color_jitter_brightness", 0) > 0
            or aug_cfg.get("color_jitter_contrast", 0) > 0
        ):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=aug_cfg.get("color_jitter_brightness", 0),
                    contrast=aug_cfg.get("color_jitter_contrast", 0),
                    saturation=aug_cfg.get("color_jitter_saturation", 0.1),
                )
            )

    else:

        transform_list.append(
            transforms.Resize((input_size, input_size), interpolation=interpolation)
        )

    # Final Steps (Conversion & Normalization)
    transform_list.extend(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
        ]
    )
    # Post-Normalization Augmentations (Random Erasing)
    if is_training and aug_cfg.get("random_erasing_p", 0) > 0:
        transform_list.append(
            transforms.RandomErasing(
                p=aug_cfg.get("random_erasing_p"),
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3),
            )
        )
    return transforms.Compose(transform_list)

class PreprocessedDataset(Dataset):
    """
    A simple wrapper that applies the model_wrapper's preprocessing 
    to each sample before the DataLoader tries to batch them.
    """
    def __init__(self, original_ds, preprocess_fn):
        self.ds = original_ds
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # Apply preprocessing here (inside the worker process)
        # This includes resizing, so all images will now have the same size.
        sample["img"] = self.preprocess_fn(sample["img"])
        return sample
    

def build_processing_fn(config, split: str):
    """
    split: 'train' or 'val'
    """
    pre_cfg = config.get("preprocessing", {})
    processor_name = pre_cfg.get(f"{split}_background", "original")

    if processor_name == "original":
        return None

    processor = PROCESSORS[processor_name]

    kwargs = {
        "base_root": resolve_path("fiftyone/init", DATA_STORE),
        "bg_dir": resolve_path(pre_cfg.get("bg_dir"), DATA_STORE),
        "edge_softness": pre_cfg.get("edge_softness", 0),
        "blur_radius": pre_cfg.get("blur_radius", 10),
        "p_original": pre_cfg.get("p_original", 0.5),
        "key_for_seed": pre_cfg.get("key_for_seed", "filename"),
    }

    return partial(processor, **kwargs)

def build_eval_processing_fn(
    processor_name: str,
    config: dict,
):
    """
    Build a processing function exactly like build_processing_fn(...),
    except that the processor name is passed explicitly instead of being
    read from config['preprocessing'].
    """
    if processor_name == "original":
        return None

    processor = PROCESSORS[processor_name]
    pre_cfg = config.get("preprocessing", {})

    kwargs = {
        "base_root": resolve_path("fiftyone/init", DATA_STORE),
        "bg_dir": resolve_path(pre_cfg.get("bg_dir"), DATA_STORE),
        "edge_softness": pre_cfg.get("edge_softness", 0),
        "blur_radius": pre_cfg.get("blur_radius", 10),
        "p_original": pre_cfg.get("p_original", 0.5),
        "key_for_seed": pre_cfg.get("key_for_seed", "filename"),
    }

    return partial(processor, **kwargs)

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
    

def split_train_val_indices_from_labels(labels, val_split_size: float, seed: int):
    """
    - stratify labels with frequency > 1
    - labels occurring once are forced into train
    """

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
        test_size=val_split_size,
        random_state=seed,
        stratify=strat_labels,
    )

    # keep singleton-label units in train so classes are not lost
    train_idx = np.concatenate([train_idx_part, singleton_indices])

    return train_idx, val_idx


def get_stratified_train_val_split(dataset, val_split_size, seed):
    # Extract labels for every image
    image_labels = [str(s.get("ground_truth").get("label")) for s in dataset.samples]

    # Reuse shared core
    train_idx, val_idx = split_train_val_indices_from_labels(
        labels=image_labels,
        val_split_size=val_split_size,
        seed=seed,
    )

    print(f"[Data] Split Complete: {len(train_idx)} train, {len(val_idx)} val.")
    print(f"[Data] Identities in train: {len(set([image_labels[i] for i in train_idx]))}")
    
    return train_idx, val_idx, image_labels


def get_group_aware_stratified_train_val_split(
    df: pd.DataFrame,
    val_split_size: float,
    seed: int,
    identity_col: str = "identity_id",
    burst_group_col: str = "burst_group_id",
    filepath_col: str = "filename",
):
    """
    Closed-set split helper: burst-preserving grouping + approximate identity stratification at group level.

    Split semantics:
    - Primary constraint: keep all images from the same burst group together (no burst leakage).
    - Secondary objective: approximate identity stratification at the *group* level.
      (i.e., stratify groups by a representative identity label, not individual rows)
    """
    out = df.copy().reset_index(drop=True)

    group_ids = []
    has_burst_col = burst_group_col in out.columns
    has_fp_col = filepath_col in out.columns

    for i, row in out.iterrows():
        bg = row.get(burst_group_col, np.nan) if has_burst_col else np.nan

        if pd.notna(bg):
            gid = f"burst::{bg}"
        else:
            # Use filepath as stable singleton group key (better than row index for traceability).
            if has_fp_col and pd.notna(row.get(filepath_col, np.nan)):
                gid = f"single::{row[filepath_col]}"
            else:
                # Fallback only if filepath is unavailable.
                gid = f"single_row::{i}"
        group_ids.append(gid)

    out["_split_group_id"] = group_ids

    
    group_rows = []
    for gid, g in out.groupby("_split_group_id", sort=False):
        ids = g[identity_col].dropna().astype(str).tolist()
        if len(ids) == 0:
            raise ValueError(f"Group {gid} has no valid identity labels.")

        id_counts = Counter(ids)
        group_identity, group_identity_count = id_counts.most_common(1)[0]

        # Mixed-identity burst groups should usually not happen; warn so data issues are visible.
        if len(id_counts) > 1:
            print(f"[WARN] Mixed identities in group {gid}: {dict(id_counts)} -> using majority '{group_identity}'")

        group_rows.append(
            {
                "_split_group_id": gid,
                "group_identity": group_identity,  # label used for group-level stratification
                "group_size": int(len(g)),         # informative: how many images this unit contributes
                "n_unique_identities_in_group": int(len(id_counts)),
                "group_identity_majority_count": int(group_identity_count),
            }
        )

    groups_df = pd.DataFrame(group_rows)
    
    train_group_idx, val_group_idx = split_train_val_indices_from_labels(
        labels=groups_df["group_identity"].tolist(),
        val_split_size=val_split_size,
        seed=seed,
    )

    train_group_ids = set(groups_df.iloc[train_group_idx]["_split_group_id"].tolist())
    val_group_ids = set(groups_df.iloc[val_group_idx]["_split_group_id"].tolist())

    train_mask = out["_split_group_id"].isin(train_group_ids).to_numpy()
    val_mask = out["_split_group_id"].isin(val_group_ids).to_numpy()

    # Safety checks: ensure a strict partition of rows.
    if np.any(train_mask & val_mask):
        raise RuntimeError("Group leakage detected: some rows assigned to both train and val.")
    if not np.all(train_mask | val_mask):
        raise RuntimeError("Some rows were not assigned to train or val.")

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    return train_idx, val_idx, out, groups_df


def load_full_jaguar_from_FO_export(
    manifest_dir,
    dataset_name="jaguar_init",
    transform=None,
    processing_fn=None,
    overwrite_db=False,
    use_fiftyone: bool = True,
):
    """
    Load the full Jaguar dataset from a FiftyOne export manifest.

    Used for setup stages that operate on the complete dataset
    before train/val splitting (e.g. burst discovery, split creation, EDA).
    """
    manifest_dir = Path(manifest_dir)

    if use_fiftyone and HAS_FIFTYONE:
        if dataset_name in fo.list_datasets() and not overwrite_db:
            aux_ds = FODataset(dataset_name, overwrite=False)
        else:
            aux_ds = FODataset.load_manifest(
                export_dir=manifest_dir,
                dataset_name=dataset_name,
                overwrite_db=overwrite_db,
            )
    else:
        aux_ds = ManifestDataset(manifest_dir)

    torch_ds = JaguarDataset(
        base_root=manifest_dir,
        data_root=DATA_ROOT,
        mode="full",
        split_parquet=None,
        transform=transform,
        processing_fn=processing_fn,
        include_duplicates=True,
    )

    return aux_ds, torch_ds


def load_split_jaguar_from_FO_export(
    manifest_dir,
    dataset_name="jaguar_splits_curated",
    transform=None,
    train_processing_fn=None,
    val_processing_fn=None,
    overwrite_db=False,
    include_duplicates=False,
    parquet_path: str = None,
    use_fiftyone: bool = True,
):
    """
    Load pre-split Jaguar train/val datasets from a FiftyOne export manifest
    plus a split parquet file.
    """
    manifest_dir = Path(manifest_dir)

    if use_fiftyone and HAS_FIFTYONE:
        if dataset_name in fo.list_datasets() and not overwrite_db:
            aux_ds = FODataset(dataset_name, overwrite=False)
        else:
            aux_ds = FODataset.load_manifest(
                export_dir=manifest_dir,
                dataset_name=dataset_name,
                overwrite_db=overwrite_db,
            )
    else:
        aux_ds = ManifestDataset(manifest_dir)

    torch_ds_train = JaguarDataset(
        base_root=manifest_dir,
        data_root=DATA_ROOT,
        mode="train",
        split_parquet=parquet_path,
        transform=transform,
        processing_fn=train_processing_fn,
        include_duplicates=include_duplicates,
    )

    torch_ds_val = JaguarDataset(
        base_root=manifest_dir,
        data_root=DATA_ROOT,
        mode="val",
        split_parquet=parquet_path,
        transform=transform,
        processing_fn=val_processing_fn,
        include_duplicates=include_duplicates,
    )

    return aux_ds, torch_ds_train, torch_ds_val



