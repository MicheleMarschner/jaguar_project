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
import fiftyone as fo
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode
from functools import partial
from torch.utils.data import Sampler, Dataset, Subset
from sklearn.model_selection import train_test_split
from pathlib import Path
from collections import defaultdict, Counter
 
from jaguar.config import DATA_ROOT, IMGNET_MEAN, IMGNET_STD, DATA_STORE
from jaguar.preprocessing.preprocessing_background import PROCESSORS
from jaguar.datasets.FiftyOneDataset import FODataset, ManifestDataset
from jaguar.datasets.JaguarDataset import JaguarDataset 
from jaguar.utils.utils import resolve_path


# --------------------Identity distribution analysis --------------------
def analyze_identity_distribution(train_ds, val_ds, save_path=None):
    train_counts = Counter(train_ds.labels_idx)
    val_counts = Counter(val_ds.labels_idx)

    all_ids = sorted(set(train_counts) | set(val_counts))

    rows = []
    for i in all_ids:
        rows.append({
            "identity": i,
            "train_images": train_counts.get(i, 0),
            "val_images": val_counts.get(i, 0),
            "total_images": train_counts.get(i, 0) + val_counts.get(i, 0),
        })

    df = pd.DataFrame(rows).sort_values("total_images")

    if save_path:
        df.to_csv(save_path, index=False)

    return df

def get_rare_identities(identity_df, threshold=3):
    rare_ids = identity_df[identity_df.total_images <= threshold]["identity"].tolist()
    return set(rare_ids)

def build_rare_val_dataset(val_ds, rare_ids):
    rare_indices = [
        i for i, lbl in enumerate(val_ds.labels_idx)
        if lbl in rare_ids
    ]

    return Subset(val_ds, rare_indices)

# --------------------Progressive resizing utilities--------------------
def _round_to_patch(size, patch):
    """Round an image size down to the nearest multiple of the patch size."""
    if patch is None:
        return int(size)
    return int(size // patch * patch)

def _build_progressive_sizes(base_size, patch=None):
    """Create the staged image sizes used for progressive resizing."""
    sizes = [
        int(base_size * 0.6),
        int(base_size * 0.85),
        base_size
    ]
    return [_round_to_patch(s, patch) for s in sizes]

def auto_generate_pr_sizes(model):
    """Generate progressive resizing stages from the model input size."""
    base_size = model.backbone_wrapper.input_size
    if not model.backbone_wrapper.supports_progressive_resizing:
        print(
            f"[ProgressiveResizing] Disabled for backbone "
            f"{model.backbone_wrapper.name} (fixed input size)"
        )
        return [base_size]
    sizes = _build_progressive_sizes(base_size)
    print(f"[ProgressiveResizing] Auto-generated sizes: {sizes}")
    return sizes

def get_resize_for_epoch(epoch, sizes, stage_epochs):
    """Return the resize value assigned to a given training epoch."""
    cumulative = 0
    for size, stage_len in zip(sizes, stage_epochs):
        cumulative += stage_len
        if epoch <= cumulative:
            return size
    return sizes[-1]

# --------------------Helpers for augmentatiosn & transformations--------------------
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
    
def get_transforms(config, model_wrapper, is_training=True, input_size_override=None):
    """Build the transform pipeline for training or evaluation images."""
    registry_entry = model_wrapper.registry_entry
    input_size = input_size_override or registry_entry["input_size"]
    interpolation = InterpolationMode.BICUBIC

    aug_cfg = config.get("augmentation", {})
    transform_list = []

    if is_training and aug_cfg.get("apply_augmentations", False):

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

    transform_list.extend(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
        ]
    )

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
    """Wrap a dataset and apply preprocessing to each image on access."""
    def __init__(self, original_ds, preprocess_fn):
        self.ds = original_ds
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        """Return the number of samples in the wrapped dataset."""
        return len(self.ds)

    def __getitem__(self, idx):
        """Load one sample and apply the preprocessing function to its image."""
        sample = self.ds[idx]
        sample["img"] = self.preprocess_fn(sample["img"])
        return sample
    

def build_processing_fn(config, split: str):
    """Build the configured background-processing function for a data split."""
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
    """Build an evaluation processing function from an explicit processor name."""
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
    """Sample batches with a fixed number of identities and images per identity."""
    def __init__(self, labels, batch_size, samples_per_class=4):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
            
        self.labels_list = list(self.label_to_indices.keys())
        self.num_identities_per_batch = batch_size // samples_per_class

    def __iter__(self):
        """Yield balanced batches of sample indices."""
        num_batches = len(self.labels) // self.batch_size
        
        for _ in range(num_batches):
            batch = []
            selected_ids = random.sample(self.labels_list, self.num_identities_per_batch)
            
            for identity in selected_ids:
                indices = self.label_to_indices[identity]
                replace = len(indices) < self.samples_per_class
                selected_indices = np.random.choice(indices, self.samples_per_class, replace=replace)
                batch.extend(selected_indices.tolist())
            
            random.shuffle(batch)
            yield batch

    def __len__(self):
        """Return the number of batches per epoch."""
        return len(self.labels) // self.batch_size
    

def split_train_val_indices_from_labels(labels, val_split_size: float, seed: int):
    """Split labels into train and validation indices with singleton labels kept in train."""
    labels = [str(x) for x in labels]
    indices = np.arange(len(labels))

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

    train_idx = np.concatenate([train_idx_part, singleton_indices])

    return train_idx, val_idx


def get_stratified_train_val_split(dataset, val_split_size, seed):
    """Create a stratified train/validation split from dataset identity labels."""
    image_labels = [str(s.get("ground_truth").get("label")) for s in dataset.samples]

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
    Create a burst-aware train/validation split with approximate identity stratification.
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
            if has_fp_col and pd.notna(row.get(filepath_col, np.nan)):
                gid = f"single::{row[filepath_col]}"
            else:
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

        if len(id_counts) > 1:
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
        val_split_size=val_split_size,
        seed=seed,
    )

    train_group_ids = set(groups_df.iloc[train_group_idx]["_split_group_id"].tolist())
    val_group_ids = set(groups_df.iloc[val_group_idx]["_split_group_id"].tolist())

    train_mask = out["_split_group_id"].isin(train_group_ids).to_numpy()
    val_mask = out["_split_group_id"].isin(val_group_ids).to_numpy()

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
    """Load the full Jaguar dataset from a FiftyOne or manifest export."""
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
    """Load train and validation Jaguar datasets from a split export."""
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