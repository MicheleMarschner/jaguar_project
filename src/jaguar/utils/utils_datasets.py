import fiftyone as fo
import numpy as np
import random 

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

def get_stratified_train_val_split(dataset, val_split, seed):
    # Extract labels for ebery image (1895 items)
    image_labels = [str(s.get("ground_truth").get("label")) for s in dataset.samples]
    indices = np.arange(len(image_labels))
    
    # Handle identities with only 1 image (Singletons)
    counts = Counter(image_labels)
    stratifiable_mask = np.array([counts[lbl] > 1 for lbl in image_labels])
    
    strat_indices = indices[stratifiable_mask]
    strat_labels = [image_labels[i] for i in strat_indices]
    singleton_indices = indices[~stratifiable_mask]

    # Stratified Split (Now lengths match: both are ~1895 minus singletons)
    from sklearn.model_selection import train_test_split
    train_idx_part, val_idx = train_test_split(
        strat_indices,
        test_size=val_split,
        random_state=seed,
        stratify=strat_labels
    )

    # ut singletons in training set so we don't lose classes
    train_idx = np.concatenate([train_idx_part, singleton_indices])

    print(f"[Data] Split Complete: {len(train_idx)} train, {len(val_idx)} val.")
    print(f"[Data] Identities in train: {len(set([image_labels[i] for i in train_idx]))}")
    
    return train_idx, val_idx, image_labels

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