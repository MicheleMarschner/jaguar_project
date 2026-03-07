import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import tomllib
from torch.utils.data import DataLoader

from jaguar.config import PATHS, DEVICE, PROJECT_ROOT, WORK_ROOT
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.utils.utils_datasets import (
    load_jaguar_from_FO_export,
    BalancedBatchSampler,
    TransformSubset,
    get_transforms,
)
from jaguar.train import JaguarTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train JaguarID model")
    parser.add_argument(
        "--config",
        type=str,
        default="leaderboard_experiments",
        help="Path to the experiment config TOML file"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional name of the experiment (used for logging/checkpoints)"
    )
    return parser.parse_args()

def main():
    # Parse CLI arguments
    args = parse_args()

    # Load Config
    with open(PATHS.configs / f"{args.config}.toml", "rb") as f:
        config = tomllib.load(f)
        
    round_name = config["training"]["experiment_name"]
    parquet_file_path = config["data"]["parquet_path"]
    parquet_root = f"{WORK_ROOT}/experiments/{round_name}/{parquet_file_path}"

    # Optionally override experiment name inside config
    if args.experiment_name is not None:
        config.setdefault("experiment", {})
        config["experiment"]["name"] = args.experiment_name

    is_full_ds = config["data"].get("full_ds", False)
    
    # Load Dataset (Existing loading logic...)
    if is_full_ds:
        parquet_root=None
        # Load one large dataset and split it later
        _, train_ds = load_jaguar_from_FO_export(
            PATHS.data_export / "init",
            dataset_name="jaguar_init",
            overwrite_db=False,
            parquet_path=parquet_root,
            full_ds=True,
            include_duplicates=config["data"]["include_duplicates"],
        )
    else:
        # Load pre-split and processed datasets (based on 'mode' in JaguarDataset)
        _, train_ds, val_ds = load_jaguar_from_FO_export(
            PATHS.data_export / "init",
            dataset_name="jaguar_init",
            overwrite_db=False,
            parquet_path=parquet_root,
            full_ds=False,
            include_duplicates=config["data"]["include_duplicates"],
        )
        
    # _, train_ds, val_ds = load_jaguar_from_FO_export(
    #     PATHS.data_export / "init",
    #     dataset_name="jaguar_init",
    #     processing_fn=None,
    #     overwrite_db=False,
    #     parquet_path=parquet_root,
    #     full_ds=False,
    # )

    # Calculate Identities
    unique_labels = sorted(list(set([str((s.get("ground_truth")).get("label")) for s in train_ds.samples])))
    num_classes = len(unique_labels)

    # Initialize Model
    model = JaguarIDModel(
        backbone_name=config['model']['backbone_name'],
        num_classes=num_classes,
        head_type=config['model']['head_type'],
        device=DEVICE,
        emb_dim=config['model']['emb_dim'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    if is_full_ds: 
        print("[Data] Splitting full dataset...")
        train_idx, val_idx, _ = get_stratified_train_val_split(
            train_ds, 
            val_split=config['data']['val_split'], 
            seed=config['training']['seed']
        )
        # Create Subsets for train/val splits
        train_ds_old = train_ds
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(train_ds, val_idx)
        
        # Apply Transforms via wrapper and modify/apply transforms from the Model Backbone
        train_ds = TransformSubset(train_subset, get_transforms(config, model.backbone_wrapper, is_training=True))
        val_ds = TransformSubset(val_subset, get_transforms(config, model.backbone_wrapper, is_training=False))
        # Labels for Sampler (must align with numeric indices in train_idx)
        train_labels = [train_ds_old.labels_idx[i] for i in train_idx]
    else:
        # Apply transforms directly to pre-split datasets
        train_ds.transform = get_transforms(config, model.backbone_wrapper, is_training=True)
        val_ds.transform = get_transforms(config, model.backbone_wrapper, is_training=False)        
        # Labels for Sampler
        train_labels = train_ds.labels_idx
    
    print(f"[JaguarIDModelInfo] Training Identities: {num_classes} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Set Transforms from the Model Backbone
    # train_ds.transform = model.backbone_wrapper.transform
    # val_ds.transform = model.backbone_wrapper.transform
    # train_ds.transform = get_transforms(config, model.backbone_wrapper, is_training=True)
    # val_ds.transform = get_transforms(config, model.backbone_wrapper, is_training=False)

    # Initialize Balanced Sampler for Re-ID
    # We use train_ds.labels_idx which contains numeric IDs for every sample
    custom_batch_sampler = BalancedBatchSampler(
        labels=train_labels, #train_ds.labels_idx,
        batch_size=config['training']['batch_size'],
        samples_per_class=config['training'].get('samples_per_class', 4) 
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        # batch_sampler=custom_batch_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # # Setup train/val splits  
    # full_ds.transform = model.backbone_wrapper.transform 
    # train_idx, val_idx, all_labels = get_stratified_train_val_split(
    #     full_ds, 
    #     val_split=config['data']['val_split'], 
    #     seed=config['training']['seed']
    # )

    # train_ds = Subset(full_ds, train_idx)
    # val_ds = Subset(full_ds, val_idx)

    # # Extract labels specific to the training subset for the sampler
    # train_subset_labels = [all_labels[i] for i in train_idx]

    # # Initialize Balanced Sampler
    # custom_batch_sampler = BalancedBatchSampler(
    #     labels=train_subset_labels,
    #     batch_size=config['training']['batch_size'],
    #     samples_per_class=4 # P=8, K=4 for a batch size of 32
    # )

    # # Create DataLoaders. Note: If using a sampler, 'shuffle' must be False
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_sampler=custom_batch_sampler,
    #     num_workers=config['data']['num_workers'],
    #     pin_memory=True
    # )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=config['training']['batch_size'],
    #     shuffle=False,
    #     num_workers=config['data']['num_workers']
    # )
    # Start Training Loop
    if not config['training']['save_dir']:
        config['training']['save_dir'] = PROJECT_ROOT
    trainer = JaguarTrainer(model, train_loader, val_loader, config)
    
    best_score = 0.0   
    patience = config["training"].get("early_stopping_patience", 5)
    patience_counter = 0
    monitor_metric = config["training"].get("monitor_metric", "mAP")
    for epoch in range(1, config['training']['epochs'] + 1):
        avg_loss = trainer.train_epoch(epoch)
        metrics = trainer.validate()
        
        print(f"\nEpoch {epoch} Summary:")
        print(
            f"Train Loss: {avg_loss:.4f} | "
            f"Val mAP: {metrics['mAP']:.4f} | "
            f"Val pairAP: {metrics['pairwise_AP']:.4f} | "
            f"Val Rank1: {metrics['rank1']:.4f} | "
            f"Val SimGap: {metrics['sim_gap']:.4f}"
        )
        
        # Save best model
        current_score = metrics[monitor_metric]
        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
            trainer.save_checkpoint(epoch, metrics)
        else:
            patience_counter += 1
            
        if config['scheduler']['type'] == "ReduceLROnPlateau":
            trainer.scheduler.step(metrics['mAP'])
        else:  
            trainer.scheduler.step()
        
        if config["training"].get("early_stopping", False):
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

if __name__ == "__main__":
    main()