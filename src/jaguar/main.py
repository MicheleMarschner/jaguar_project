import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import tomllib
import torch
from torch.utils.data import DataLoader, Subset, random_split

from jaguar.config import PATHS, DEVICE, PROJECT_ROOT
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.utils.utils_datasets import (
    load_jaguar_from_FO_export,
    get_stratified_train_val_split,
    BalancedBatchSampler,
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

    # Optionally override experiment name inside config
    if args.experiment_name is not None:
        config.setdefault("experiment", {})
        config["experiment"]["name"] = args.experiment_name

    # Load Dataset (Existing loading logic...)
    _, full_ds = load_jaguar_from_FO_export(
        PATHS.data_export / "init",
        dataset_name="jaguar_init",
        processing_fn=None,
        overwrite_db=False,
    )

    # Calculate Identities
    unique_labels = sorted(list(set([str((s.get("ground_truth")).get("label")) for s in full_ds.samples])))
    num_classes = len(unique_labels)
    print(f"[Info] Identities: {num_classes} | Total Images: {len(full_ds)}")

    # Initialize Model
    model = JaguarIDModel(
        backbone_name=config['model']['backbone_name'],
        num_classes=num_classes,
        head_type=config['model']['head_type'],
        device=DEVICE,
        emb_dim=config['model']['emb_dim'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    # Setup train/val splits  
    full_ds.transform = model.backbone_wrapper.transform 
    train_idx, val_idx, all_labels = get_stratified_train_val_split(
        full_ds, 
        val_split=config['data']['val_split'], 
        seed=config['training']['seed']
    )

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    # Extract labels specific to the training subset for the sampler
    train_subset_labels = [all_labels[i] for i in train_idx]

    # Initialize Balanced Sampler
    custom_batch_sampler = BalancedBatchSampler(
        labels=train_subset_labels,
        batch_size=config['training']['batch_size'],
        samples_per_class=4 # P=8, K=4 for a batch size of 32
    )

    # Create DataLoaders. Note: If using a sampler, 'shuffle' must be False
    train_loader = DataLoader(
        train_ds,
        batch_sampler=custom_batch_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    # Start Training Loop
    config['training']['save_dir'] = PROJECT_ROOT
    trainer = JaguarTrainer(model, train_loader, val_loader, config)
    
    best_mAP = 0.0
    for epoch in range(1, config['training']['epochs'] + 1):
        avg_loss = trainer.train_epoch(epoch)
        metrics = trainer.validate()
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Loss: {avg_loss:.4f} | mAP: {metrics['mAP']:.4f} | Rank-1: {metrics['rank1']:.4f}")
        
        # Save best model
        if metrics['mAP'] > best_mAP:
            best_mAP = metrics['mAP']
            trainer.save_checkpoint(epoch, metrics)
            
        trainer.scheduler.step()

if __name__ == "__main__":
    main()