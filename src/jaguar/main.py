
import os

from jaguar.experiments.experiment_runner import load_toml_config
from jaguar.experiments.experiment_xai import deep_update
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import time

from jaguar.utils.utils import ensure_dir, resolve_path, set_seeds
from jaguar.config import EXPERIMENTS_STORE, PATHS, DEVICE, PROJECT_ROOT
from jaguar.utils.utils_experiments import deep_update, load_toml_config
from jaguar.experiments.experiment_output import save_requested_outputs
from jaguar.utils.utils_output import build_output_artifacts
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.utils.utils_datasets import (
    build_processing_fn,
    get_resize_for_epoch,
    load_split_jaguar_from_FO_export,
    BalancedBatchSampler,
    get_transforms,
)
from jaguar.train import JaguarTrainer
from jaguar.logging.wandb_logger import (
    init_wandb_run, log_wandb_dataset_info, log_wandb_epoch_metrics,
    finish_wandb_run, log_wandb_output_artifact, log_wandb_model_info,
    log_wandb_checkpoint_artifact
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train JaguarID model")
    parser.add_argument(
        "--base_config",
        type=str,
        help="Path to the base config TOML file, relative to PATHS.configs and without .toml",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to the experiment override TOML file, relative to PATHS.configs and without .toml",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Optional name of the experiment (used for logging/checkpoints)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Override backbone name from config"
    )
    return parser.parse_args()


def main():
    # Parse CLI arguments
    args = parse_args()

    # Load and merge configs: base + experiment override
    base_config = load_toml_config(args.base_config)
    experiment_config = load_toml_config(args.experiment_config)

    config = deep_update(base_config, experiment_config)

    # Optionally override experiment name inside config
    if args.experiment_name is not None:
        config.setdefault("training", {})
        config["training"]["experiment_name"] = args.experiment_name


    checkpoints_dir = Path(config["training"]["save_dir"])
    exp_name = config["training"]["experiment_name"]
    experiment_group = config.get("output", {}).get("experiment_group")
    print(checkpoints_dir, experiment_group)

    # run artifact directory
    if experiment_group:
        run_dir = PATHS.runs / experiment_group / exp_name
        config["training"]["save_dir"] = str(checkpoints_dir / experiment_group / exp_name)
    else:
        run_dir = PATHS.runs / exp_name
        config["training"]["save_dir"] = str(checkpoints_dir / exp_name)
    ensure_dir(run_dir)

    wandb_run = init_wandb_run(
        config=config,
        run_dir=run_dir,
        exp_name=exp_name,
        experiment_group=experiment_group,
        job_type="train",
    )
    
    print("\n[RUN]")
    print("experiment_name:", exp_name)
    print("run_dir:", run_dir)
            
    parquet_file_path = config["data"]["split_data_path"]
    print(f"\n[DATA] path:{parquet_file_path}")
    parquet_root = resolve_path(parquet_file_path, EXPERIMENTS_STORE)

    set_seeds(config["training"]["seed"])

    train_processing_fn = build_processing_fn(config, split="train")
    val_processing_fn = build_processing_fn(config, split="val")

    pr_cfg = config.get("progressive_resizing", {})
    pr_enabled = pr_cfg.get("enabled", False)
    pr_sizes = pr_cfg.get("sizes", [])
    pr_stage_epochs = pr_cfg.get("stage_epochs", [])

    if pr_enabled:
        if len(pr_sizes) != len(pr_stage_epochs):
            raise ValueError(
                "progressive_resizing.sizes and progressive_resizing.stage_epochs must have the same length"
            )
        if sum(pr_stage_epochs) != config["training"]["epochs"]:
            raise ValueError(
                "Sum of progressive_resizing.stage_epochs must equal training.epochs"
            )

    current_resize = None
    
    # Load pre-split and processed datasets (based on 'mode' in JaguarDataset)
    _, train_ds, val_ds = load_split_jaguar_from_FO_export(
        PATHS.data_export / "splits_curated",
        overwrite_db=False,
        parquet_path=parquet_root,
        train_processing_fn=train_processing_fn,
        val_processing_fn=val_processing_fn,
        include_duplicates=config["split"]["include_duplicates"],
        use_fiftyone=config["data"]["use_fiftyone"]
    )

    # Calculate Identities
    num_classes = len(train_ds.label_to_idx)

    log_wandb_dataset_info(
        run=wandb_run,
        run_dir=run_dir,
        parquet_root=parquet_root,
        train_size=len(train_ds),
        val_size=len(val_ds),
        num_classes=num_classes,
        device=DEVICE,
    )

    # Initialize Model
    model = JaguarIDModel(
        backbone_name=config['model']['backbone_name'],
        num_classes=num_classes,
        head_type=config['model']['head_type'],
        device=DEVICE,
        emb_dim=config['model']['emb_dim'],
        freeze_backbone=config['model']['freeze_backbone'],
        loss_s=config["model"].get("s", 30.0),
        loss_m=config["model"].get("m", 0.5),
        use_projection=config['model']['use_projection'],
        use_forward_features=config['model']['use_forward_features'],
    )

    log_wandb_model_info(
        run=wandb_run,
        model=model,
    )

    if pr_enabled:
        current_resize = get_resize_for_epoch(1, pr_sizes, pr_stage_epochs)
        print(f"[ProgressiveResizing] Initial input size: {current_resize}")

        # Apply transforms directly to pre-split datasets
        train_ds.transform = get_transforms(
            config, model.backbone_wrapper, is_training=True, input_size_override=current_resize
        )
        val_ds.transform = get_transforms(
            config, model.backbone_wrapper, is_training=False, input_size_override=current_resize
        )
    else:
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
        # batch_size=config['training']['batch_size'],
        # shuffle=True,
        batch_sampler=custom_batch_sampler,
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
    #     val_split_size=config['data']['val_split_size'], 
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
    best_metrics = None
    best_epoch = None 
    history = []
    epoch_times = []
    best_checkpoint_path = None

    patience = config["training"].get("early_stopping_patience", 5)
    patience_counter = 0
    monitor_metric = config["training"].get("monitor_metric", "mAP")
    for epoch in range(1, config['training']['epochs'] + 1):
        epoch_start_time = time.perf_counter()
        if pr_enabled:
            epoch_resize = get_resize_for_epoch(epoch, pr_sizes, pr_stage_epochs)
            if epoch_resize != current_resize:
                current_resize = epoch_resize
                print(f"\n[ProgressiveResizing] Switching input size to {current_resize} at epoch {epoch}")

                train_ds.transform = get_transforms(
                    config,
                    model.backbone_wrapper,
                    is_training=True,
                    input_size_override=current_resize,
                )
                val_ds.transform = get_transforms(
                    config,
                    model.backbone_wrapper,
                    is_training=False,
                    input_size_override=current_resize,
                )
                
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
            best_metrics = metrics
            best_epoch = epoch
            patience_counter = 0
            best_checkpoint_path = trainer.save_checkpoint(epoch, metrics)
        else:
            patience_counter += 1
            
        if config["scheduler"]["type"] == "ReduceLROnPlateau":
            trainer.scheduler.step(metrics["mAP"])
        elif config["scheduler"]["type"] != "OneCycleLR":
            trainer.scheduler.step()

        current_lr = trainer.optimizer.param_groups[0]["lr"]

        epoch_time_sec = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_time_sec)

        history.append({
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "val_mAP": float(metrics["mAP"]),
            "val_pairwise_AP": float(metrics["pairwise_AP"]),
            "val_rank1": float(metrics["rank1"]),
            "val_sim_gap": float(metrics["sim_gap"]),
            "lr": float(current_lr),
            "input_size": current_resize if pr_enabled else model.backbone_wrapper.input_size,
            "epoch_time_sec": float(epoch_time_sec),
        })

        log_wandb_epoch_metrics(
            run=wandb_run,
            epoch=epoch,
            avg_loss=avg_loss,
            metrics=metrics,
            current_lr=current_lr,
            epoch_time_sec=epoch_time_sec,
            input_size=current_resize if pr_enabled else model.backbone_wrapper.input_size,
        )
        
        if config["training"].get("early_stopping", False):
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break
                
    final_results = {
        "experiment_name": config["training"]["experiment_name"],
        "best_epoch": best_epoch,
        "monitor_metric": monitor_metric,
        "best_score": best_score,
        "metrics": best_metrics,
    }

    artifacts = build_output_artifacts(
        run_dir=run_dir,
        config=config,
        final_results=final_results,
        train_history=history,
        model=model,
        epoch_times=epoch_times,
    )

    save_requested_outputs(config, artifacts)

    log_wandb_output_artifact(
        run=wandb_run,
        run_dir=run_dir,
        artifact_name=f"{exp_name}-outputs",
    )

    log_wandb_checkpoint_artifact(
        run=wandb_run,
        checkpoint_path=best_checkpoint_path,
        artifact_name=f"{exp_name}-checkpoint",
        metadata={
            "best_epoch": best_epoch,
            "monitor_metric": monitor_metric,
            "best_score": float(best_score),
        },
        aliases=["best"],
    )

    finish_wandb_run(
        run=wandb_run,
        best_epoch=best_epoch,
        best_score=best_score,
        monitor_metric=monitor_metric,
        best_metrics=best_metrics,
        epochs_completed=len(history),
        total_train_time_sec=sum(epoch_times),
    )
        

if __name__ == "__main__":
    main()