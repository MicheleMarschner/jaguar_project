import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import tomli_w
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from muon import Muon

from jaguar.evaluation.metrics import ReIDEvalBundle
from jaguar.config import PATHS, DEVICE 
from jaguar.utils.utils_scheduler import JaguardIdScheduler

class JaguarTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = DEVICE
        
        # Define experiments paths and folders 
        self.experiment_name = config['training']['experiment_name']
        self.config_folder = config['training']['backbone_name'] #['config_folder'] 
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # self.optimizer = optim.Adam( #optim.AdamW
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     lr=config["scheduler_params"]["lr_start"],
        #     # weight_decay=config['training']['weight_decay']
        # )
        # self.scheduler = JaguardIdScheduler(self.optimizer, **dict(config["scheduler_params"]))
        
        # First version with CosineAnnealingLR for comparison
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=config['training']['epochs']
        # )
        
        # ----------------------------
        # Optimizer
        # ----------------------------
        opt_cfg = config['optimizer']
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_cfg['type'] == "AdamW":
            self.optimizer = torch.optim.AdamW(
                params, lr=opt_cfg['lr'], weight_decay=opt_cfg.get('weight_decay', 0)
            )
        elif opt_cfg['type'] == "Adam":
            self.optimizer = torch.optim.Adam(
                params, lr=opt_cfg['lr'], betas=tuple(opt_cfg.get('betas', [0.9,0.999])), weight_decay=opt_cfg.get('weight_decay', 0)
            )
        elif opt_cfg['type'] == "SGD":
            self.optimizer = torch.optim.SGD(
                params, lr=opt_cfg['lr'], momentum=opt_cfg.get('momentum', 0.9), weight_decay=opt_cfg.get('weight_decay',0)
            )
        elif opt_cfg['type'] == "Muon":  # Muon accepts similar arguments to AdamW
            self.optimizer = Muon(
                params, lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0), betas=tuple(opt_cfg.get("betas", [0.9,0.999]))
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg['type']}")

        # ----------------------------
        # Scheduler
        # ----------------------------
        sched_cfg = config['scheduler']
        sched_type = sched_cfg.get("type", "CosineAnnealingLR")

        if sched_type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sched_cfg['T_max'], eta_min=sched_cfg.get('lr_min',0)
            )
        elif sched_type == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=sched_cfg['lr_max'],
                total_steps=sched_cfg.get('total_steps', None),
                epochs=sched_cfg.get('epochs', None),
                steps_per_epoch=len(self.train_loader),
            )
        elif sched_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sched_cfg.get('factor', 0.1),
                patience=sched_cfg.get('patience', 2),
                min_lr=sched_cfg.get('lr_min', 1e-7)
            )
        elif sched_type == "JaguardIdScheduler":
            self.scheduler = JaguardIdScheduler(self.optimizer, **sched_cfg)
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            imgs = batch["img"].to(self.device)
            labels = batch["label_idx"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # JaguarIDModel returns (loss, logits) when labels are provided
            loss, _ = self.model(imgs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        print("[Info] Validating and computing ReID metrics...")
        for batch in tqdm(self.val_loader, desc="Extracting Val Embeds"):
            imgs = batch["img"].to(self.device)
            # Use the model utility to get normalized embeddings
            emb = self.model.get_embeddings(imgs)
            
            all_embeddings.append(emb.cpu())
            all_labels.append(batch["label_idx"])

        full_embeddings = torch.cat(all_embeddings, dim=0)
        full_labels = torch.cat(all_labels, dim=0)

        bundle = ReIDEvalBundle(
            model=None, 
            embeddings=full_embeddings, 
            labels=full_labels,
            device="cpu"
        )
        return bundle.compute_all()

    def save_checkpoint(self, epoch, metrics):
        path = self.save_dir / self.config_folder / f"{self.experiment_name}"
        os.makedirs(path, exist_ok=True)
        save_path = path / "best_model.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, save_path)
        print(f"[Info] Saved checkpoint: {save_path}")
        
        # save final runtime config
        config_save_path = path / "config_leaderboard_exp.toml"
        with open(config_save_path, "wb") as f:
            tomli_w.dump(self.config, f)

        print(f"[Info] Saved checkpoint: {save_path}")