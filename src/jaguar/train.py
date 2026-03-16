import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import tomli_w
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from muon import SingleDeviceMuonWithAuxAdam
from torch.cuda.amp import GradScaler
from timm.utils import ModelEmaV3

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
        self.config_folder = config['model']['backbone_name'] #['training']['config_folder'] 
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initial setup of optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # EMA Setup
        self.use_ema = self.config['training'].get("ema", False)
        self.ema_decay = self.config['training'].get("ema_decay", 0.995)
        self.ema_model = None
        if self.use_ema:
            print(f"[Trainer] Using EMA with decay={self.ema_decay}")
            self.ema_model = ModelEmaV3(
                model, 
                decay=self.ema_decay
            )

        # AMP Scaler
        self.scaler = GradScaler()

    def _setup_optimizer(self):
        """Logic to create optimizer with optional differential learning rates."""
        opt_cfg = self.config['optimizer']
        train_cfg = self.config['training']
        
        # Separate backbone and head parameters to apply differential LR
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # Differential Learning Rate logic
        # If backbone is frozen, backbone_params will be empty
        backbone_lr = opt_cfg.get("backbone_lr", opt_cfg['lr'] * 0.1)
        
        params_groups = [
            {'params': head_params, 'lr': opt_cfg['lr']},
            {'params': backbone_params, 'lr': backbone_lr}
        ]

        if opt_cfg['type'] == "AdamW":
            self.optimizer = torch.optim.AdamW(
                params_groups, weight_decay=opt_cfg.get('weight_decay', 0)
            )
        elif opt_cfg['type'] == "Adam":
            self.optimizer = torch.optim.Adam(
                params_groups, betas=tuple(opt_cfg.get('betas', [0.9,0.999])), weight_decay=opt_cfg.get('weight_decay', 0)
            )
        elif opt_cfg['type'] == "SGD":
            self.optimizer = torch.optim.SGD(
                params_groups, lr=opt_cfg['lr'], momentum=opt_cfg.get('momentum', 0.9), weight_decay=opt_cfg.get('weight_decay',0)
            )
        elif opt_cfg['type'] == "Muon":
            betas = tuple(opt_cfg.get("betas", [0.9, 0.95]))
            weight_decay = opt_cfg.get("weight_decay", 0.01)

            # Backbone parameters (only those requiring grad)
            backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
            hidden_weights = [p for p in backbone_params if p.ndim >= 2]
            hidden_gains_biases = [p for p in backbone_params if p.ndim < 2]

            # Non-backbone parameters
            nonhidden_params = []
            if hasattr(self.model, "neck"):
                nonhidden_params += [p for p in self.model.neck.parameters() if p.requires_grad]
            if hasattr(self.model, "head"):
                nonhidden_params += [p for p in self.model.head.parameters() if p.requires_grad]

            param_groups = []
            # Muon group (only if backbone has trainable weights)
            if len(hidden_weights) > 0:
                param_groups.append(
                    dict(
                        params=hidden_weights,
                        use_muon=True,
                        lr=opt_cfg["lr_muon"],
                        weight_decay=weight_decay,
                    )
                )
            # Adam group
            adam_params = hidden_gains_biases + nonhidden_params
            if len(adam_params) > 0:
                param_groups.append(
                    dict(
                        params=adam_params,
                        use_muon=False,
                        lr=opt_cfg["lr"],
                        betas=betas,
                        weight_decay=weight_decay,
                    )
                )

            self.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        
        print(f"[Trainer] Optimizer built. Head LR: {opt_cfg['lr']} | Backbone LR: {backbone_lr}")

    def _setup_scheduler(self):
        """Logic to create the scheduler."""
        sched_cfg = self.config['scheduler']
        opt_cfg = self.config['optimizer']
        sched_type = sched_cfg.get("type", "CosineAnnealingLR")

        if sched_type == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sched_cfg['T_max'], eta_min=sched_cfg.get('lr_min',0)
            )
        elif sched_type == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[g["lr"] for g in self.optimizer.param_groups], # differential LR groups are preserved
                epochs=self.config["training"]["epochs"], 
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,
                cycle_momentum=False if opt_cfg['type'] == "Muon" else True,  # Only use momentum cycling for Adam/W, not Muon
            )
        elif sched_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=opt_cfg.get('factor', 0.1),
                patience=opt_cfg.get('patience', 2),
                min_lr=sched_cfg.get('lr_min', 1e-7)
            )
        elif sched_type == "JaguardIdScheduler":
            self.scheduler = JaguardIdScheduler(self.optimizer, **sched_cfg)
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
        
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
        
        # # ----------------------------
        # # Optimizer
        # # ----------------------------
        # opt_cfg = config['optimizer']
        # params = filter(lambda p: p.requires_grad, self.model.parameters())

        # if opt_cfg['type'] == "AdamW":
        #     self.optimizer = torch.optim.AdamW(
        #         params, lr=opt_cfg['lr'], weight_decay=opt_cfg.get('weight_decay', 0)
        #     )
        # elif opt_cfg['type'] == "Adam":
        #     self.optimizer = torch.optim.Adam(
        #         params, lr=opt_cfg['lr'], betas=tuple(opt_cfg.get('betas', [0.9,0.999])), weight_decay=opt_cfg.get('weight_decay', 0)
        #     )
        # elif opt_cfg['type'] == "SGD":
        #     self.optimizer = torch.optim.SGD(
        #         params, lr=opt_cfg['lr'], momentum=opt_cfg.get('momentum', 0.9), weight_decay=opt_cfg.get('weight_decay',0)
        #     )
        # elif opt_cfg['type'] == "Muon":  # Muon accepts similar arguments to AdamW
        #     self.optimizer = Muon(
        #         params, lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0), betas=tuple(opt_cfg.get("betas", [0.9,0.999]))
        #     )
        # else:
        #     raise ValueError(f"Unknown optimizer type: {opt_cfg['type']}")

        # # ----------------------------
        # # Scheduler
        # # ----------------------------
        # sched_cfg = config['scheduler']
        # sched_type = sched_cfg.get("type", "CosineAnnealingLR")

        # if sched_type == "CosineAnnealingLR":
        #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #         self.optimizer, T_max=sched_cfg['T_max'], eta_min=sched_cfg.get('lr_min',0)
        #     )
        # elif sched_type == "OneCycleLR":
        #     self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #         self.optimizer,
        #         max_lr=sched_cfg['lr_max'],
        #         total_steps=sched_cfg.get('total_steps', None),
        #         epochs=sched_cfg.get('epochs', None),
        #         steps_per_epoch=len(self.train_loader),
        #     )
        # elif sched_type == "ReduceLROnPlateau":
        #     self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         self.optimizer,
        #         mode='max',
        #         factor=sched_cfg.get('factor', 0.1),
        #         patience=sched_cfg.get('patience', 2),
        #         min_lr=sched_cfg.get('lr_min', 1e-7)
        #     )
        # elif sched_type == "JaguardIdScheduler":
        #     self.scheduler = JaguardIdScheduler(self.optimizer, **sched_cfg)
        # else:
        #     raise ValueError(f"Unknown scheduler type: {sched_type}")

    def train_epoch(self, epoch):
        # Check if it's time to unfreeze part of the backbone
        unfreeze_ep = self.config['training'].get('unfreeze_epoch', 0)
        if (unfreeze_ep > 0 and epoch == unfreeze_ep): # and self.model.head_type != "triplet"): 
            num_blocks = self.config['training'].get('unfreeze_blocks', 2)
            print(f"\n[Trainer] Triggering Backbone Fine-tuning at epoch {epoch}")
            self.model.unfreeze_backbone_layers(num_blocks)
            
            # Re-build optimizer to include the now-active backbone parameters
            self._setup_optimizer()
            # Re-build scheduler to align with new optimizer
            if self.config["scheduler"]["type"] != "OneCycleLR":
                self._setup_scheduler()

        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            imgs = batch["img"].to(self.device)
            labels = batch["label_idx"].to(self.device)
            
            self.optimizer.zero_grad()
            # # JaguarIDModel returns (loss, logits) when labels are provided
            # loss, _ = self.model(imgs, labels)
            
            # loss.backward()
            # self.optimizer.step()
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.autocast(device_type=device, dtype=torch.float16):
                loss, _ = self.model(imgs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.config["scheduler"]["type"] == "OneCycleLR": #gets optimized per batch
                self.scheduler.step()
            
            if self.use_ema and self.ema_model is not None:
                self.ema_model.update(self.model)
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch=None, loader=None):
        # self.model.eval()
        eval_model = self.ema_model.module if self.use_ema and hasattr(self.ema_model, "module") else self.model
        eval_model.eval()
            
        all_embeddings = []
        all_labels = []
        running_loss = 0.0
        num_batches = 0
        
        eval_loader = loader if loader is not None else self.val_loader
        print("[Info] Validating and computing ReID metrics...")
        for batch in tqdm(eval_loader, desc="Extracting Val Embeds"):
            imgs = batch["img"].to(self.device)
            labels = batch["label_idx"].to(self.device)

            # Use the model utility to get normalized embeddings
            loss, _ = eval_model(imgs, labels)
            emb = eval_model.get_embeddings(imgs) #self.model.get_embeddings(imgs)

            running_loss += loss.item()
            num_batches += 1
            
            all_embeddings.append(emb.cpu())
            all_labels.append(batch["label_idx"])
        
        val_loss = running_loss / max(num_batches, 1)

        full_embeddings = torch.cat(all_embeddings, dim=0)
        full_labels = torch.cat(all_labels, dim=0)
        
        analysis_cfg = self.config.get("mining_analysis", {})
        freq = analysis_cfg.get("silhouette_freq", 5)
        eval_rare = self.config.get("rare_identity_eval", {}).get("enabled", False)
        print(f"RARE EVAL : {eval_rare}")
        
        is_mining_exp = (self.model.head_type == "triplet") or analysis_cfg.get("force_silhouette", False) 
        do_heavy_metrics = is_mining_exp and (epoch is not None and epoch % freq == 0) and not eval_rare
        print(f"DO HEAVY : {do_heavy_metrics}")

        bundle = ReIDEvalBundle(
            model=None, 
            embeddings=full_embeddings, 
            labels=full_labels,
            device="cpu"
        )
        metrics = bundle.compute_all(include_silhouette=do_heavy_metrics)
        metrics["val_loss"] = val_loss
        return metrics, full_embeddings, full_labels, do_heavy_metrics
        # return bundle.compute_all()

    def save_checkpoint(self, epoch, metrics) -> Path:
        path = self.save_dir
        os.makedirs(path, exist_ok=True)
        save_path = path / "best_model.pth"
        save_dict = {                                                ## !TODO should be in experiments folder!!
            'epoch': epoch,
            # 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.ema_model is not None:
            save_dict['model_state_dict'] = self.ema_model.module.state_dict() # EMA weights
        else:
            save_dict['model_state_dict'] = self.model.state_dict()
    
        # if self.use_ema and self.ema_model is not None:
        #     save_dict['ema_state_dict'] = self.ema_model.state_dict()

        torch.save(save_dict, save_path)
        print(f"[Info] Saved checkpoint: {save_path}")
        
        # save final runtime config
        config_save_path = path / "config_leaderboard_exp.toml"         ## !TODO here or in main - should be in experiments folder!!
        with open(config_save_path, "wb") as f:
            tomli_w.dump(self.config, f)

        print(f"[Info] Saved config file: {config_save_path}")

        return config_save_path, save_path