import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from jaguar.evaluation.metrics import ReIDEvalBundle
from jaguar.config import PATHS, DEVICE 

class JaguarTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = DEVICE
        
        # Define experiments paths and folders 
        self.experiment_name = config['training']['experiment_name']
        self.config_folder = config['training']['config_folder']
        
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['training']['epochs']
        )
        
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

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
        path = self.save_dir / self.config_folder 
        os.makedirs(path, exist_ok=True)
        save_path = path / f"{self.experiment_name}_epoch_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, save_path)
        print(f"[Info] Saved checkpoint: {save_path}")