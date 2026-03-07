import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from jaguar.config import DATA_STORE
from jaguar.utils.utils import resolve_path, save_npy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.evaluation.metrics import ReIDEvalBundle
from jaguar.utils.utils_losses import (
    ArcFaceLoss,
    CosFaceLoss,
    SphereFaceLoss,
    TripletLoss,
)

# ---------------------------
# Heads
# ---------------------------

class GeM(nn.Module):
    """Generalized Mean Pooling"""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)

class BaseMarginHead(nn.Module):
    """Base class for margin-based heads like ArcFace, CosFace, SphereFace."""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor):
        features = F.normalize(features, dim=1)
        weights = F.normalize(self.weight, dim=1)
        return F.linear(features, weights)

class LinearHead(nn.Module):
    """Simple linear classification head (Softmax)"""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)

class EmbeddingHead(nn.Module):
    """Projection head (Neck) for ReID features"""
    def __init__(self, in_features: int, emb_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_features, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)  #Added BN

    def forward(self, x):
        x = self.proj(x)
        x = self.bn(x) 
        return x # Return raw for Head, Normalize in get_embeddings

# ---------------------------
# Main Jaguar ID Model
# ---------------------------
class JaguarIDModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        head_type: str = "arcface",
        device: str = "cuda",
        emb_dim: int = 512,
        freeze_backbone: bool = True,
        loss_s: float = 30.0,
        loss_m: float = 0.5,
        use_gem: bool = False,      
        gem_p: float = 3.0,    
        use_projection: bool = True,     
    ):
        super().__init__()
        self.device = device
        self.head_type = head_type.lower()
        # Initialize GeM if needed
        self.use_gem = use_gem
        if self.use_gem:
            self.gem = GeM(p=gem_p)
        # Load foundation backbone
        self.backbone_wrapper = FoundationModelWrapper(backbone_name, device=device)
        self.backbone = self.backbone_wrapper.model
        self.use_projection = use_projection

        # Freeze backbone if needed
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Dynamic feature_dim # --- DYNAMIC DIMENSION INFERENCE ---
        self.backbone.eval()
        input_res = self.backbone_wrapper.input_size # Use the wrapper's config!
        
        with torch.no_grad():
            # Create dummy input based on actual model requirements
            dummy = torch.randn(1, 3, input_res, input_res).to(device)      
            out = self.backbone(dummy)
            if isinstance(out, (tuple, list)): out = out[0]
            if out.ndim > 2: out = out.mean(dim=(2, 3))
            self.feature_dim = out.shape[1]

        print(f"[JaguarID] Backbone: {backbone_name} | Input Res: {input_res} | Feature dim: {self.feature_dim}")

        # Optionally create a projector (EmbeddingHead)
        if self.use_projection:
            self.neck = EmbeddingHead(self.feature_dim, emb_dim)
            head_input_dim = emb_dim
        else:
            self.neck = nn.Identity()
            self.bn = nn.BatchNorm1d(self.feature_dim)
            head_input_dim = self.feature_dim

        print(f"[JaguarID] Feature dim: {self.feature_dim}")

        # Select head + loss
        if self.head_type == "arcface":
            self.head = BaseMarginHead(head_input_dim, num_classes)
            self.criterion = ArcFaceLoss(loss_s, loss_m)
        elif self.head_type == "cosface":
            self.head = BaseMarginHead(head_input_dim, num_classes)
            self.criterion = CosFaceLoss()
        elif self.head_type == "sphereface":
            self.head = BaseMarginHead(head_input_dim, num_classes)
            self.criterion = SphereFaceLoss()
        elif self.head_type == "softmax":
            self.head = LinearHead(head_input_dim, num_classes)
            self.criterion = nn.CrossEntropyLoss()
        elif self.head_type == "triplet":
            self.head = nn.Identity() # Triplet just uses the neck
            self.criterion = TripletLoss()
        else:
            raise ValueError(f"Unknown head type: {head_type}")

        self.to(device)
        
    def unfreeze_backbone_layers(self, num_blocks: int):
        """
        Unfreezes the last 'num_blocks' of the backbone.
        Handles both ViT-style (blocks) and CNN-style (stages/layers) backbones.
        """
        if num_blocks <= 0:
            return
        #Identify the list of modules to choose from
        modules_to_unfreeze = []
      
        # ViT / EVA / DINO / Swin style
        if hasattr(self.backbone, 'blocks'):
            modules_to_unfreeze = list(self.backbone.blocks)
        # ConvNeXt / EfficientNet style
        elif hasattr(self.backbone, 'stages'):
            for stage in self.backbone.stages:
                if hasattr(stage, 'blocks'):
                    modules_to_unfreeze.extend(list(stage.blocks))
        # ResNet / MegaDescriptor style
        elif hasattr(self.backbone, 'layer4'):
            modules_to_unfreeze = list(self.backbone.layer4)

        if not modules_to_unfreeze:
            print("[Warning] Could not detect block structure. Unfreezing entire backbone.")
            for p in self.backbone.parameters():
                p.requires_grad = True
            return
        
        # Unfreeze last n modules
        num_to_unfreeze = min(num_blocks, len(modules_to_unfreeze))
        target_modules = modules_to_unfreeze[-num_to_unfreeze:]
        for mod in target_modules:
            for p in mod.parameters():
                p.requires_grad = True
        
        print(f"[JaguarID] Unfroze the last {num_to_unfreeze} blocks of the backbone.")
    
    def get_embeddings(self, x):
        """Utility method to extract normalized embeddings for ReID evaluation."""
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            if self.use_gem:
                features = self.gem(features).flatten(1)
            else:
                features = features.mean(dim=(2, 3))

        # Pass through the learned projection neck or a batch normalization layer
        embeddings = self.neck(features)
        if not self.use_projection:
            embeddings = self.bn(embeddings)
        return F.normalize(embeddings, dim=1)
    
    def save_embeddings(self, embeddings: np.ndarray, split="training", folder=None):
        if folder is None:
            folder = resolve_path("embeddings", DATA_STORE)
        os.makedirs(folder, exist_ok=True)
        filename = f"embeddings_{self.backbone_wrapper.name}_{self.head_type}_{split}.npy"
        path = os.path.join(folder, filename)
        save_npy(path, embeddings)
        print(f"[Info] Saved embeddings to {path}")
        return path
    
    def load_embeddings(self, split="training", folder=None):
        if folder is None:
            folder = resolve_path("embeddings", DATA_STORE)
        filename = f"embeddings_{self.backbone_wrapper.name}_{self.head_type}_{split}.npy"
        path = os.path.join(folder, filename)
        emb = np.load(path)
        print(f"[Info] Loaded embeddings from {path}, shape={emb.shape}")
        return emb

    def forward(self, x, labels=None):
        features = self.backbone(x)

        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            if self.use_gem:
                features = self.gem(features).flatten(1)
            else:
                features = features.mean(dim=(2, 3))
            
        # Pass through neck or a batch normalization layer
        embeddings = self.neck(features)
        if not self.use_projection:
            embeddings = self.bn(embeddings)

        # Triplet case (returns embeddings)
        if self.head_type == "triplet":
            return F.normalize(embeddings, dim=1)
        # Classification heads
        logits = self.head(embeddings)

        if labels is not None:
            if self.head_type in ["arcface", "cosface", "sphereface"]:
                loss, scaled_logits = self.criterion(logits, labels)
                return loss, scaled_logits
            else:
                return self.criterion(logits, labels), logits
        return logits

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from pathlib import Path
    from jaguar.config import PATHS, DEVICE
    from jaguar.models.foundation_models import FoundationModelWrapper
    from jaguar.utils.utils_datasets import load_jaguar_from_FO_export

    print("[Debug] Running ReID Evaluation...")

    # Load Dataset
    _, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export / "init",
        dataset_name="jaguar_init",
        processing_fn=None,
        overwrite_db=False,
    )
    
    # Calculate unique identities
    unique_labels = sorted(list(set([str((s.get("ground_truth")).get("label")) for s in torch_ds.samples])))
    num_classes = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    print(f"[Info] Dataset loaded: {len(torch_ds)} images, {num_classes} identities.")

    # Initialize the full JaguarID Model
    # Baseline example: MegaDescriptor-L + ArcFace head
    model = JaguarIDModel(
        backbone_name="EVA-02", # DINOv2_for_wildlife, ConvNeXt-V2, EfficientNet-B4, Swin-Transformer, EVA-02, MegaDescriptor-L, MiewID
        num_classes=num_classes,
        head_type="arcface",
        device=str(DEVICE),
        freeze_backbone=True,  # Usually True if just evaluating foundation features
        use_gem=True,   # Set to True to use GeM pooling instead of avg pooling
    )
    model.eval()
    
    # Setup DataLoader 
    torch_ds.transform = model.backbone_wrapper.transform 
    loader = DataLoader(
        torch_ds, 
        batch_size=32, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    example = next(iter(loader))
    print(f"[Debug] Example batch - img shape: {example['img'].shape}")
    
    all_embeddings = []
    all_labels = []
    print(f"[Info] Extracting embeddings through JaguarID (ArcFace pipeline)...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            imgs = batch_data["img"].to(DEVICE)
            
            # Use the model's extraction logic to get embeddings
            emb = model.get_embeddings(imgs)
            all_embeddings.append(emb.cpu())
            all_labels.extend(batch_data["label_idx"])  
            
            if batch_idx >= 5: break # Sanity check limit

    full_embeddings = torch.cat(all_embeddings, dim=0)
    full_labels = torch.tensor(all_labels)

    # Concatenate all batches into one large tensor
    full_embeddings = torch.cat(all_embeddings, dim=0)
    full_labels = torch.tensor(all_labels)
    print(f"[Debug] Final Embeddings shape: {full_embeddings.shape}")

    # Run ReID evaluation bundle
    # We pass the pre-computed embeddings. 
    bundle = ReIDEvalBundle(
        model=None, #we already extracted the final embeddings
        embeddings=full_embeddings, 
        labels=full_labels,
        device="cpu"
    )
    
    results = bundle.compute_all()
    
    print("\n========== ReID Metrics ==========")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")