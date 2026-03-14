import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from jaguar.config import DATA_STORE
from jaguar.utils.utils import resolve_path, save_npy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from jaguar.models.foundation_models import FoundationModelWrapper
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
        use_forward_features: bool = False,  
        mining_type: str = "hard",
        label_smooth=0.0,
    ):
        super().__init__()
        self.device = device
        self.head_type = head_type.lower()
        self.use_forward_features = use_forward_features
        
        if self.head_type == "triplet": use_projection = True
        
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
        input_res = self.backbone_wrapper.input_size 
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, input_res, input_res).to(device)
            if self.use_forward_features and hasattr(self.backbone, "forward_features"):
                out = self.backbone.forward_features(dummy)
                # handle token -> map if necessary
                if out.ndim == 3:  # [B, N, C] from ViT-like backbones
                    B, N, C = out.shape
                    H = W = int(N ** 0.5)
                    out = out[:, : H * W, :]
                    out = out.permute(0, 2, 1).reshape(B, C, H, W)
            else:
                out = self.backbone(dummy)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            if out.ndim > 2:
                out = out.mean(dim=(2, 3))
            self.feature_dim = out.shape[1]

        # Optionally create a projector (EmbeddingHead)
        if self.use_projection:
            self.neck = EmbeddingHead(self.feature_dim, emb_dim)
            head_input_dim = emb_dim
        else:
            self.neck = nn.Identity()
            self.bn = nn.BatchNorm1d(self.feature_dim)
            head_input_dim = self.feature_dim

<<<<<<< HEAD
=======
        print(f"[JaguarID] Feature dim: {self.feature_dim}")     

>>>>>>> 6c72455f70423f354c26d0c011032db04b0b84a0
        # Select head + loss
        if self.head_type == "arcface":
            self.head = BaseMarginHead(head_input_dim, num_classes)
            self.criterion = ArcFaceLoss(loss_s, loss_m)
        elif self.head_type == "cosface":
            self.head = BaseMarginHead(head_input_dim, num_classes)
            self.criterion = CosFaceLoss(loss_s, loss_m)
        elif self.head_type == "sphereface":
            self.head = BaseMarginHead(head_input_dim, num_classes)
            self.criterion = SphereFaceLoss(loss_s, loss_m)
        elif self.head_type == "softmax":
            self.head = LinearHead(head_input_dim, num_classes)
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        elif self.head_type == "triplet":
            self.bn = nn.BatchNorm1d(emb_dim)
            self.classifier = nn.Linear(head_input_dim, num_classes, bias=False)
            self.criterion_tri = TripletLoss(loss_m, mining_type)
            self.criterion_ce = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        else:
            raise ValueError(f"Unknown head type: {head_type}")

        self.to(device)
    
    def _infer_feature_dim(self):
        """Infer backbone output feature dimension."""
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.input_size, self.input_size)

            features = self.backbone(dummy)

            if isinstance(features, (list, tuple)):
                features = features[0]

            if len(features.shape) == 4:
                features = torch.flatten(features, 1)

            return features.shape[1]
        
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
        if hasattr(self.backbone, "blocks"):
            modules_to_unfreeze = list(self.backbone.blocks)
            
        # ConvNeXt / EfficientNet style
        elif hasattr(self.backbone, "stages"):
            for stage in self.backbone.stages:
                if hasattr(stage, "blocks"):
                    modules_to_unfreeze.extend(list(stage.blocks))
                else:
                    modules_to_unfreeze.append(stage)
                    
        # ResNet / MegaDescriptor style
        elif hasattr(self.backbone, "layer4"):
            modules_to_unfreeze = list(self.backbone.layer4)

        # Swin Transformer (timm style)
        elif hasattr(self.backbone, "layers"):
            for layer in self.backbone.layers:
                if hasattr(layer, "blocks"):
                    modules_to_unfreeze.extend(list(layer.blocks))
                else:
                    modules_to_unfreeze.append(layer)
        
        # EfficientNet (timm)
        elif hasattr(self.backbone, "blocks"):
            modules_to_unfreeze = list(self.backbone.blocks)
                    
        elif hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer4"):
            modules_to_unfreeze = list(self.backbone.encoder.layer4)

        elif hasattr(self.backbone, "backbone") and hasattr(self.backbone.backbone, "layer4"):
            modules_to_unfreeze = list(self.backbone.backbone.layer4)
        
        # Fallback
        else:
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Sequential):
                    modules_to_unfreeze.extend(list(module))

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
    
    def _extract_features(self, x):
        """Handles backbone feature extraction with optional forward_features."""
        if self.use_forward_features and hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
            # Handle ViT-style token -> spatial map
            if features.ndim == 3:  # [B, N, C]
                B, N, C = features.shape
                H = W = int(N ** 0.5)
                features = features[:, : H * W, :]
                features = features.permute(0, 2, 1).reshape(B, C, H, W)
        else:
            features = self.backbone(x)
            if isinstance(features, (tuple, list)):
                features = features[0]
        return features
    
    def get_embeddings(self, x):
        """Utility method to extract normalized embeddings for ReID evaluation."""
        features = self._extract_features(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            if self.use_gem:
                features = self.gem(features).flatten(1)
            else:
                features = features.mean(dim=(2, 3))
        # Pass through the learned projection neck; otherwise through a batch normalization layer
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
        features = self._extract_features(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            if self.use_gem:
                features = self.gem(features).flatten(1)
            else:
                features = features.mean(dim=(2, 3))
               
        # Pass through the learned projection neck; otherwise through a batch normalization layer
        embeddings = self.neck(features)
        if not self.use_projection:
            embeddings = self.bn(embeddings)

        # Triplet case (returns embeddings)
        if self.head_type == "triplet":
            feat_after_bn = self.bn(embeddings)   # Used for CE and Final Embedding
            if labels is not None:
                logits = self.classifier(feat_after_bn)
                loss_tri = self.criterion_tri(embeddings, labels)
                loss_ce = self.criterion_ce(logits, labels)
                # Combined Loss (1.0 * Triplet + 1.0 * CE)
                return loss_tri + loss_ce, logits
            # For inference, use normalized feat_after_bn
            return F.normalize(feat_after_bn, p=2, dim=1) # for Triplet loss we normalize here
        
        # Classification heads
        logits = self.head(embeddings)
        if labels is not None:
            if self.head_type in ["arcface", "cosface", "sphereface"]:
                loss, scaled_logits = self.criterion(logits, labels)
                return loss, scaled_logits
            else:
                return self.criterion(logits, labels), logits
        return logits