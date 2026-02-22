import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ):
        super().__init__()
        self.device = device
        self.head_type = head_type.lower()

        # Load foundation backbone
        self.backbone_wrapper = FoundationModelWrapper(backbone_name, device=device)
        self.backbone = self.backbone_wrapper.model

        # Freeze backbone if needed
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Infer feature_dim
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 384, 384).to(device) # Compatibility for MegaDescriptor
            out = self.backbone(dummy)
            if isinstance(out, (tuple, list)): out = out[0]
            if out.ndim > 2: out = out.mean(dim=(2, 3))
            self.feature_dim = out.shape[1]

        # Always create a neck (EmbeddingHead)
        self.neck = EmbeddingHead(self.feature_dim, emb_dim)

        print(f"[JaguarID] Feature dim: {self.feature_dim}")

        # Select head + loss
        if self.head_type == "arcface":
            self.head = BaseMarginHead(emb_dim, num_classes)
            self.criterion = ArcFaceLoss()
        elif self.head_type == "cosface":
            self.head = BaseMarginHead(emb_dim, num_classes)
            self.criterion = CosFaceLoss()
        elif self.head_type == "sphereface":
            self.head = BaseMarginHead(emb_dim, num_classes)
            self.criterion = SphereFaceLoss()
        elif self.head_type == "softmax":
            self.head = LinearHead(emb_dim, num_classes)
            self.criterion = nn.CrossEntropyLoss()
        elif self.head_type == "triplet":
            self.head = nn.Identity() # Triplet just uses the neck
            self.criterion = TripletLoss()
        else:
            raise ValueError(f"Unknown head type: {head_type}")

        self.to(device)
    
    def get_embeddings(self, x):
        """Utility method to extract normalized embeddings for ReID evaluation."""
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.mean(dim=(2, 3))

        # Pass through the learned projection neck
        embeddings = self.neck(features)
        return F.normalize(embeddings, dim=1)

    def forward(self, x, labels=None):
        features = self.backbone(x)

        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = features.mean(dim=(2, 3))
            
        # Pass through neck
        embeddings = self.neck(features)

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
        PATHS.data_export,
        dataset_name="jaguar_stage0",
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
        backbone_name="MegaDescriptor-L",
        num_classes=num_classes,
        head_type="arcface",
        device=str(DEVICE),
        freeze_backbone=True  # Usually True if just evaluating foundation features
    )
    model.eval()
    
    # Setup DataLoader 
    torch_ds.transform = model.backbone_wrapper.transform 
    loader = DataLoader(
        torch_ds, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

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