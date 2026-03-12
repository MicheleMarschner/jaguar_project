from pathlib import Path

from jaguar.config import DATA_STORE, DEVICE, PATHS
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.utils.utils import ensure_dir, resolve_path, save_npy
from jaguar.utils.utils_datasets import PreprocessedDataset
import torch
import numpy as np
import timm
import functools
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModel
from torchvision.models import (
    convnext_large, ConvNeXt_Large_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    swin_b, Swin_B_Weights
)

# ----------------------------
# MegaDescriptor
# ----------------------------
def load_megadescriptor_model(size="L"):
    """
    Load MegaDescriptor-L/M/S pre-trained model from timm.
    Args:
        size (str): One of "S", "M", or "L" for model size.
    """
    model_name_map = {
        "S": "hf-hub:BVRA/MegaDescriptor-S-384",
        "M": "hf-hub:BVRA/MegaDescriptor-M-384",
        "L": "hf-hub:BVRA/MegaDescriptor-L-384",
    }
    
    if size not in model_name_map:
        raise ValueError(f"Invalid size {size}. Choose from S, M, L.")
    
    print(f"Loading MegaDescriptor-{size}-384 model...")
    # num_classes=0 removes the classification head, returning features/embeddings
    model = timm.create_model(model_name_map[size], pretrained=True, num_classes=0) 
    model.eval()
    return model

# ----------------------------
# DINO
# ----------------------------
def load_dino_model(model_name: str, model_size: str, patch_size: int, pretrained: bool = True, with_register_tokens: bool = False):
    """
    Load DINO, DINOv2 or DINOv3 ViT models via torch.hub.

    Args:
        model_name: "dino", "dinov2 or dinov3"
        model_size: "small", "base" or "large"
        patch_size: patch size (DINO: 8 or 16, DINOv2: 14, DINOv3: 16)
        pretrained: load pretrained weights (default True)
        with_register_tokens: only for DINOv2, whether to include register tokens
    """
    if model_name.lower() == "dino":
        assert patch_size in (8, 16), "DINO only supports patch size 8 or 16"
        print(f"Loading DINO ViT-{model_size.capitalize()} patch{patch_size}")
        model = torch.hub.load(
            'facebookresearch/dino:main',
            f'dino_vit{model_size[0]}{patch_size}',
            pretrained=pretrained,
        )
    elif model_name.lower() == "dinov2":
        assert patch_size == 14, "DINOv2 only supports patch size 14"
        repo_model = f'dinov2_vit{model_size[0]}{patch_size}_reg' if with_register_tokens else f'dinov2_vit{model_size[0]}{patch_size}'
        print(f"Loading DINOv2 ViT-{model_size.capitalize()} patch{patch_size} {'with' if with_register_tokens else 'without'} register tokens")
        model = torch.hub.load(
            'facebookresearch/dinov2',
            repo_model,
            trust_repo=True,
            pretrained=pretrained,
        )
    elif model_name.lower() == "dinov3":
        assert patch_size == 16, "DINOv3 only supports patch size 16"
        assert model_size.lower() in ("small", "base", "large"), "DINOv3 supports small, base, or large"
        size_to_entry = {
            "small": "dinov3_vits16",
            "base":  "dinov3_vitb16",
            "large": "dinov3_vitl16",
        }
        repo_model = size_to_entry[model_size.lower()]

        # DINOv3 does not use the DINOv2 `_reg` naming style
        if with_register_tokens:
            print("Warning: with_register_tokens is ignored for DINOv3.")
        print(f"Loading DINOv3 ViT-{model_size.capitalize()} patch{patch_size}")
        model = torch.hub.load(
            "facebookresearch/dinov3",
            repo_model,
            trust_repo=True,
            pretrained=pretrained,  # if this fails in your setup, use explicit weights=... instead
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    model.eval()
    return model

# ----------------------------
# MegaDescriptor - DINOv2 for Wildlife 
# ----------------------------
def load_megadescriptor_dino_model():
    print("[Info] Patching torch.load to allow MegaDescriptor weights...")

    # Store the original torch.load
    original_load = torch.load
    # Create a wrapper that forces weights_only=False
    @functools.wraps(original_load)
    def safe_load(*args, **kwargs):
        if 'weights_only' in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    # Apply the patch
    torch.load = safe_load
    try:
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-DINOv2-518", pretrained=True)
    finally:
        torch.load = original_load
    model.eval()
    return model

# ----------------------------
# ConvNeXt-V2
# ----------------------------
def load_convnext_v2(size="large"):
    print(f"Loading ConvNeXt-V2 {size} model...")
    model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)
    model.eval()
    return model

# ----------------------------
# EfficientNet-B4
# ----------------------------
def load_efficientnet_b4():
    print("Loading EfficientNet-B4 model...")
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    model.eval()
    return model

# ----------------------------
# Swin Transformer
# ----------------------------
def load_swin_transformer(size="base"):
    print(f"Loading Swin-Transformer {size} model...")
    model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
    model.eval()
    return model

# ----------------------------
# EVA-02 
# ----------------------------
def load_eva_02():
    print("Loading EVA-02 model...")
    # num_classes=0 removes the classification head, returning features/embeddings
    # eva_model = timm.create_model('eva02_large_patch14_224.mim_in22k', pretrained=True, num_classes=0) 
    eva_model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True, num_classes=0) 
    eva_model.eval() 
    return eva_model   

# ----------------------------
# MIEWv3 ID 
# ----------------------------
def load_miewid():
    # Force low_cpu_mem_usage=False to avoid the Meta device/AttributeError bug
    miew_model = AutoModel.from_pretrained(
        "conservationxlabs/miewid-msv3", 
        trust_remote_code=True,
        low_cpu_mem_usage=False 
    )
    miew_model.eval()
    return miew_model


# --- Helper functions for GradCAM on Transformers ---
def vit_reshape_transform(tensor):
    """
    Reshapes ViT (EVA-02, DINO) tokens [B, N, C] -> [B, C, H, W]
    """
    n_tokens = tensor.shape[1]
    
    # default: assume 1 CLS token
    start_index = 1
    spatial_tokens = n_tokens - 1
    
    # Check for DINOv2 with registers (usually 4 registers + 1 CLS = 5)
    # If the standard math (tokens-1) isn't a perfect square, try (tokens-5)
    if int(np.sqrt(spatial_tokens)) ** 2 != spatial_tokens:
        if int(np.sqrt(n_tokens - 5)) ** 2 == (n_tokens - 5):
            start_index = 5
    
    result = tensor[:, start_index:, :] 
    
    # Calculate grid size dynamically
    target_len = result.shape[1]
    grid_size = int(np.sqrt(target_len))
    
    # [B, N, C] -> [B, C, N] -> [B, C, H, W]
    result = result.transpose(1, 2)
    result = result.reshape(tensor.size(0), result.size(1), grid_size, grid_size)
    return result


def swin_reshape_transform(tensor):
    # Swin output is usually already spatial but channel-last in some implementations
    # Check your specific Swin implementation. Usually: [B, H, W, C] -> [B, C, H, W]
    if len(tensor.shape) == 4:
        return tensor.permute(0, 3, 1, 2)
    return tensor


def load_or_extract_embeddings(model_wrapper, torch_ds, split="training", batch_size=32, num_workers=4):
    folder = resolve_path("embeddings", DATA_STORE)
    ensure_dir(folder)

    filename = f"embeddings_{model_wrapper.name}_{split}.npy"
    path = folder / filename

    if path.exists():
        emb = np.load(path)
        print(f"[Info] Loaded embeddings from {path}, shape={emb.shape}")
        return emb

    print(f"[Info] Embeddings not found at {path}. Extracting...")

    # Wrap the dataset so preprocessing happens on the fly
    wrapped_ds = PreprocessedDataset(torch_ds, model_wrapper.preprocess)

    # Create DataLoader from the wrapped dataset
    dataloader = DataLoader(
        wrapped_ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True, 
        shuffle=False
    )

    all_embeddings = []
    # Process batches
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        imgs = batch["img"]
        # Extract embeddings (usually moves data to GPU internally)
        batch_emb = model_wrapper.extract_embeddings(imgs)  
        # Ensure it's a numpy array before storing to save RAM
        if torch.is_tensor(batch_emb):
            batch_emb = batch_emb.cpu().numpy()
        all_embeddings.append(batch_emb)

    emb = np.concatenate(all_embeddings, axis=0)
    save_npy(path, emb)
    print(f"[Info] Saved embeddings to {path}, shape={emb.shape}")
    return emb


def load_or_extract_jaguarid_embeddings(
    model,
    torch_ds,
    split: str = "training",
    batch_size: int = 32,
    num_workers: int = 0,
    use_tta: bool = False,
    cache_prefix: str | None = None,
    folder=None,
):
    """
    Returns embeddings as np.ndarray [N, D].
    Loads from disk if available; otherwise extracts with JaguarIDModel.get_embeddings() and saves.
    """
    if folder is None:
        folder = resolve_path("embeddings", DATA_STORE)
    folder.mkdir(parents=True, exist_ok=True)

    model_name = model.backbone_wrapper.name
    head_type = getattr(model, "head_type", "unknown")
    tta_tag = "tta" if use_tta else "no_tta"

    prefix = f"{cache_prefix}_" if cache_prefix else ""
    filename = f"{prefix}embeddings_{model_name}_{head_type}_{split}_{tta_tag}.npy"
    path = folder / filename

    if path.exists():
        emb = np.load(path)
        print(f"[Info] Loaded embeddings from {path}, shape={emb.shape}")
        return emb

    print(f"[Info] Embeddings not found at {path}. Extracting...")

    loader = DataLoader(
        torch_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(str(model.device).startswith("cuda") or getattr(model.device, "type", "") == "cuda"),
    )

    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract embeddings [{split}]"):
            imgs = batch["img"].to(model.device)

            feats = model.get_embeddings(imgs)

            if use_tta:
                flipped = torch.flip(imgs, dims=[3])
                feats_flip = model.get_embeddings(flipped)
                feats = (feats + feats_flip) / 2.0
                feats = torch.nn.functional.normalize(feats, dim=1)

            all_embeddings.append(feats.cpu().numpy())

    emb = np.concatenate(all_embeddings, axis=0)
    save_npy(path, emb)
    print(f"[Info] Saved embeddings to {path}")
    return emb


def load_checkpoint_into_model(model: JaguarIDModel, checkpoint_path: Path) -> None:
    """Load checkpoint weights into a JaguarIDModel."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

