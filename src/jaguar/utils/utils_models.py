"""
Utility functions to load foundation models for ReID.
These are placeholders; replace with actual model imports / checkpoints.
"""

import torch
import timm
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
        "S": "megadescriptor_s_384",
        "M": "megadescriptor_m_384",
        "L": "megadescriptor_l_384",
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
    Load DINO or DINOv2 ViT models via torch.hub.

    Args:
        model_name: "dino" or "dinov2"
        model_size: "small" or "base"
        patch_size: patch size (DINO: 8 or 16, DINOv2: 14)
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
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    model.eval()
    return model

# ----------------------------
# MegaDescriptor - DINOv2 for Wildlife 
# ----------------------------
def load_megadescriptor_dino_model():
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-DINOv2-518", pretrained=True)

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
# EVA-02 (placeholder)
# ----------------------------
def load_eva_02():
    print("Loading EVA-02 model...")
    # num_classes=0 removes the classification head, returning features/embeddings
    eva_model = timm.create_model('eva02_large_patch14_224.mim_in22k', pretrained=True, num_classes=0) 
    eva_model.eval() 
    return eva_model   