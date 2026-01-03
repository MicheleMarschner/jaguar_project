import torch
import math
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import DEVICE

# ---------------------------------------------------------
# 1. WRAPPER (Handles Input Arguments)
# ---------------------------------------------------------
class FacebookDinoWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # The grad-cam library passes the tensor 'x' directly.
        # We call forward_features to get the dictionary.
        out = self.model.forward_features(x)
        
        # Return ONLY the CLS token (the ReID vector).
        # This makes GradCAM happy because it sees a Tensor, not a Dict.
        return out["x_norm_clstoken"]
    

class MegaDescriptorWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x=None, pixel_values=None):
        # Resolve input (GradCAM passes 'x', we might use 'pixel_values')
        img_tensor = x if x is not None else pixel_values
        
        if img_tensor is None:
            raise ValueError("Model called without input tensor!")
            
        # timm's Swin implementation with num_classes=0 returns the 
        # Global Average Pooled embedding directly.
        # Shape: [Batch, 1536]
        return self.model(img_tensor)


# ---------------------------------------------------------
# 2. RESHAPE TRANSFORM (Fixes the Grid Size issue for visualization)
# ---------------------------------------------------------
def dinov2_reshape_transform(tensor, **kwargs):
    # DINOv2 output format: [Batch, Tokens, Dim]
    # Tokens = 1 CLS + N Patches.
    
    # 1. Calculate the grid size dynamically (e.g., 518px -> 37x37)
    num_patches = tensor.shape[1] - 1
    side = int(math.sqrt(num_patches))
    
    # 2. Remove CLS token (Index 0)
    # 3. Reshape Patches: [B, N, D] -> [B, H, W, D] -> [B, D, H, W]
    result = tensor[:, 1:, :].reshape(tensor.size(0), side, side, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def swin_reshape_transform(tensor, **kwargs):
    # Swin output at the 'norm' layer is usually: [Batch, H, W, Channels]
    # OR [Batch, L, Channels] (Flattened). We handle both.
    
    # Case A: Already H, W, C (Recent timm versions)
    if tensor.ndim == 4:
        # Permute from [B, H, W, C] -> [B, C, H, W]
        return tensor.permute(0, 3, 1, 2)

    # Case B: Flattened [B, L, C] (Older timm versions)
    elif tensor.ndim == 3:
        num_tokens = tensor.shape[1]
        side = int(math.sqrt(num_tokens))
        # Reshape [B, L, C] -> [B, H, W, C] -> [B, C, H, W]
        return tensor.reshape(tensor.size(0), side, side, tensor.size(2)).permute(0, 3, 1, 2)

    else:
        raise ValueError(f"Unknown tensor shape: {tensor.shape}")


# ---------------------------------------------------------
# 3. SIMILARITY TARGET (Defines "What makes a match?")
# ---------------------------------------------------------
class SimilarityTarget:
    def __init__(self, target_embedding):
        self.target_embedding = target_embedding.detach()

    def __call__(self, model_output):
        # We want to maximize the similarity (Dot Product)
        return torch.sum(model_output * self.target_embedding, dim=1)


GRAD_CAM_REGISTRY = {
    "DINOv2_Base": {
        "wrapper": FacebookDinoWrapper,
        "reshape_fn": dinov2_reshape_transform,
        # Function to extract layer from the raw model
        "layer_extractor": lambda m: m.blocks[-1].norm1, 
        "input_size": (518, 518)
    },
    "MegaDescriptor_L": {
        "wrapper": MegaDescriptorWrapper,
        "reshape_fn": swin_reshape_transform,
        # Function to extract layer from the raw model
        "layer_extractor": lambda m: m.norm,            # !!TODO: nicht? target_layer = m.layers[-1].blocks[-1].norm1
        "input_size": (384, 384)
    }
}

def compute_grad_cam(model_config, imgs, device=DEVICE):
    heatmaps = []

    name, raw_model, transform = model_config
    config = GRAD_CAM_REGISTRY[name]
    
    # 2. Wrap Model & Get Layer (Just-in-Time)
    wrapped_model = config["wrapper"](raw_model)
    target_layer = config["layer_extractor"](raw_model)
    reshape_fn = config["reshape_fn"]

    cam = GradCAM(
        model=wrapped_model, 
        target_layers=[target_layer], 
        reshape_transform=reshape_fn
    )

    print("Creating Heatmaps...")

    for img in tqdm(imgs):
        input_tensor = transform(img).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True) 

        # Get Target Embedding (Self-Similarity)
        with torch.no_grad():
            target_embedding = wrapped_model(input_tensor)

        # Run
        # Force gradients enabled just for this block
        with torch.enable_grad():
            targets = [SimilarityTarget(target_embedding)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Visualize & Save
        # Resize original image to 518x518 to match the CAM
        w, h = config["input_size"]
        img_resized = np.array(img.resize((w, h)))
        img_float = np.float32(img_resized) / 255.0

        heatmap = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        heatmaps.append(heatmap)
    
    return heatmaps
