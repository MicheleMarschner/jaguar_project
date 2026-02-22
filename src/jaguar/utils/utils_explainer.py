import torch
import numpy as np
import zennit
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlus, EpsilonGammaBox

## initialize like:
## wrapper = FoundationModelWrapper(name, device=DEVICE)
## try:
##      target_layers, reshape_transform = wrapper.get_grad_cam_config()
##  except NotImplementedError:
##      print(f"Skipping GradCAM for {name}")
##      continue
## cam = GradCAM(
##      model=wrapper.model, 
##      target_layers=target_layers, 
##      reshape_transform=reshape_transform
##  )



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


'''
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

'''

def lrp_zennit_input_relevance(
    model: torch.nn.Module,
    x: torch.Tensor,              # [1,3,H,W]
    target_idx: int,
    composite: str = "epsilon_plus",
    canonizers=None,
) -> np.ndarray:
    """
    Returns relevance heatmap [H,W] for input x using zennit composites.
    """
    from zennit.attribution import Gradient
    from zennit.composites import EpsilonPlus, EpsilonGammaBox

    if composite == "epsilon_plus":
        comp = EpsilonPlus()
    elif composite == "epsilon_gamma_box":
        comp = EpsilonGammaBox()
    else:
        raise ValueError(f"Unknown composite: {composite}")

    model.eval()
    x = x.requires_grad_(True)

    with comp.context(model, canonizers=canonizers):
        out = model(x)

        # handle common model outputs
        if isinstance(out, dict):
            out = out.get("logits", None) or out.get("preds", None) or next(iter(out.values()))
        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.ndim != 2:
            raise ValueError(f"Expected logits [B,C], got {tuple(out.shape)}. "
                             f"For embeddings, define a scalar target (see note below).")

        score = out[:, target_idx].sum()
        attr = Gradient(model)(x, score)  # relevance-like attribution at input

    heat = attr.sum(dim=1).squeeze(0)  # [H,W]
    heat = heat.detach().cpu().numpy()
    return heat