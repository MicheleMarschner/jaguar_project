'''
ArcFace CAM is best to inspect:
- “what makes it predict ID y?” (supervised evidence)
- whether the model is learning foreground vs background under the training loss

Similarity CAM (query→positive ref and query→hard negative)
'''

import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients

from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.config import DEVICE, PATHS

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

class SimilarityForward(torch.nn.Module):
    """
    Returns scalar similarity per sample:
        s(x) = cosine(f(x), ref_emb)
    Output shape: [B]
    """
    def __init__(self, foundation_wrapper, ref_emb: torch.Tensor, maximize: bool = True):
        super().__init__()
        self.w = foundation_wrapper

        ref = ref_emb.detach()
        if ref.ndim == 2 and ref.shape[0] == 1:
            ref = ref.squeeze(0)  # [D]
        self.register_buffer("ref", F.normalize(ref, p=2, dim=0))
        self.maximize = maximize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.w.get_embeddings_tensor(x)  # [B,D]
        emb = F.normalize(emb, p=2, dim=1)
        sim = (emb * self.ref.unsqueeze(0)).sum(dim=1)  # [B]
        return sim if self.maximize else -sim


class CosineSimilarityTarget:
    def __init__(self, ref_embedding, maximize=True):
        ref = ref_embedding.detach()
        if ref.ndim == 2 and ref.shape[0] == 1:
            ref = ref.squeeze(0)
        self.ref_embedding = F.normalize(ref, p=2, dim=0)   # [D]
        self.maximize = maximize

    def __call__(self, model_output):
        emb = model_output
        if isinstance(emb, dict):
            emb = emb.get("embeddings") or next(iter(emb.values()))
        elif isinstance(emb, (tuple, list)):
            emb = emb[0]
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)  # [D]
        emb = F.normalize(emb, p=2, dim=0)
        sim = (emb * self.ref_embedding).sum()
        return sim if self.maximize else -sim


class EmbeddingForwardWrapper(torch.nn.Module):
    """
    Wraps a model to ensure pytorch-grad-cam's forward hooks intercept the 
    final embedding vector, even if the model uses a custom method like `get_embeddings`.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        if hasattr(self.base_model, "get_embeddings"):
            return self.base_model.get_embeddings(x)
        return self.base_model(x)


def generate_similarity_cam(wrapper, query_img: Image.Image, ref_img: Image.Image, maximize: bool = True):
    """
    Generates a GradCAM heatmap on the 'query_img' based on its similarity to 'ref_img'.
    Accepts your FoundationModelWrapper as the first argument.
    """
    device = wrapper.device
    model = wrapper.model
    
    # 1. Prepare tensors
    query_tensor = wrapper.preprocess(query_img).unsqueeze(0).to(device)
    ref_tensor = wrapper.preprocess(ref_img).unsqueeze(0).to(device)

    # 2. Extract Reference Embedding
    with torch.no_grad():
        ref_emb = wrapper.get_embeddings_tensor(ref_tensor)   # [1,D]

    # 3. Setup GradCAM using the Explainer Wrapper
    cam_model = EmbeddingForwardWrapper(model)
    target_layers, reshape_transform = wrapper.get_grad_cam_config()
    
    cam = GradCAM(
        model=cam_model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    # 4. Define target and generate CAM
    targets = [CosineSimilarityTarget(ref_emb, maximize=maximize)]
    grayscale_cam = cam(input_tensor=query_tensor, targets=targets)[0, :]

    # 5. Create Visual Overlay
    target_size = (query_tensor.shape[3], query_tensor.shape[2]) # (W, H)
    query_img_resized = query_img.resize(target_size, Image.Resampling.LANCZOS)
    query_img_rgb = np.array(query_img_resized).astype(np.float32) / 255.0
    
    visualization = show_cam_on_image(query_img_rgb, grayscale_cam, use_rgb=True)

    return grayscale_cam, visualization


def ig_saliency_similarity(
    x: torch.Tensor,                  # [B,3,H,W], preprocessed (can be CPU)
    explainer: IntegratedGradients,
    device: torch.device | str,
    steps: int = 32,
    internal_bs: int = 16,
) -> torch.Tensor:
    """
    IG for similarity scalar s(x)=cos(f(x), ref_emb).
    No class target is used because forward returns [B].
    Returns:
        saliency [B,H,W] on CPU
    """
    x = x.to(device).requires_grad_(True)
    baseline = torch.zeros_like(x)

    attr = explainer.attribute(
        inputs=x,
        baselines=baseline,
        target=None,
        n_steps=steps,
        internal_batch_size=internal_bs,
        method="riemann_trapezoid",
    )  # [B,3,H,W]

    sal = attr.abs().sum(dim=1)  # [B,H,W]
    return sal.detach().cpu()


def ig_saliency_batched_similarity(X, explainer, device, steps=32, internal_bs=32, batch_size=32):
    outs = []
    n = X.size(0)
    print(f"[IG] Start batched IG: N={n}, batch_size={batch_size}, steps={steps}, internal_bs={internal_bs}")
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        print(f"[IG] Batch {i//batch_size + 1}/{(n + batch_size - 1)//batch_size} | xb.shape={tuple(xb.shape)}")
        out = ig_saliency_similarity(
            xb, explainer=explainer, device=device, steps=steps, internal_bs=internal_bs
        )
        print(f"[IG] Done batch {i//batch_size + 1}, out.shape={tuple(out.shape)}")
        outs.append(out)
    return torch.cat(outs, dim=0)


def normalize_heatmap(h):
    """
    h: [H,W] tensor/ndarray
    returns [H,W] in [0,1]
    """
    if isinstance(h, torch.Tensor):
        h = h.detach().cpu().float().numpy()
    h = h.astype(np.float32)

    h_min = h.min()
    h_max = h.max()
    if h_max - h_min < 1e-12:
        return np.zeros_like(h, dtype=np.float32)
    return (h - h_min) / (h_max - h_min)


