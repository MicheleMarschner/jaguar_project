'''
ArcFace CAM is best to inspect:
- “what makes it predict ID y?” (supervised evidence)
- whether the model is learning foreground vs background under the training loss

Similarity CAM (query→positive ref and query→hard negative)
'''

from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.v2 as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients

from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.config import DEVICE, PATHS, IMGNET_MEAN, IMGNET_STD
from jaguar.utils.utils_datasets import JaguarDataset


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


# ---------------------------------------------------------
# 1. SPECIALIZED DATASET (Inherits from your JaguarDataset)
# ---------------------------------------------------------

class MaskAwareJaguarDataset(JaguarDataset):
    def __init__(self, 
                 jaguar_model,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # 1. Access the inner wrapper configuration
        # We assume jaguar_model has 'backbone_wrapper'
        wrapper = jaguar_model.backbone_wrapper
        config = wrapper.get_config() # Or wrapper.MODEL_REGISTRY[wrapper.model_name]
        
        self.input_size = config.get("input_size", 256)
        self.mean = config.get("mean", [0.485, 0.456, 0.406])
        self.std = config.get("std", [0.229, 0.224, 0.225])
        
        # 2. Resize RGBA (Image + Mask together)
        self.resize_rgba = transforms.Resize(
            (self.input_size, self.input_size), 
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        
        # 3. Normalize RGB (After masking)
        self.to_tensor_norm = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.alpha_threshold = 128
        self.mask_fill_color = (128, 128, 128)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        img_path = self._resolve_path(s[self.filepath_key])
        
        # Load RGBA
        try:
            rgba = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None # Skip or handle error

        # Resize
        rgba = self.resize_rgba(rgba)
        
        # Split Mask
        r, g, b, a = rgba.split()
        fg_mask = np.array(a) > self.alpha_threshold
        
        # Create Variants
        rgb_np = np.array(Image.merge("RGB", (r, g, b)))
        
        img_orig = rgb_np.copy()
        
        img_bg_masked = rgb_np.copy()
        img_bg_masked[~fg_mask] = self.mask_fill_color # Keep Jaguar
        
        img_fg_masked = rgb_np.copy()
        img_fg_masked[fg_mask] = self.mask_fill_color # Keep Background
        
        # Normalize
        def process(arr):
            return self.to_tensor_norm(Image.fromarray(arr))

        return {
            "t_orig": process(img_orig),
            "t_bg_masked": process(img_bg_masked),
            "t_fg_masked": process(img_fg_masked),
            "label_idx": self.labels_idx[idx] if not self.is_test else -1,
            "filepath": str(img_path),
            "id": str(s.get("ground_truth", {}).get("label", "unknown"))
        }



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

    sal = attr.sum(dim=1)  # [B,H,W]
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


def find_module_name(model: torch.nn.Module, target_module: torch.nn.Module) -> str:
    for name, m in model.named_modules():
        if m is target_module:
            return name
    return "<unnamed>"


def save_vec(save_dir: Path, prefix: str, expl: str, pt: str, vec: np.ndarray) -> str:
    fname = f"{prefix}__{expl}__{pt}.npy"
    p = Path(save_dir) / fname
    np.save(p, np.asarray(vec, dtype=np.float32))
    return fname