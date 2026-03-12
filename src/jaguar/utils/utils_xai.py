'''
ArcFace CAM is best to inspect:
- “what makes it predict ID y?” (supervised evidence)
- whether the model is learning foreground vs background under the training loss

Similarity CAM (query→positive ref and query→hard negative)
'''

from pathlib import Path
from typing import Sequence
from jaguar.utils.utils import ensure_dir, save_npy
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
import pandas as pd

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients


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


def manual_gradcam_class(model, target_layer, x, class_idx, reshape_transform=None):
    acts = []
    grads = []

    def fwd_hook(m, inp, out):
        acts.append(out)

    def bwd_hook(m, gin, gout):
        grads.append(gout[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        x = x.clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        logits = model(x)              # [1, C]
        score = logits[0, class_idx]   # scalar
        score.backward()

        A = acts[0]
        dA = grads[0]

        # Transformer-style token outputs -> convert to spatial feature maps
        if A.ndim == 3:
            if reshape_transform is None:
                raise ValueError(
                    f"Got 3D activations {tuple(A.shape)} but no reshape_transform was provided."
                )
            A = reshape_transform(A)
            dA = reshape_transform(dA)

        # CNN-style outputs should already be [B, C, H, W]
        if A.ndim != 4 or dA.ndim != 4:
            raise ValueError(
                f"Grad-CAM expects 4D tensors after optional reshape, got "
                f"A={tuple(A.shape)}, dA={tuple(dA.shape)}"
            )

        w = dA.mean(dim=(2, 3), keepdim=True)      # [1, C, 1, 1]
        cam = (w * A).sum(dim=1, keepdim=True)     # [1, 1, H, W]
        cam = F.relu(cam)

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-12)

        return cam.detach().cpu().numpy()

    finally:
        h1.remove()
        h2.remove()


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



# ============================================================
# Deterministic query selection (curated val subset)
# ============================================================

def get_curated_indices(split_df: pd.DataFrame, splits: Sequence[str]) -> np.ndarray:
    """Return global emb_row ids for curated samples in the requested splits."""
    df = split_df[
        split_df["split_final"].isin(list(splits))
        & split_df["keep_curated"].fillna(False).astype(bool)
    ]
    return df["emb_row"].astype(np.int64).to_numpy()


def sample_indices(indices: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Deterministically sample up to n_samples indices without replacement."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)

    if len(indices) == 0:
        return indices

    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)
    return np.sort(chosen)


def get_val_query_indices(
    split_df: pd.DataFrame,
    out_root: Path,
    n_samples: int | None,
    seed: int,
) -> np.ndarray:
    """
    Cache the exact global emb_row query subset so runs can be compared on the
    same validation images across repeated runs.
    """
    n_tag = "full" if n_samples is None else str(n_samples)
    idx_path = out_root / f"xai_val_idx_n{n_tag}.npy"

    if idx_path.exists():
        return np.load(idx_path)

    val_pool = get_curated_indices(split_df, splits=["val"])
    val_chosen = sample_indices(val_pool, n_samples=n_samples, seed=seed)

    ensure_dir(idx_path.parent)
    save_npy(idx_path, val_chosen)
    return val_chosen



def resolve_n_samples(n_samples: int | str | None) -> int | None:
    """
    Resolve configured sample count.
    Returns None for full-split usage.
    """
    if n_samples is None:
        return None

    if isinstance(n_samples, str):
        value = n_samples.strip().lower()
        if value in {"full", "all"}:
            return None
        return int(value)

    return int(n_samples)


def format_n_samples_tag(n_samples: int | str | None) -> str:
    """
    Stable tag for run/file names.
    """
    resolved = resolve_n_samples(n_samples)
    return "full" if resolved is None else str(resolved)
