import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients


class SimilarityForward(torch.nn.Module):
    """
    Wrap a foundation model so it returns per-sample cosine similarity to one fixed reference embedding.
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
    """
    Define a Grad-CAM target based on cosine similarity to a fixed reference embedding.
    """
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
    Wrap a model so Grad-CAM sees the final embedding output even when embeddings are exposed via a custom method.
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
    Generate a Grad-CAM heatmap on a query image using embedding similarity to a reference image as the target.
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
    Compute Integrated Gradients saliency maps for similarity-to-reference scores.
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
    """
    Run similarity-based Integrated Gradients in batches and concatenate the resulting saliency maps.
    """
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