from typing import Dict, Any, List
import numpy as np
import torch
from tqdm import tqdm
from captum.attr import IntegratedGradients
from pytorch_grad_cam import GradCAM


class ClassLogitForward(torch.nn.Module):
    """
    Forward wrapper that returns the logit of a fixed target class.
    """
    def __init__(self, model, target_class: int):
        super().__init__()
        self.model = model
        self.target_class = int(target_class)
        self.device = model.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)  # [B,C]
        return logits[:, self.target_class]


def ig_saliency_batched_class(
    x_batch: torch.Tensor,
    explainer: IntegratedGradients,
    device: torch.device,
    steps: int,
    internal_bs: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Returns [B,H,W] signed saliency maps for class attribution.
    """
    outs: List[torch.Tensor] = []

    for start in range(0, len(x_batch), batch_size):
        xb = x_batch[start:start + batch_size].to(device)

        attr = explainer.attribute(
            xb,
            baselines=torch.zeros_like(xb),
            n_steps=steps,
            internal_batch_size=internal_bs,
        )  # [B,3,H,W]

        sal = attr.sum(dim=1).detach().cpu()  # [B,H,W]
        outs.append(sal)

    return torch.cat(outs, dim=0)


def compute_saliency_ig_class(
    resolve_sample,
    model,
    artifact: dict,
    cfg,
) -> Dict[str, Any]:
    """
    Compute IG saliency for gold-class attribution.

    Expected artifact input:
    - sample_indices: tensor [N]
    Optionally:
    - group / meta fields for bookkeeping
    """
    sample_indices = artifact["sample_indices"].cpu().numpy().astype(np.int64)

    saliency_out: List[torch.Tensor] = [None] * len(sample_indices)
    gold_idx_out: List[int] = [None] * len(sample_indices)
    pred_out: List[int] = [None] * len(sample_indices)
    correct_out: List[bool] = [None] * len(sample_indices)

    model.eval()

    for i, sample_idx in enumerate(tqdm(sample_indices, desc="IG class saliency")):
        ds, local_idx, _ = resolve_sample(int(sample_idx))
        sample = ds[local_idx]

        x = sample["img"]                    # [3,H,W]
        gold_idx = int(sample["label_idx"])

        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(model.device))
            pred = int(logits.argmax(dim=1).item())

        class_model = ClassLogitForward(model, target_class=gold_idx).to(model.device).eval()
        ig = IntegratedGradients(class_model)

        sal = ig_saliency_batched_class(
            x_batch=torch.stack([x], dim=0),
            explainer=ig,
            device=model.device,
            steps=cfg.ig_steps,
            internal_bs=cfg.ig_internal_bs,
            batch_size=1,
        )[0]

        saliency_out[i] = sal
        gold_idx_out[i] = gold_idx
        pred_out[i] = pred
        correct_out[i] = (pred == gold_idx)

    return {
        "meta": {
            "explainer": "IG",
            "target": "gold_class_logit",
            "group": artifact.get("meta", {}).get("group", "all"),
            "ig_steps": cfg.ig_steps,
            "ig_internal_bs": cfg.ig_internal_bs,
            "ig_batch_size": cfg.ig_batch_size,
        },
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "gold_idx": torch.tensor(gold_idx_out, dtype=torch.long),
        "pred_orig": torch.tensor(pred_out, dtype=torch.long),
        "is_correct_orig": torch.tensor(correct_out, dtype=torch.bool),
        "saliency": torch.stack(saliency_out, dim=0),   # [N,H,W]
    }


class ClassTarget:
    """
    GradCAM target for one class logit.
    """
    def __init__(self, class_idx: int):
        self.class_idx = int(class_idx)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[:, self.class_idx]


def get_class_gradcam_config(model):
    """
    Reuse the backbone-specific GradCAM config, but apply it to the full classifier.
    """
    wrapper = model.backbone_wrapper
    grad_cam_cfg = wrapper.registry_entry["grad_cam"]
    layer_getter = grad_cam_cfg["layer_getter"]
    reshape_transform = grad_cam_cfg["reshape_transform"]
    target_layer = layer_getter(model.backbone)
    return target_layer, reshape_transform


def compute_saliency_gradcam_class(
    resolve_sample,
    model,
    artifact: dict,
    cfg,
) -> Dict[str, Any]:
    """
    Compute GradCAM saliency for gold-class attribution.

    Expected artifact input:
    - sample_indices: tensor [N]
    - meta.group optional
    """
    sample_indices = artifact["sample_indices"].cpu().numpy().astype(np.int64)

    saliency_out: List[torch.Tensor] = [None] * len(sample_indices)
    gold_idx_out: List[int] = [None] * len(sample_indices)
    pred_out: List[int] = [None] * len(sample_indices)
    correct_out: List[bool] = [None] * len(sample_indices)

    model.eval()

    target_layer, reshape_transform = get_class_gradcam_config(model)

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform,
    )

    for i, sample_idx in enumerate(tqdm(sample_indices, desc="GradCAM class saliency")):
        ds, local_idx, _ = resolve_sample(int(sample_idx))
        sample = ds[local_idx]

        x = sample["img"].unsqueeze(0).to(model.device)   # [1,3,H,W]
        gold_idx = int(sample["label_idx"])

        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())

        targets = [ClassTarget(gold_idx)]
        grayscale_cam = cam(input_tensor=x, targets=targets)  # [1,h,w]
        cam_t = torch.as_tensor(grayscale_cam[0], dtype=torch.float32)

        H, W = x.shape[-2], x.shape[-1]
        if cam_t.shape != (H, W):
            cam_t = torch.nn.functional.interpolate(
                cam_t[None, None, ...],
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        saliency_out[i] = cam_t.cpu()
        gold_idx_out[i] = gold_idx
        pred_out[i] = pred
        correct_out[i] = (pred == gold_idx)

    return {
        "meta": {
            "explainer": "GradCAM",
            "target": "gold_class_logit",
            "group": artifact.get("meta", {}).get("group", "all"),
            "target_layer_type": target_layer.__class__.__name__,
            "reshape_transform": None if reshape_transform is None else getattr(
                reshape_transform, "__name__", str(reshape_transform)
            ),
        },
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "gold_idx": torch.tensor(gold_idx_out, dtype=torch.long),
        "pred_orig": torch.tensor(pred_out, dtype=torch.long),
        "is_correct_orig": torch.tensor(correct_out, dtype=torch.bool),
        "saliency": torch.stack(saliency_out, dim=0),   # [N,H,W]
    }