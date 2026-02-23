import torch
import numpy as np

import zennit
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlus, EpsilonGammaBox, LayerMapComposite
import zennit.rules as z_rules
from lxt.efficient import monkey_patch, monkey_patch_zennit

import dino.vision_transformer as dino_vit
from jaguar.config import PATHS, DEVICE
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.models.foundation_models import FoundationModelWrapper


## load dino
if __name__ == "__main__":

    model_name = "DINOv2-Base"         # MegaDescriptor-L, DINOv2-Base, MiewID, ConvNeXt-V2
    dataset_name = "jaguar_stage0"
    base_root = PATHS.data_export

    fo_ds, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export,
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=False,
    )

    print("[Info] Dataset size:", len(torch_ds))

    filepaths = [str(torch_ds._resolve_path(s[torch_ds.filepath_key])) for s in torch_ds.samples]
    labels    = np.asarray(torch_ds.labels)

    # Load model wrapper
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    split = "training"
    print("[Info] Loaded model:", model_wrapper.name)

    emb_dir = PATHS.data / "embeddings"
    emb_path = emb_dir / f"embeddings_{model_wrapper.name}_{split}.npy"

    if emb_path.exists():
        embs = np.load(str(emb_path))
        print(f"[Info] Loaded embeddings from {emb_path} shape={embs.shape}")
    else:
        print(f"[Info] Computing embeddings -> {emb_path}")
        batch_size = 32
        all_embs = []
        for start in range(0, len(filepaths), batch_size):
            batch_paths = filepaths[start:start + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            batch_emb = model_wrapper.extract_embeddings(imgs)  # (B,D) numpy
            all_embs.append(batch_emb)

        embs = np.vstack(all_embs)
        np.save(str(emb_path), embs)
        print(f"[Info] Saved embeddings to {emb_path} shape={embs.shape}")

        assert embs.shape[0] == len(filepaths), "Embeddings must align with dataset order!"
        print("[Info] Embedding dim:", embs.shape[1])

# Patch the *DINO* ViT implementation module
monkey_patch(dino_vit, verbose=True)
monkey_patch_zennit(verbose=True)

# Basic composite
comp = LayerMapComposite([
    (torch.nn.Conv2d, z_rules.Gamma(0.25)),  # patch embed is conv in many ViTs
    (torch.nn.Linear, z_rules.Gamma(0.05)),  # MLP + qkv/proj are Linear
])

# Preprocess + forward
## apply transform
## get one image from ds

# DINO hub models often output embeddings, not logits
comp.register(model)
x = x.requires_grad_()
feat = model(x)            # usually [B, D] embedding

# choose a scalar target for relevance (examples)
target = feat[0, 0]        # single embedding dimension
# target = feat.norm()     # embedding norm (also scalar)

target.backward()

# LXT quickstart focuses on Input*Gradient formulation for AttnLRP
# so this "x * grad" heatmap is the expected baseline form
heatmap = (x.grad * x).sum(dim=1)[0].detach().cpu()

comp.remove()