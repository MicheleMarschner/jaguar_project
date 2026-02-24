## Source: https://github.com/rachtibat/LRP-eXplains-Transformers/blob/main/examples/vit_torch.py#L21

import torch, itertools
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
import inspect

import zennit
from zennit.attribution import Gradient
from zennit.composites import EpsilonPlus, EpsilonGammaBox, LayerMapComposite
from zennit.image import imgify
import zennit.rules as z_rules
from lxt.efficient import monkey_patch, monkey_patch_zennit

#import dino.vision_transformer as dino_vit
from jaguar.config import PATHS, DEVICE
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.models.foundation_models import FoundationModelWrapper


def _preprocess(img_path: str, device: str):
    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    return x


def _try_enable_return_attn(model):
    changed = []
    for i, blk in enumerate(model.blocks):
        a = blk.attn
        if hasattr(a, "return_attn"):
            a.return_attn = True
            changed.append((i, "return_attn"))
        if hasattr(a, "return_attention"):
            a.return_attention = True
            changed.append((i, "return_attention"))
    print("[Option1] enabled flags on blocks:", changed[:5], ("..." if len(changed) > 5 else ""))
    if not changed:
        print("[Option1] No return_attn/return_attention flags found.")
    return changed

@torch.no_grad()
def _extract_attn_if_present(model, x):
    """
    Runs forward once and tries to find attention stored on modules or returned by forward.
    Returns list of attn tensors [B,H,N,N] if found, else None.
    """
    out = model(x)

    # Case A: model returns a tuple/dict containing attentions
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        print("[Option1] model returned tuple/list, types:", [type(o) for o in out])
        # try to find something that looks like attention
        for o in out[1:]:
            if torch.is_tensor(o) and o.ndim == 4:
                print("[Option1] found attention in returned outputs:", o.shape)
                return [o]
        # sometimes attentions are a list
        for o in out[1:]:
            if isinstance(o, list) and o and torch.is_tensor(o[0]) and o[0].ndim == 4:
                print("[Option1] found attention list in returned outputs:", o[0].shape, "len", len(o))
                return o


if __name__ == "__main__":
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)

    dataset_name = "jaguar_stage0"
    base_root = PATHS.data_export / "init"

    fo_ds, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export / "init",
        dataset_name=dataset_name,
        processing_fn=None,
        transform=None,
        overwrite_db=False,
    )

    print("[Info] Dataset size:", len(torch_ds))

    filepaths = [str(torch_ds._resolve_path(s[torch_ds.filepath_key])) for s in torch_ds.samples]
    labels    = np.asarray(torch_ds.labels)

    x = _preprocess(filepaths[0], DEVICE)

    # -------- Option 1 --------
    _try_enable_return_attn(model)
    attn_list = _extract_attn_if_present(model, x)

    if attn_list is not None:
        print("[Option1] Great — we found attention tensors. You can now do grad-weighted rollout.")
        # If you want grad-weighted rollout, we'd need backward + attn gradients.
        # But since DINOv2 doesn't expose attn tensors in your printout, you’ll likely land in Option2.

    '''
    model_name = "DINOv2-Base"         # MegaDescriptor-L, DINOv2-Base, MiewID, ConvNeXt-V2
    dataset_name = "jaguar_stage0"
    base_root = PATHS.data_export / "init"

    fo_ds, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export / "init",
        dataset_name=dataset_name,
        processing_fn=None,
        transform=None,
        overwrite_db=False,
    )

    print("[Info] Dataset size:", len(torch_ds))

    filepaths = [str(torch_ds._resolve_path(s[torch_ds.filepath_key])) for s in torch_ds.samples]
    labels    = np.asarray(torch_ds.labels)

    # Load model wrapper
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    dino_model = model_wrapper.model
    split = "training"
    print("[Info] Loaded model:", model_wrapper.name)

    vit_mod = inspect.getmodule(dino_model.__class__)     # the module where VisionTransformer is defined
    print("DINO module:", vit_mod.__name__, vit_mod.__file__)

    # Patch the *DINO* ViT implementation module
    monkey_patch(vit_mod, verbose=True)
    monkey_patch_zennit(verbose=True)

    for p in dino_model.parameters():
        p.requires_grad_(False)

    torch_ds.transform = dino_model.transform
    x1 = torch_ds[0]
    x2 = torch_ds[1]

    ## get weights of frozen model

    # Load the pre-trained ViT model
    #! model, weights = get_vit_imagenet()

    # Load and preprocess the input image
    #! image = Image.open('docs/source/_static/cat_dog.jpg').convert('RGB')
    #! input_tensor = weights.transforms()(image).unsqueeze(0).to("cuda")

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

    # Store the generated heatmaps
    heatmaps = []

    # Experiment with different gamma values for Conv2d and Linear layers
    # Gamma is a hyperparameter in LRP that controls how much positive vs. negative
    # contributions are considered in the explanation
    for conv_gamma, lin_gamma in itertools.product([0.25], [1]):           # [0.1, 0.25, 100], [0, 0.01, 0.05, 0.1, 1]
        x1 = x1.detach().clone().requires_grad_(True)
        x2 = x2.detach().clone().requires_grad_(True)
        print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)
        
        # Define rules for the Conv2d and Linear layers using 'zennit'
        # LayerMapComposite maps specific layer types to specific LRP rule implementations
        zennit_comp = LayerMapComposite([
            (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
        ])
        
        # Register the composite rules with the model
        zennit_comp.register(dino_model)
        
        # Forward pass with gradient tracking enabled
        y1 = dino_model(x1)
        y2 = dino_model(x2)

        target = F.cosine_similarity(y1, y2).squeeze()  # scalar
        # target = y1[0, 0]
        target.backward()
        
        # Get the top 5 predictions
        # _, top5_classes = torch.topk(y, 5, dim=1)
        # top5_classes = top5_classes.squeeze(0).tolist()
        
        # Get the class labels
        # labels = weights.meta["categories"]
        # top5_labels = [labels[class_idx] for class_idx in top5_classes]
        
        # Print the top 5 predictions and their labels
        # for i, class_idx in enumerate(top5_classes):
        #     print(f'Top {i+1} predicted class: {class_idx}, label: {top5_labels[i]}')
        
        # Backward pass for the highest probability class
        # This initiates the LRP computation through the network
        #y[0, top5_classes[0]].backward()
        
        # Remove the registered composite to prevent interference in future iterations
        zennit_comp.remove()
        
        # Calculate the relevance by computing Gradient * Input
        # This is the final step of LRP to get the pixel-wise explanation
        heatmap = (x1 * x1.grad).sum(1)
        
        # Normalize relevance between [-1, 1] for plotting
        heatmap = heatmap / (heatmap.abs().max() + 1e-12)
        
        # Store the normalized heatmap
        heatmaps.append(heatmap[0].detach().cpu().numpy())

    # Visualize all heatmaps in a grid (3×5) and save to a file
    # vmin and vmax control the color mapping range
    imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save('dino_heatmap.png')


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

    '''