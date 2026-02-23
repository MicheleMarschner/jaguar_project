import torch, itertools
import numpy as np
from PIL import Image

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
        transform=None,
        overwrite_db=False,
    )

    print("[Info] Dataset size:", len(torch_ds))

    filepaths = [str(torch_ds._resolve_path(s[torch_ds.filepath_key])) for s in torch_ds.samples]
    labels    = np.asarray(torch_ds.labels)

    # Load model wrapper
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    split = "training"
    print("[Info] Loaded model:", model_wrapper.name)

    for p in model_wrapper.model.parameters():
        p.requires_grad_(False)

    torch_ds.transform = model_wrapper.model.transform
    x0 = torch_ds[0]


    ## get weights of frozen model

    # Load the pre-trained ViT model
    #! model, weights = get_vit_imagenet()

    # Load and preprocess the input image
    #! image = Image.open('docs/source/_static/cat_dog.jpg').convert('RGB')
    #! input_tensor = weights.transforms()(image).unsqueeze(0).to("cuda")

    # Store the generated heatmaps
    heatmaps = []

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


    # Experiment with different gamma values for Conv2d and Linear layers
    # Gamma is a hyperparameter in LRP that controls how much positive vs. negative
    # contributions are considered in the explanation
    for conv_gamma, lin_gamma in itertools.product([0.25], [1]):           # [0.1, 0.25, 100], [0, 0.01, 0.05, 0.1, 1]
        input_tensor.grad = None  # Reset gradients
        print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)
        
        # Define rules for the Conv2d and Linear layers using 'zennit'
        # LayerMapComposite maps specific layer types to specific LRP rule implementations
        zennit_comp = LayerMapComposite([
            (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
        ])
        
        # Register the composite rules with the model
        zennit_comp.register(model)
        
        # Forward pass with gradient tracking enabled
        y = model(input_tensor.requires_grad_())
        
        # Get the top 5 predictions
        _, top5_classes = torch.topk(y, 5, dim=1)
        top5_classes = top5_classes.squeeze(0).tolist()
        
        # Get the class labels
        labels = weights.meta["categories"]
        top5_labels = [labels[class_idx] for class_idx in top5_classes]
        
        # Print the top 5 predictions and their labels
        for i, class_idx in enumerate(top5_classes):
            print(f'Top {i+1} predicted class: {class_idx}, label: {top5_labels[i]}')
        
        # Backward pass for the highest probability class
        # This initiates the LRP computation through the network
        y[0, top5_classes[0]].backward()
        
        # Remove the registered composite to prevent interference in future iterations
        zennit_comp.remove()
        
        # Calculate the relevance by computing Gradient * Input
        # This is the final step of LRP to get the pixel-wise explanation
        heatmap = (input_tensor * input_tensor.grad).sum(1)
        
        # Normalize relevance between [-1, 1] for plotting
        heatmap = heatmap / abs(heatmap).max()
        
        # Store the normalized heatmap
        heatmaps.append(heatmap[0].detach().cpu().numpy())

    # Visualize all heatmaps in a grid (3×5) and save to a file
    # vmin and vmax control the color mapping range
    imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save('vit_heatmap.png')



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