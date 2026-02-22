from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import torch
from functools import partial
from PIL import Image
import numpy as np
from pathlib import Path
import random
import os

from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.config import PATHS, DEVICE
from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.preprocessing import PROCESSORS
from jaguar.utils.utils_explainer import generate_similarity_cam

def save_npz(out_path: Path, **arrays):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print("Saved:", out_path)


def overlay_cam_on_image(img_pil: Image.Image, cam_2d: np.ndarray, out_path: Path):
    """
    img_pil: original RGB PIL
    cam_2d:  HxW float array
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam = cam_2d.astype(np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_pil)
    ax.imshow(cam, alpha=0.45)  # heat overlay
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def show_grid_from_torch_ds(torch_ds, n=16, cols=4):
    n = min(n, len(torch_ds))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(n):
        it = torch_ds[i]
        img = it["img"]

        # handle PIL vs tensor
        if isinstance(img, torch.Tensor):
            # assume CHW float tensor in [0,1] or normalized; just for viewing:
            x = img.detach().cpu()
            if x.ndim == 3 and x.shape[0] in (1, 3):
                x = x.permute(1, 2, 0).numpy()
            x = np.clip(x, 0, 1)
            img_vis = x
        else:
            # PIL
            img_vis = np.array(img)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_vis)
        plt.axis("off")
        plt.title(f'{it.get("id","")}', fontsize=8)

    plt.tight_layout()
    plt.show()


def get_positive_for_query(dataset, query_idx):
    """
    Returns: (pos_path, pos_label)
    """
    query_label = dataset.labels[query_idx]          # <- plain string label
    all_pos = dataset.idx_by_id.get(query_label, [])  # or dataset.idx_by_id
    candidates = [i for i in all_pos if i != query_idx]
    if not candidates:
        raise ValueError(f"No positive for label={query_label} at idx={query_idx}")

    pos_idx = random.choice(candidates)
    pos_path = dataset._resolve_path(dataset.samples[pos_idx][dataset.filepath_key])
    return str(pos_path), dataset.labels[pos_idx]


def get_negative_for_query(dataset, query_idx):
    """
    Returns: (neg_path, neg_label)
    """
    query_label = dataset.labels[query_idx]
    all_labels = list(dataset.idx_by_id.keys())
    neg_labels = [l for l in all_labels if l != query_label]
    if not neg_labels:
        raise ValueError("No negative labels available")

    neg_label = random.choice(neg_labels)
    neg_idx = random.choice(dataset.idx_by_id[neg_label])
    neg_path = dataset._resolve_path(dataset.samples[neg_idx][dataset.filepath_key])
    return str(neg_path), dataset.labels[neg_idx]

def run_baseline():
    processor_name = "random_bg"  # e.g. from YAML
    model_name = "EfficientNet-B4"
    out_dir = PATHS.data / "embeddings"
    bg_dir = PATHS.data / "backgrounds"   
    base_root = Path(PATHS.data_export)   

    processing_fn = partial(
        PROCESSORS[processor_name],
        base_root=base_root,
        bg_dir=str(bg_dir),
        key_for_seed="filename",  # or "filepath"
    )
    model_wrapper = FoundationModelWrapper(model_name, device=str(DEVICE))

    # get data from fifty One for that experiment
    fo_ds, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export,
        dataset_name="jaguar_stage0",
        processing_fn=processing_fn,
        overwrite_db=False,
    )

    idx = 4
    
    query_sample = torch_ds.samples[idx]

    print(query_sample)

    query_label = torch_ds.labels[idx]
    query_path = str(torch_ds._resolve_path(query_sample[torch_ds.filepath_key]))
    
    print(f"\n[Info] Query Jaguar ID: {query_label} | Path: {query_path}")

    # --- 3. Dynamically fetch Positive and Negative Pairs ---
    pos_path, pos_label = get_positive_for_query(torch_ds, idx)
    neg_path, neg_label = get_negative_for_query(torch_ds, idx)

    print(f"[Info] Positive Ref ID: {pos_label} | Path: {pos_path}")
    print(f"[Info] Negative Ref ID: {neg_label} | Path: {neg_path}")

    # --- 4. Load Images via PIL (Enforcing 3-channel RGB) ---
    query_img = Image.open(query_path).convert('RGB')
    pos_ref_img = Image.open(pos_path).convert('RGB')
    neg_ref_img = Image.open(neg_path).convert('RGB')

    # --- 5. Initialize Model Wrapper ---
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)

    # --- 6. Generate CAMs ---
    print("\n[Info] Generating Positive Similarity CAM...")
    _, vis_positive = generate_similarity_cam(
        wrapper=model_wrapper,
        query_img=query_img, 
        ref_img=pos_ref_img, 
        maximize=True   # "What makes them identical?"
    )
    
    print("[Info] Generating Negative Similarity CAM...")
    _, vis_negative = generate_similarity_cam(
        wrapper=model_wrapper,
        query_img=query_img, 
        ref_img=neg_ref_img, 
        maximize=False  # "What features uniquely distinguish them?"
    )

    # --- 7. Save to Disk ---
    # Include the labels in the filename for easy review!
    os.makedirs("cam_outputs", exist_ok=True)
    pos_filename = f"cam_outputs/cam_positive_Q{query_label}_R{pos_label}.jpg"
    neg_filename = f"cam_outputs/cam_negative_Q{query_label}_R{neg_label}.jpg"
    
    Image.fromarray(vis_positive).save(pos_filename)
    Image.fromarray(vis_negative).save(neg_filename)
    
    print(f"[Info] Saved to {pos_filename} and {neg_filename}")


    '''
    show_grid_from_torch_ds(torch_ds, n=16, cols=4)

    
    
    # extract embeddings
    emb = wrapper.extract_embeddings(pil_images)  # np (N, D)
    print("Embeddings shape:", emb.shape)

    wrapper.save_embeddings(emb)

    '''

