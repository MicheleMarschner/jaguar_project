import json

from jaguar.utils.utils import ensure_dir, save_parquet
from jaguar.utils.utils_xai import MaskAwareJaguarDataset
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import cv2
from collections import defaultdict
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from jaguar.config import IMGNET_MEAN, IMGNET_STD, PATHS, DEVICE, SEED
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.models.jaguarid_models import JaguarIDModel



def compute_logit_sensitivity(model, dataloader, device):
    results = []
    model.eval()
    
    print("--- Running Classification Sensitivity (Logit Drop) ---")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: continue
            
            orig = batch["t_orig"].to(device)
            bg_masked = batch["t_bg_masked"].to(device)
            fg_masked = batch["t_fg_masked"].to(device)
            targets = batch["label_idx"].to(device)
            
            # Forward pass on JaguarIDModel returns LOGITS when labels=None
            logits_orig = model(orig)
            logits_bg_masked = model(bg_masked)
            logits_fg_masked = model(fg_masked)

            # Compute log-probabilities (stable + comparable)
            logp_orig = F.log_softmax(logits_orig, dim=1)
            logp_bg   = F.log_softmax(logits_bg_masked, dim=1)
            logp_fg   = F.log_softmax(logits_fg_masked, dim=1)
            
            for i in range(len(targets)):
                c = int(targets[i].item())
                s0   = float(logits_orig[i, c].item())
                s_bg = float(logits_bg_masked[i, c].item())
                s_fg = float(logits_fg_masked[i, c].item())

                # Use log-prob as primary score
                s0_lp   = float(logp_orig[i, c].item())
                s_bg_lp = float(logp_bg[i, c].item())
                s_fg_lp = float(logp_fg[i, c].item())
                            
                results.append({
                    "id": batch["id"][i],
                    "filepath": Path(batch["filepath"][i]).name,
                    "score_orig_logit": s0,
                    "score_jaguar_only_logit": s_bg,
                    "score_bg_only_logit": s_fg,
                    "score_orig_logp": s0_lp,
                    "score_jaguar_only_logp": s_bg_lp,
                    "score_bg_only_logp": s_fg_lp,
                    "drop_bg": s0 - s_bg,
                    "drop_fg": s0 - s_fg,
                    "is_spurious": s_fg > s_bg,
                    "spurious_margin": s_fg - s_bg
                })
    return pd.DataFrame(results)


def compute_embedding_stability(model, dataloader, device):
    results = []
    model.eval()
    
    print("--- Running Embedding Stability Analysis ---")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: continue

            orig = batch["t_orig"].to(device)
            bg_masked = batch["t_bg_masked"].to(device)
            fg_masked = batch["t_fg_masked"].to(device)
            
            # Use get_embeddings() for ReID vectors
            emb_orig = model.get_embeddings(orig)
            emb_bg   = model.get_embeddings(bg_masked)
            emb_fg   = model.get_embeddings(fg_masked)
            
            # Cosine Similarity (Vectors are already normalized in get_embeddings)
            sim_bg = (emb_orig * emb_bg).sum(dim=1)
            sim_fg = (emb_orig * emb_fg).sum(dim=1)
            
            for i in range(len(batch["id"])):
                results.append({
                    "id": batch["id"][i],
                    "filepath": Path(batch["filepath"][i]).name,
                    "stability_jaguar_only": sim_bg[i].item(),
                    "stability_bg_only": sim_fg[i].item(),
                    # Spurious if vector stays closer to original when ONLY BG is present
                    "is_spurious": sim_fg[i].item() > sim_bg[i].item(),
                    "spurious_margin": float(sim_fg[i].item() - sim_bg[i].item())
                })
    return pd.DataFrame(results)


def select_random_datasubset_balanced(
    torch_ds,
    n: int,
    seed: int = 51,
    k_per_id: int = 5,
) -> np.ndarray:
    """
    Deterministically sample up to k_per_id per identity until reaching n.
    Requires torch_ds.labels.
    """
    labels = np.asarray(torch_ds.labels)
    rng = np.random.default_rng(seed)

    id_to_indices = defaultdict(list)
    for idx, y in enumerate(labels):
        id_to_indices[int(y)].append(idx)

    # deterministic shuffle within each identity
    for y in id_to_indices:
        rng.shuffle(id_to_indices[y])

    # deterministic identity order
    ids = sorted(id_to_indices.keys())
    rng.shuffle(ids)

    selected = []
    # round-robin to keep it balanced
    round_i = 0
    while len(selected) < min(n, len(labels)):
        made_progress = False
        for y in ids:
            src = id_to_indices[y]
            take_pos = round_i
            if take_pos < min(k_per_id, len(src)):
                selected.append(src[take_pos])
                made_progress = True
                if len(selected) >= n:
                    break
        if not made_progress:
            break
        round_i += 1

    return np.asarray(selected, dtype=np.int64)


def generate_visuals_logits(model, dataloader, df_results, output_dir, device):
    """
    Generates GradCAM overlays for spurious cases.
    """
    output_dir = Path(output_dir) / "heatmaps"
    ensure_dir(output_dir)
    
    # 1. Resolve Target Layer
    # Access the registry via the inner wrapper
    wrapper = model.backbone_wrapper
    config = wrapper.get_config()
    layer_getter = config["grad_cam"]["layer_getter"]
    
    # Apply getter to the BACKBONE, not the full JaguarIDModel
    target_layer = layer_getter(model.backbone) 
    
    # Setup CAM: construct CAM on the full model, but target the inner layer
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Pick top 5 spurious examples
    spurious = df_results[df_results["is_spurious"] == True].head(5)
    if spurious.empty:
        print("No spurious cases found! Showing random examples.")
        spurious = df_results.head(5)
        
    target_files = set(spurious["filepath"].values)
    
    print(f"Generating visuals for {len(target_files)} images...")
    
    for batch in dataloader:
        for i in range(len(batch["filepath"])):
            fname = Path(batch["filepath"][i]).name
            if fname in target_files:
                
                # Prepare Inputs
                inputs = {
                    "Orig": batch["t_orig"][i].unsqueeze(0).to(device),
                    "Jaguar": batch["t_bg_masked"][i].unsqueeze(0).to(device),
                    "BG": batch["t_fg_masked"][i].unsqueeze(0).to(device)
                }
                
                imgs_vis = []
                
                for title, tensor in inputs.items():
                    # Generate CAM
                    target_cls = batch["label_idx"][i].item()
                    targets = [ClassifierOutputTarget(target_cls)]
                    
                    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0, :]
                    
                    # Un-norm for display
                    img_disp = tensor.squeeze().cpu().permute(1,2,0).numpy()
                    mean = np.array(IMGNET_MEAN).reshape(1,1,3)
                    std  = np.array(IMGNET_STD).reshape(1,1,3)
                    img_disp = img_disp * std + mean
                    img_disp = np.clip(img_disp, 0, 1)
                    
                    vis = show_cam_on_image(img_disp, grayscale_cam, use_rgb=True)
                    imgs_vis.append(vis)
                
                combined = np.hstack(imgs_vis)
                cv2.imwrite(str(output_dir / f"cam_logits__{Path(fname).stem}.png"), combined)


if __name__ == "__main__":
    BATCH_SIZE = 16
    backbone_name = "MegaDescriptor-L"
    head_type = "arcface"
    n_samples = 10
    dataset_name = "jaguar_init"
    manifest_dir = PATHS.data_export / "init"
    checkpoint_path = ""

    save_path = PATHS.runs / "xai/background_sensitivity"
    ensure_dir(save_path)
    results_path = PATHS.results / "xai/background_sensitivity"
    ensure_dir(results_path)
    
    # Load Sample List
    _, temp_ds = load_jaguar_from_FO_export(manifest_dir, dataset_name=dataset_name)
    torch_idx = select_random_datasubset_balanced(temp_ds, n=n_samples, seed=SEED, k_per_id=5)
    samples_list = [temp_ds.samples[i] for i in torch_idx]

    num_classes = int(len(np.unique(np.asarray(temp_ds.labels))))

    # Load Model (Pre-trained/Fine-tuned)
    model = JaguarIDModel(
        backbone_name=backbone_name, 
        num_classes=num_classes,
        head_type=head_type,
        device=str(DEVICE)
    )
    # model.load_state_dict(torch.load("path/to/weights.pt"))       ## TODO!
    model = model.to(DEVICE).eval()

    # Create Final Dataset
    xai_ds = MaskAwareJaguarDataset(
        jaguar_model=model,
        base_root=PATHS.data_train,
        samples_list=samples_list,
        is_test=False
    )
    loader = torch.utils.data.DataLoader(xai_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Run Sensitivity Analysis on classification (logits)
    logits_res = compute_logit_sensitivity(model, loader, DEVICE)

    # Run Stability (ReID) anaylsis on similarity (cosine)
    similarity_res = compute_embedding_stability(model, loader, DEVICE)
    similarity_res.to_csv(save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.csv", index=False)
    
    save_parquet(save_path / f"classification_sensitivity__{backbone_name}_{head_type}__n{n_samples}.parquet", logits_res)
    save_parquet(save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.parquet", similarity_res)

    config = {
        "n_samples": n_samples,
        "sample_indices": [int(x) for x in torch_idx],
        "data": {
            "dataset_name": dataset_name,
            "manifest_dir": str(manifest_dir),
        },
        "model": {
            "checkpoint_path": checkpoint_path,
            "num_classes":  num_classes,
            "model_backbone": backbone_name,
            "head_type": head_type,
        },
        "masking": {
            "alpha_threshold": xai_ds.alpha_threshold,
            "mask_fill_color": xai_ds.mask_fill_color
        },
        "outputs": {
            "logits_csv": str(save_path / f"classification_sensitivity__{backbone_name}_{head_type}__n{n_samples}.csv"),
            "similarity_csv": str(save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.csv"),
            "heatmaps_dir": str(results_path / "heatmaps"),
        },
    }

    with open(save_path / f"run_config__{backbone_name}_{head_type}__n{n_samples}.json", "w") as f:
        json.dump(config, f, indent=2)

    # save heatmaps
    generate_visuals_logits(model, loader, logits_res, results_path, DEVICE)