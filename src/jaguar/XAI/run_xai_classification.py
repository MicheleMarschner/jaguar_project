import json
from jaguar.utils.utils_xai import manual_gradcam_class
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from jaguar.config import DATA_STORE, EXPERIMENTS_STORE, IMGNET_MEAN, IMGNET_STD, PATHS, DEVICE, RESULTS_STORE, SEED
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.datasets.JaguarDataset import MaskAwareJaguarDataset
from jaguar.utils.utils import ensure_dir, resolve_path, save_parquet, to_rel_path



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
        id_to_indices[str(y)].append(idx)

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
    """"
    Generates GradCam overlays for spurious cases
    """
    output_dir = Path(output_dir) / "heatmaps"
    ensure_dir(output_dir)

    wrapper = model.backbone_wrapper
    grad_cam_cfg = wrapper.registry_entry["grad_cam"]
    layer_getter = grad_cam_cfg["layer_getter"]
    reshape_transform = grad_cam_cfg["reshape_transform"]
    target_layer = layer_getter(model.backbone)

    eigen_cam = EigenCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

    spurious = df_results[df_results["is_spurious"] == True].head(5)
    if spurious.empty:
        spurious = df_results.head(5)
    target_files = set(spurious["filepath"].values)

    for batch in dataloader:
        for i in range(len(batch["filepath"])):
            fname = Path(batch["filepath"][i]).name
            if fname not in target_files:
                continue

            cls = int(batch["label_idx"][i].item())

            inputs = {
                "Orig": batch["t_orig"][i].unsqueeze(0).to(device),
                "Jaguar": batch["t_bg_masked"][i].unsqueeze(0).to(device),
                "BG": batch["t_fg_masked"][i].unsqueeze(0).to(device),
            }

            plain_row = []
            eigen_row = []
            grad_row  = []

            for _, tensor in inputs.items():
                # tensor: [1,3,H,W] normalized, exactly what CAM uses
                x = tensor[0].detach().cpu()  # [3,H,W]

                # ---- image used by CAM (unnormalize) ----
                mean = torch.tensor(IMGNET_MEAN)[:, None, None]
                std  = torch.tensor(IMGNET_STD)[:, None, None]
                img = (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()  # [H,W,3] in [0,1]
                H, W = img.shape[:2]

                # ---- EigenCAM heatmap ----
                e_map = eigen_cam(input_tensor=tensor)[0, :]                  # [h,w]
                e_map = cv2.resize(e_map, (W, H), interpolation=cv2.INTER_LINEAR)
                e_vis = show_cam_on_image(img, e_map, use_rgb=True)

                # ---- Manual GradCAM heatmap ----
                g_map = manual_gradcam_class(model, target_layer, tensor, cls)  # [h,w]
                g_map = cv2.resize(g_map, (W, H), interpolation=cv2.INTER_LINEAR)
                g_vis = show_cam_on_image(img, g_map, use_rgb=True)

                # =========================
                # CHECK 3: does displayed img have background pixels (from file RGB under alpha==0)?
                # =========================
                rgba = Image.open(batch["filepath"][i]).convert("RGBA").resize((W, H), Image.BILINEAR)
                a = np.array(rgba.getchannel("A"))
                bg_mask = (a == 0)

                img_u8 = (img * 255).astype(np.uint8)
                bg_pix = img_u8[bg_mask]

                print(
                    f"[CHK3] {Path(batch['filepath'][i]).name} | variant={'??'} "
                    f"| bg_pix_max={bg_pix.max(axis=0) if bg_pix.size else None} "
                    f"| bg_pix_mean={bg_pix.mean(axis=0) if bg_pix.size else None}"
                )

                plain_row.append((img * 255).astype(np.uint8))
                eigen_row.append(e_vis)
                grad_row.append(g_vis)

            top = np.hstack(plain_row)
            eigen_grid = np.vstack([top, np.hstack(eigen_row)])
            grad_grid  = np.vstack([top, np.hstack(grad_row)])

            stem = Path(fname).stem
            cv2.imwrite(str(output_dir / f"cam_grid__eigen__{stem}.png"), eigen_grid)
            cv2.imwrite(str(output_dir / f"cam_grid__grad__{stem}.png"), grad_grid)




if __name__ == "__main__":
    BATCH_SIZE = 16
    backbone_name = "MiewID"
    head_type = "arcface"
    n_samples = 10
    dataset_name = "jaguar_init"
    manifest_dir = resolve_path("fiftyone/init", DATA_STORE)
    checkpoint_path = PATHS.checkpoints / "jaguar_reid_v1_epoch_16.pth"

    save_path = resolve_path("xai/background_sensitivity", EXPERIMENTS_STORE)
    ensure_dir(save_path)
    results_path = resolve_path("xai/background_sensitivity", RESULTS_STORE)
    ensure_dir(results_path)
    
    # Load Sample List
    _, temp_ds = load_jaguar_from_FO_export(manifest_dir, dataset_name=dataset_name)
    labels_all = [str(x) for x in temp_ds.labels]
    unique_ids = sorted(set(labels_all))
    label2idx = {lab: i for i, lab in enumerate(unique_ids)}
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
    
    # load model weights
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tmp_state = ckpt["model_state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in tmp_state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("full model missing:", len(missing), "unexpected:", len(unexpected))
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
    
    # save results
    similarity_res.to_csv(save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.csv", index=False)
    save_parquet(save_path / f"classification_sensitivity__{backbone_name}_{head_type}__n{n_samples}.parquet", logits_res)
    save_parquet(save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.parquet", similarity_res)

    config = {
        "n_samples": n_samples,
        "sample_indices": [int(x) for x in torch_idx],
        "data": {
            "dataset_name": dataset_name,
            "manifest_dir": to_rel_path(manifest_dir),
        },
        "model": {
            "checkpoint_path": to_rel_path(checkpoint_path),
            "num_classes":  num_classes,
            "model_backbone": backbone_name,
            "head_type": head_type,
        },
        "masking": {
            "alpha_threshold": xai_ds.alpha_threshold,
            "mask_fill_color": xai_ds.mask_fill_color
        },
        "outputs": {
            "logits_parquet": to_rel_path(save_path / f"classification_sensitivity__{backbone_name}_{head_type}__n{n_samples}.parquet"),
            "similarity_parquet": to_rel_path(save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.parquet"),
            "heatmaps_dir": to_rel_path(results_path / "heatmaps"),
        },
    }

    with open(save_path / f"run_config__{backbone_name}_{head_type}__n{n_samples}.json", "w") as f:
        json.dump(config, f, indent=2)

    # save heatmaps
    generate_visuals_logits(model, loader, logits_res, results_path, DEVICE)