import json
from jaguar.evaluation.run_background_reliance_eval import build_original_gallery_base, extract_query_variant_embeddings
from jaguar.utils.utils_evaluation import map_emb_rows_to_local_indices
from jaguar.utils.utils_models import load_or_extract_jaguarid_embeddings
from jaguar.utils.utils_xai import get_val_query_indices, manual_gradcam_class
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
from jaguar.utils.utils_datasets import load_full_jaguar_from_FO_export
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

                pred_orig = int(logits_orig[i].argmax().item())
                pred_jaguar_only = int(logits_bg_masked[i].argmax().item())
                pred_bg_only = int(logits_fg_masked[i].argmax().item())
                            
                results.append({
                    "id": batch["id"][i],
                    "filepath": Path(batch["filepath"][i]).name,
                    "gold_idx": c,

                    "pred_orig": pred_orig,
                    "pred_jaguar_only": pred_jaguar_only,
                    "pred_bg_only": pred_bg_only,
                    "is_correct_orig": pred_orig == c,
                    "is_correct_jaguar_only": pred_jaguar_only == c,
                    "is_correct_bg_only": pred_bg_only == c,

                    "gold_logit_orig": s0,
                    "gold_logit_jaguar_only": s_bg,
                    "gold_logit_bg_only": s_fg,

                    "gold_logp_orig": s0_lp,
                    "gold_logp_jaguar_only": s_bg_lp,
                    "gold_logp_bg_only": s_fg_lp,

                    "delta_remove_bg_logit": s0 - s_bg,   # original - jaguar_only
                    "delta_remove_fg_logit": s0 - s_fg,   # original - bg_only

                    "delta_remove_bg_logp": s0_lp - s_bg_lp,
                    "delta_remove_fg_logp": s0_lp - s_fg_lp,

                    "bg_minus_jaguar_logit": s_fg - s_bg,
                    "bg_minus_jaguar_logp": s_fg_lp - s_bg_lp,

                    "is_bg_dominant_logit": s_fg > s_bg,
                    "is_bg_dominant_logp": s_fg_lp > s_bg_lp,
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
                    "is_bg_dominant": sim_fg[i].item() > sim_bg[i].item(),
                    "bg_minus_jaguar_stability": float(sim_fg[i].item() - sim_bg[i].item())
                })
    return pd.DataFrame(results)


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

    spurious = df_results[df_results["is_bg_dominant_logp"] == True].head(5)
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

                plain_row.append((img * 255).astype(np.uint8))
                eigen_row.append(e_vis)
                grad_row.append(g_vis)

            top = np.hstack(plain_row)
            eigen_grid = np.vstack([top, np.hstack(eigen_row)])
            grad_grid  = np.vstack([top, np.hstack(grad_row)])

            stem = Path(fname).stem
            cv2.imwrite(str(output_dir / f"cam_grid__eigen__{stem}.png"), eigen_grid)
            cv2.imwrite(str(output_dir / f"cam_grid__grad__{stem}.png"), grad_grid)



def summarize_retrieval_variant(df: pd.DataFrame, suffix: str) -> dict:
    return {
        "rank1": float(df[f"is_rank1_{suffix}"].mean()),
        "rank5": float(df[f"is_rank5_{suffix}"].mean()),
        "median_gold_rank": float(df[f"gold_rank_{suffix}"].median()),
        "mean_gold_rank": float(df[f"gold_rank_{suffix}"].mean()),
        "mean_margin": float(df[f"margin_gold_minus_impostor_{suffix}"].mean()),
    }

def summarize_bg_vs_jaguar(df: pd.DataFrame) -> dict:
    return {
        "share_bg_better_rank": float(df["bg_better_than_jag_rank"].mean()),
        "share_bg_better_rank1": float(df["bg_better_than_jag_rank1"].mean()),
        "share_bg_better_rank5": float(df["bg_better_than_jag_rank5"].mean()),
        "share_bg_better_margin": float(df["bg_better_than_jag_margin"].mean()),
        "median_rank_delta_bg_minus_jag": float(df["gold_rank_delta_bg_minus_jag"].median()),
        "mean_margin_delta_bg_minus_jag": float(df["margin_delta_bg_minus_jag"].mean()),
    }

def summarize_embedding_stability(df: pd.DataFrame) -> dict:
    delta = df["stability_bg_only"] - df["stability_jaguar_only"]
    return {
        "mean_stability_jaguar_only": float(df["stability_jaguar_only"].mean()),
        "mean_stability_bg_only": float(df["stability_bg_only"].mean()),
        "median_stability_jaguar_only": float(df["stability_jaguar_only"].median()),
        "median_stability_bg_only": float(df["stability_bg_only"].median()),
        "share_bg_more_stable": float((df["stability_bg_only"] > df["stability_jaguar_only"]).mean()),
        "mean_stability_delta_bg_minus_jag": float(delta.mean()),
        "median_stability_delta_bg_minus_jag": float(delta.median()),
    }



def select_val_samples_from_emb_rows(ctx_orig, query_emb_rows: np.ndarray) -> list[dict]:
    """
    Resolve global val emb_row ids to the corresponding val dataset samples.
    """
    val_local_idx = map_emb_rows_to_local_indices(
        query_emb_rows,
        ctx_orig.val_local_to_emb_row,
    )
    return [ctx_orig.val_ds.samples[int(i)] for i in val_local_idx]

if __name__ == "__main__":
    n_samples = 10
    dataset_name = "jaguar_init"
    manifest_dir = resolve_path("fiftyone/init", DATA_STORE)
    run_name = f"{backbone_name}__{head_type}__bg-{BACKGROUND}"
    config = ""

    save_path = resolve_path(f"xai/background_sensitivity/{run_name}", EXPERIMENTS_STORE)
    ensure_dir(save_path)
    results_path = resolve_path("xai/background_sensitivity", RESULTS_STORE)
    ensure_dir(results_path)

    checkpoint_dir = Path(config["evaluation"]["checkpoint_dir"])
    base = build_original_gallery_base(
        config=config,
        checkpoint_dir=checkpoint_dir,
    )
    ctx_orig = base["ctx_orig"]
        
    # Load Sample List
    query_emb_rows = get_val_query_indices(
        split_df=ctx_orig.split_df,
        out_root=save_path,
        n_samples=n_samples,
        seed=config["xai"]["seed"],
    )

    samples_list = select_val_samples_from_emb_rows(
        ctx_orig=ctx_orig,
        query_emb_rows=query_emb_rows,
    )

    
    """
    # load model weights
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tmp_state = ckpt["model_state_dict"]
    state = {k.replace("module.", "", 1): v for k, v in tmp_state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("full model missing:", len(missing), "unexpected:", len(unexpected))
    model = model.to(DEVICE).eval()
    """

    # Create DataLoaders
    gallery_ds.transform = ctx_orig.model.backbone_wrapper.transform

    # 2) build query dataset/loader
    query_ds = MaskAwareJaguarDataset(
        jaguar_model=ctx_orig.model,
        base_root=PATHS.data_train,
        data_root=PATHS.data.parent,
        samples_list=samples_list,
    )
    query_loader = torch.utils.data.DataLoader(
        query_ds,
        batch_size=config["inference"]["batch_size"],
        shuffle=False,
    )
    
    
    # 3) embeddings
    


    # 4) retrieval dataframe
    retrieval_df = compute_retrieval_bg_vs_jaguar(
        query_emb_orig=query_emb_orig,
        query_emb_jaguar_only=query_emb_jag,
        query_emb_bg_only=query_emb_bg,
        query_ids=query_ids,
        query_files=query_files,
        gallery_emb=gallery_emb,
        gallery_ids=gallery_ids,
        gallery_files=gallery_files,
    )

    # Run Sensitivity Analysis on classification (logits)
    logits_res = compute_logit_sensitivity(model, query_loader, DEVICE)

    # Run Stability (ReID) anaylsis on similarity (cosine)
    similarity_res = compute_embedding_stability(model, query_loader, DEVICE)
    
    analysis_df = retrieval_df.merge(
        similarity_res,
        on=["id", "filepath"],
        how="left",
    )

    retrieval_variant_summary = {
        "orig": summarize_retrieval_variant(analysis_df, "orig"),
        "jaguar_only": summarize_retrieval_variant(analysis_df, "jaguar_only"),
        "bg_only": summarize_retrieval_variant(analysis_df, "bg_only"),
    }

    analysis_correct = analysis_df[analysis_df["is_rank1_orig"]].copy()
    analysis_wrong = analysis_df[~analysis_df["is_rank1_orig"]].copy()

    summary_all = summarize_bg_vs_jaguar(analysis_df)
    summary_correct = summarize_bg_vs_jaguar(analysis_correct)
    summary_wrong = summarize_bg_vs_jaguar(analysis_wrong)

    similarity_summary = {
        "all": summarize_embedding_stability(analysis_df),
        "orig_rank1_correct": summarize_embedding_stability(analysis_correct),
        "orig_rank1_wrong": summarize_embedding_stability(analysis_wrong),
    }
    
    # save results
    similarity_res.to_csv(save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.csv", index=False)
    
    # classification_sensitivity.parquet
    save_parquet(save_path / f"classification_sensitivity__{backbone_name}_{head_type}__n{n_samples}.parquet", logits_res)
    # similarity_stability.parquet
    save_parquet(save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.parquet", similarity_res)
    
    # analysis_merged.parquet
    save_parquet(
        save_path / f"analysis_merged__{backbone_name}_{head_type}__n{n_samples}.parquet",
        analysis_df
    )

    config = {
        "n_samples": n_samples,
        "sample_indices": [int(x) for x in torch_idx],
        "background_setting": BACKGROUND,
        "seed": SEED,
        "data": {
            "dataset_name": dataset_name,
            "manifest_dir": to_rel_path(manifest_dir),
            "gallery_source": "train_full"          ## !TODO muss eigentlich abhängig vom Modell sein oder? Was es im training gesehen hat - split data als source of truth
        },
        "model": {
            "checkpoint_path": to_rel_path(checkpoint_path),
            "num_classes":  num_classes,
            "model_backbone": backbone_name,
            "head_type": head_type,
        },
        "masking": {
            "alpha_threshold": query_ds.alpha_threshold,
            "mask_fill_color": query_ds.mask_fill_color
        },
        "outputs": {
            "logits_parquet": to_rel_path(save_path / f"classification_sensitivity__{backbone_name}_{head_type}__n{n_samples}.parquet"),
            "similarity_parquet": to_rel_path(save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.parquet"),
            "retrieval_parquet": to_rel_path(save_path / f"retrieval_bg_vs_jaguar__{backbone_name}_{head_type}__n{n_samples}.parquet"),
            "retrieval_summary_json": to_rel_path(save_path / f"retrieval_summary__{backbone_name}_{head_type}__n{n_samples}.json"),
            "similarity_summary_json": to_rel_path(save_path / f"similarity_summary__{backbone_name}_{head_type}__n{n_samples}.json"),
            "heatmaps_dir": to_rel_path(results_path / "heatmaps"),
            "analysis_merged_parquet": to_rel_path(save_path / f"analysis_merged__{backbone_name}_{head_type}__n{n_samples}.parquet"),
        },
    }

    # run_config.json
    with open(save_path / f"run_config__{backbone_name}_{head_type}__n{n_samples}.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # similarity_summary.json
    with open(save_path / f"similarity_summary__{backbone_name}_{head_type}__n{n_samples}.json", "w") as f:
        json.dump(similarity_summary, f, indent=2)

    # retrieval_bg_vs_jaguar.parquet
    save_parquet(
        save_path / f"retrieval_bg_vs_jaguar__{backbone_name}_{head_type}__n{n_samples}.parquet",
        retrieval_df
    )

    retrieval_summary = {
        "all": summary_all,
        "orig_rank1_correct": summary_correct,
        "orig_rank1_wrong": summary_wrong,
        "variant_summary": retrieval_variant_summary,
    }

    # retrieval_summary.json
    with open(save_path / f"retrieval_summary__{backbone_name}_{head_type}__n{n_samples}.json", "w") as f:
        json.dump(retrieval_summary, f, indent=2)

    # save heatmaps
    generate_visuals_logits(model, query_loader, logits_res, results_path, DEVICE)