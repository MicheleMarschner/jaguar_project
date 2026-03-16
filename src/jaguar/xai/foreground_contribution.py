"""
Background Sensitivity Analysis for Jaguar Re-ID.

Project role:
- Compares retrieval and classification behavior across original, jaguar-only, and background-only query variants.
- Quantifies whether identity evidence is carried more by the animal region or the background.
- Produces qualitative heatmaps for selected background-dominant cases.

Procedure:
- Extract embeddings for original, jaguar-only, and background-only query views.
- Compare retrieval outcomes against a fixed gallery across the three query variants.
- Measure classification sensitivity via logit/log-prob changes under masking.
- Measure embedding stability via cosine similarity to the original embedding.
- Summarize background-vs-jaguar dominance and save analysis artifacts.

Purpose:
- Test whether retrieval or classification decisions rely disproportionately on background cues.
- Provide quantitative and qualitative evidence for spurious background dependence.
"""

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import cv2
import torch.nn.functional as F
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from jaguar.config import IMGNET_MEAN, IMGNET_STD, DEVICE
from jaguar.utils.utils import ensure_dir, save_parquet, to_rel_path
from jaguar.utils.utils_xai_similarity import manual_gradcam_class

def extract_query_variant_embeddings(model, dataloader, device):
    """Extracts embeddings for original, jaguar-only, and background-only query views."""
    model.eval()

    emb_orig_all = []
    emb_jag_all = []
    emb_bg_all = []

    query_ids = []
    query_files = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extract query embeddings"):
            if batch is None:
                continue

            x_orig = batch["t_orig"].to(device)
            x_jag  = batch["t_bg_masked"].to(device)   # jaguar-only
            x_bg   = batch["t_fg_masked"].to(device)   # bg-only

            emb_orig = model.get_embeddings(x_orig).detach().cpu().numpy()
            emb_jag  = model.get_embeddings(x_jag).detach().cpu().numpy()
            emb_bg   = model.get_embeddings(x_bg).detach().cpu().numpy()

            emb_orig_all.append(emb_orig)
            emb_jag_all.append(emb_jag)
            emb_bg_all.append(emb_bg)

            query_ids.extend(batch["id"])
            query_files.extend([Path(p).name for p in batch["filepath"]])

    return (
        np.vstack(emb_orig_all),
        np.vstack(emb_jag_all),
        np.vstack(emb_bg_all),
        list(query_ids),
        list(query_files),
    )


def compute_retrieval_metrics_per_query(
    query_emb: np.ndarray,
    query_ids: list[str],
    query_files: list[str],
    gallery_emb: np.ndarray,
    gallery_ids: list[str],
    gallery_files: list[str],
) -> pd.DataFrame:
    """
    Computes per-query retrieval outcomes; excludes exact self-match via filename.
    Assumes embeddings are normalized.
    """
    sim = query_emb @ gallery_emb.T

    gallery_ids = np.asarray(gallery_ids)
    gallery_files = np.asarray(gallery_files)

    rows = []

    for i in range(len(query_ids)):
        gold_id = query_ids[i]
        qfile = query_files[i]

        scores = sim[i].copy()

        self_mask = (gallery_files == qfile)
        scores[self_mask] = -np.inf

        ranked_idx = np.argsort(-scores)

        gold_mask = (gallery_ids == gold_id) & (~self_mask)
        non_gold_mask = (gallery_ids != gold_id) & (~self_mask)

        if not gold_mask.any():
            rows.append({
                "id": gold_id,
                "filepath": qfile,
                "gold_rank": np.nan,
                "is_rank1": False,
                "is_rank5": False,
                "best_gold_similarity": np.nan,
                "best_impostor_similarity": np.nan,
                "margin_gold_minus_impostor": np.nan,
            })
            continue

        best_gold_similarity = float(scores[gold_mask].max())
        best_impostor_similarity = float(scores[non_gold_mask].max()) if non_gold_mask.any() else np.nan

        ranked_gallery_ids = gallery_ids[ranked_idx]
        first_gold_pos = int(np.where(ranked_gallery_ids == gold_id)[0][0]) + 1

        rows.append({
            "id": gold_id,
            "filepath": qfile,
            "gold_rank": first_gold_pos,
            "is_rank1": first_gold_pos <= 1,
            "is_rank5": first_gold_pos <= 5,
            "best_gold_similarity": best_gold_similarity,
            "best_impostor_similarity": best_impostor_similarity,
            "margin_gold_minus_impostor": (
                best_gold_similarity - best_impostor_similarity
                if not np.isnan(best_impostor_similarity) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def compute_retrieval_bg_vs_jaguar(
    query_emb_orig: np.ndarray,
    query_emb_jaguar_only: np.ndarray,
    query_emb_bg_only: np.ndarray,
    query_ids: list[str],
    query_files: list[str],
    gallery_emb: np.ndarray,
    gallery_ids: list[str],
    gallery_files: list[str],
) -> pd.DataFrame:
    """Compares whether Jaguar identity is driven more by the animal region or the background."""
    df_orig = compute_retrieval_metrics_per_query(
        query_emb_orig, query_ids, query_files,
        gallery_emb, gallery_ids, gallery_files
    ).add_suffix("_orig")

    df_jag = compute_retrieval_metrics_per_query(
        query_emb_jaguar_only, query_ids, query_files,
        gallery_emb, gallery_ids, gallery_files
    ).add_suffix("_jaguar_only")

    df_bg = compute_retrieval_metrics_per_query(
        query_emb_bg_only, query_ids, query_files,
        gallery_emb, gallery_ids, gallery_files
    ).add_suffix("_bg_only")

    df = pd.concat([df_orig, df_jag, df_bg], axis=1)

    # restore clean identifiers
    df["id"] = df["id_orig"]
    df["filepath"] = df["filepath_orig"]

    df["bg_better_than_jag_rank"] = df["gold_rank_bg_only"] < df["gold_rank_jaguar_only"]
    df["bg_better_than_jag_rank1"] = (
        df["is_rank1_bg_only"].astype(int) > df["is_rank1_jaguar_only"].astype(int)
    )
    df["bg_better_than_jag_rank5"] = (
        df["is_rank5_bg_only"].astype(int) > df["is_rank5_jaguar_only"].astype(int)
    )
    df["bg_better_than_jag_margin"] = (
        df["margin_gold_minus_impostor_bg_only"] >
        df["margin_gold_minus_impostor_jaguar_only"]
    )

    df["gold_rank_delta_bg_minus_jag"] = (
        df["gold_rank_bg_only"] - df["gold_rank_jaguar_only"]
    )
    df["margin_delta_bg_minus_jag"] = (
        df["margin_gold_minus_impostor_bg_only"] -
        df["margin_gold_minus_impostor_jaguar_only"]
    )

    return df

def compute_logit_sensitivity(model, dataloader, device):
    """
    Compare gold-class classification scores across original, jaguar-only, and background-only queries.

    Runs the classifier on all three query variants and records how much the gold-class
    logit and log-probability drop when foreground or background information is removed.
    """
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
    """
    Measure how much masked-query embeddings stay aligned with the original query embedding.

    Extracts ReID embeddings for original, jaguar-only, and background-only inputs and
    compares each masked variant to the original with cosine similarity.
    """
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
    """
    Create CAM comparison grids for the most background-dominant classification cases.

    For selected samples, saves side-by-side original, EigenCAM, and GradCAM views for
    original, jaguar-only, and background-only query variants.
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
                g_map = manual_gradcam_class(
                    model,
                    target_layer,
                    tensor,
                    cls,
                    reshape_transform=reshape_transform,
                )
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


def run_foreground_contribution_analysis(
    model,
    query_loader,
    results_path: Path,
) -> dict:
    """
    Run foreground-vs-background diagnostics on classification and embedding behavior.

    Produces per-sample classification sensitivity scores, embedding stability scores,
    and qualitative CAM visualizations for representative cases.
    """
    
    # Run Sensitivity Analysis on classification (logits)
    logits_res = compute_logit_sensitivity(model, query_loader, DEVICE)
    # Run Stability (ReID) anaylsis on similarity (cosine)
    similarity_res = compute_embedding_stability(model, query_loader, DEVICE)

    generate_visuals_logits(model, query_loader, logits_res, results_path, DEVICE)

    return {
        "logits_res": logits_res,
        "similarity_res": similarity_res,
    }


def run_bg_vs_jaguar_stress_analysis(
    model,
    query_loader,
    gallery_emb,
    gallery_ids,
    gallery_files,
) -> dict:
    """
    Evaluate retrieval changes when queries keep only jaguar content or only background.

    Extracts embeddings for all query variants and compares retrieval outcomes against a
    fixed gallery to quantify foreground-versus-background dependence in ranking behavior.
    """
    query_emb_orig, query_emb_jag, query_emb_bg, query_ids, query_files = extract_query_variant_embeddings(
        model,
        query_loader,
        DEVICE,
    )

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

    return {
        "retrieval_df": retrieval_df,
    }


def save_bg_sensitivity_outputs(
    save_path: Path,
    results_path: Path,
    config: dict,
    train_config: dict,
    manifest_dir: Path,
    ctx_orig,
    query_ds,
    query_emb_rows: np.ndarray,
    logits_res: pd.DataFrame,
    similarity_res: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    retrieval_summary: dict,
    similarity_summary: dict,
) -> None:
    """
    Persist all foreground-vs-background analysis artifacts and a run-level config record.

    Saves the core per-sample tables, summary JSON files, and a compact metadata file
    that documents which data, model, masking setup, and output paths belong to the run.
    """
    backbone_name = train_config["model"]["backbone_name"]
    head_type = train_config["model"]["head_type"]
    n_samples = config["xai"]["n_samples"]
    dataset_name = config["xai"]["dataset_name"]

    save_parquet(
        save_path / f"classification_sensitivity__{backbone_name}_{head_type}__n{n_samples}.parquet",
        logits_res,
    )

    save_parquet(
        save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.parquet",
        similarity_res,
    )

    save_parquet(
        save_path / f"analysis_merged__{backbone_name}_{head_type}__n{n_samples}.parquet",
        analysis_df,
    )

    save_parquet(
        save_path / f"retrieval_bg_vs_jaguar__{backbone_name}_{head_type}__n{n_samples}.parquet",
        retrieval_df,
    )

    with open(save_path / f"similarity_summary__{backbone_name}_{head_type}__n{n_samples}.json", "w") as f:
        json.dump(similarity_summary, f, indent=2)

    with open(save_path / f"retrieval_summary__{backbone_name}_{head_type}__n{n_samples}.json", "w") as f:
        json.dump(retrieval_summary, f, indent=2)

    run_config = {
        "n_samples": n_samples,
        "query_emb_rows": [int(x) for x in query_emb_rows],
        "seed": config["xai"]["seed"],
        "data": {
            "dataset_name": dataset_name,
            "manifest_dir": to_rel_path(manifest_dir),
            "gallery_source": "original_train_plus_val_from_shared_base",
        },
        "model": {
            "checkpoint_dir": to_rel_path(ctx_orig.checkpoint_dir),
            "model_backbone": backbone_name,
            "head_type": head_type,
        },
        "masking": {
            "alpha_threshold": query_ds.alpha_threshold,
            "mask_fill_color": query_ds.mask_fill_color,
        },
        "outputs": {
            "logits_parquet": to_rel_path(
                save_path / f"classification_sensitivity__{backbone_name}_{head_type}__n{n_samples}.parquet"
            ),
            "similarity_parquet": to_rel_path(
                save_path / f"similarity_stability__{backbone_name}_{head_type}__n{n_samples}.parquet"
            ),
            "retrieval_parquet": to_rel_path(
                save_path / f"retrieval_bg_vs_jaguar__{backbone_name}_{head_type}__n{n_samples}.parquet"
            ),
            "retrieval_summary_json": to_rel_path(
                save_path / f"retrieval_summary__{backbone_name}_{head_type}__n{n_samples}.json"
            ),
            "similarity_summary_json": to_rel_path(
                save_path / f"similarity_summary__{backbone_name}_{head_type}__n{n_samples}.json"
            ),
            "heatmaps_dir": to_rel_path(results_path / "heatmaps"),
            "analysis_merged_parquet": to_rel_path(
                save_path / f"analysis_merged__{backbone_name}_{head_type}__n{n_samples}.parquet"
            ),
        },
    }

    with open(save_path / f"run_config__{backbone_name}_{head_type}__n{n_samples}.json", "w") as f:
        json.dump(run_config, f, indent=2)