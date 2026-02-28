"""
XAI metric evaluation for Jaguar Re-ID (post-hoc explanation assessment).

Purpose in the project:
- takes the saved XAI saliency artifacts from run_xai (IG / GradCAM, easy_pos / hard_neg, ...)
- computes a small set of *explanation quality signals*:
  - sanity: does the explanation depend on model parameters?
  - faithfulness: does removing “important” pixels reduce similarity?
  - complexity: how sparse / concentrated is the explanation?
- saves per-sample metric vectors + a summary CSV for reporting/comparison across explainers

This script evaluates *explanations*, not retrieval performance.
"""

import gc
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import quantus
import copy
import pandas as pd

from typing import Any, Dict

from jaguar.XAI.run_xai_similarity import XAIConfig, compute_saliency_gradcam_for_pair_type, compute_saliency_ig_for_pair_type
from jaguar.config import DEVICE, PATHS, SEED
from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.utils.utils import ensure_dir, load_parquet
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.utils.utils_xai import SimilarityForward, save_vec


# ============================================================
# Faithfulness
# ============================================================
def _auc_trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))

@torch.no_grad()
def faithfulness_deletion_auc(
    artifact: dict,
    torch_ds,
    model_wrapper,
    *,
    steps: int = 20,
    baseline: str = "zeros",         
    use_abs: bool = True,            
) -> dict:
    """
    Faithfulness (Deletion AUC) for pair-similarity explanations.
    If the saliency highlights truly important pixels, then deleting them should reduce 
    query↔reference similarity quickly (lower curve / lower AUC after normalization).
    """
    device = model_wrapper.device
    model_wrapper.eval()

    q = artifact["query_indices"].cpu().numpy().astype(np.int64)
    r = artifact["ref_indices"].cpu().numpy().astype(np.int64)
    sal = artifact["saliency"].cpu()  # [N,H,W]
    if use_abs:
        sal = sal.abs()

    N, H, W = sal.shape
    # group pair positions by ref index to reuse ref embedding
    groups = defaultdict(list)
    for pos, ref_idx in enumerate(r):
        groups[int(ref_idx)].append(int(pos))

    # x-axis as fraction deleted
    fracs = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)

    aucs = np.empty((N,), dtype=np.float32)

    # Build a deletion curve by progressively replacing the most salient pixels with a baseline.
    # Curve is measured in similarity-to-reference, because our task is retrieval similarity.
    for ref_idx, positions in tqdm(groups.items(), desc="Faithfulness (deletion) refs"):
        # reference embedding once
        ref_sample = torch_ds[int(ref_idx)]
        ref_x = ref_sample["img"].unsqueeze(0).to(device)  # [1,3,H,W]
        ref_emb = model_wrapper.get_embeddings_tensor(ref_x).squeeze(0)  # [D]

        # scoring function f_ref(x) -> similarity
        sim_model = SimilarityForward(model_wrapper, ref_emb, maximize=True).to(device).eval()

        # process each pair in this ref group
        for pos in positions:
            qi = int(q[pos])
            x0 = torch_ds[qi]["img"].to(device)  # [3,H,W]

            # baseline image
            if baseline == "zeros":
                x_base = torch.zeros_like(x0)
            else:
                raise ValueError(f"Unknown baseline: {baseline}")

            # flatten saliency and get deletion order
            a = sal[pos].reshape(-1)  # [H*W]
            order = torch.argsort(a, descending=True)  # most relevant first

            # evaluate similarity curve
            sims = []
            for t in range(steps + 1):
                k = int(fracs[t] * (H * W))
                x = x0.clone()
                if k > 0:
                    flat = x.view(3, -1)         # [3, H*W]
                    flat[:, order[:k]] = x_base.view(3, -1)[:, order[:k]]
                s = sim_model(x.unsqueeze(0)).detach().float().cpu().item()  # scalar
                sims.append(s)

            sims = np.asarray(sims, dtype=np.float32)
            # (optional) normalize by initial similarity so curves are comparable
            if sims[0] != 0:
                sims_norm = sims / sims[0]
            else:
                sims_norm = sims

            aucs[pos] = _auc_trapz(sims_norm, fracs)

    return {
        "faith_vec": aucs,                         # [N]
        "faith_mean": float(np.mean(aucs)),
        "faith_median": float(np.median(aucs)),
        "faith_std": float(np.std(aucs)),
        "meta": {
            "metric": "deletion_auc",
            "steps": int(steps),
            "baseline": baseline,
            "use_abs": bool(use_abs),
        }
    }


def run_faithfulness_metric(artifact, torch_ds, model_wrapper):
    res_faith = faithfulness_deletion_auc(artifact, torch_ds, model_wrapper, steps=20, use_abs=True)
    gc.collect(); torch.cuda.empty_cache()

    return res_faith

# ============================================================
# Sanity / Robustness
# ============================================================

def _spearman_corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    # ranks (dense enough for our use)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra.astype(np.float32)
    rb = rb.astype(np.float32)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
    return float((ra @ rb) / denom)

def _randomize_model_params_(model: torch.nn.Module) -> None:
    """
    In-place randomization: calls reset_parameters() where available.
    Falls back to normal init for weights if reset_parameters doesn't exist.
    """
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            try:
                m.reset_parameters()
            except Exception:
                pass


def sanity_parameter_randomization(
    artifact: dict,
    torch_ds,
    model_wrapper,
    compute_saliency_fn,
    ref_df_pair: "pd.DataFrame",
    cfg,
) -> dict:
    """
    Sanity check: compare saved saliency maps vs saliency maps from a randomized model.
    Explanations should change when the model is randomized. If correlation stays high, 
    the “explanation” may be dataset/architecture bias rather than model reasoning.
    """
    # original saliency
    sal0 = artifact["saliency"].cpu().numpy()  # [N,H,W]
    N = sal0.shape[0]

    # clone wrapper+model, randomize weights
    mw_rand = copy.deepcopy(model_wrapper)
    _randomize_model_params_(mw_rand.model)
    mw_rand.eval()

    # recompute saliency with randomized model
    art_rand = compute_saliency_fn(torch_ds, mw_rand, ref_df_pair, cfg)
    sal1 = art_rand["saliency"].cpu().numpy()  # [N,H,W]

    # correlations
    cors = np.empty((N,), dtype=np.float32)
    for i in range(N):
        cors[i] = _spearman_corr_flat(sal0[i].reshape(-1), sal1[i].reshape(-1))

    return {
        "sanity_vec": cors,
        "sanity_mean": float(np.mean(cors)),
        "sanity_median": float(np.median(cors)),
        "sanity_std": float(np.std(cors)),
        "meta": {"metric": "parameter_randomization_spearman"},
    }


def run_sanity_metric(
    explainer_name: str, 
    artifact: Dict[str, Any], 
    torch_ds, 
    model_wrapper, 
    ref_df_pair, 
    cfg
):
    if explainer_name == "IG":
        compute_saliency_fn = compute_saliency_ig_for_pair_type
    elif explainer_name == "GradCAM":
        compute_saliency_fn = compute_saliency_gradcam_for_pair_type
    else:
        compute_saliency_fn = None

    if compute_saliency_fn is not None:
        res_sanity = sanity_parameter_randomization(
            artifact=artifact, 
            torch_ds=torch_ds, 
            model_wrapper=model_wrapper, 
            compute_saliency_fn=compute_saliency_fn, 
            ref_df_pair=ref_df_pair, 
            cfg=cfg
        )
        gc.collect(); torch.cuda.empty_cache()

    return res_sanity


# ============================================================
# Complexity
# ============================================================

def run_complexity_metric(artifact, model_wrapper):
    """
    Complexity signal: measures how “focused” the saliency is. Useful as a descriptive statistic 
    when comparing explainers (not inherently good or bad — depends on whether you expect 
    localized vs distributed cues).
    """
    sal = artifact["saliency"].cpu().numpy()       # [N,H,W]
    a_batch = sal[:, None, :, :]                   # [N,1,H,W]

    # dummy inputs to satisfy signature
    N, _, H, W = a_batch.shape
    x_batch = np.zeros((N, 1, H, W), dtype=np.float32)
    y_batch = np.zeros((N,), dtype=np.int64)

    metric = quantus.Sparseness(
        abs=True,
        normalise=True,
        return_aggregate=False,
        # aggregate_func=np.mean,
        disable_warnings=True,
        display_progressbar=True,
    )
    model_wrapper.eval()

    sparseness = metric(
        model=None,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        s_batch=None,
    )

    spar = np.asarray(sparseness, dtype=np.float32)
    res_complexity = {
        "complexity_vec": spar,
        "complexity_mean": float(spar.mean()),
        "complexity_median": float(np.median(spar)),
        "complexity_std": float(np.std(spar)),
        "meta": {"metric": "sparseness"}
        }
    gc.collect(); torch.cuda.empty_cache()

    return res_complexity



# ============================================================
# Orchestration
# ============================================================

def run_xai_metrics(cfg) -> pd.DataFrame:
    model_name = "MiewID"      # MiewID, ConvNeXt-V2
    explainer_names = ["GradCAM", "IG"]

    run_root = cfg.out_root / f"{model_name}__{cfg.split_name}__n{cfg.n_samples}__seed{cfg.seed}"
    # locate an existing XAI run folder (metrics assume saliency artifacts already exist)
    if not run_root.exists():
        print(f"[Skip] Missing run_root: {run_root} (run XAI first)")
        raise SystemExit(0)
    
    metrics_path = run_root / "metrics"
    ensure_dir(metrics_path)

    # Load the same dataset + model backbone used for scoring similarity 
    # (needed for faithfulness and sanity recomputation).
    _, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export / "splits_curated",      
        dataset_name=cfg.dataset_name,
        processing_fn=None,
        overwrite_db=False, 
    )
    
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    torch_ds.transform = model_wrapper.preprocess 
    model_wrapper.eval()

    # Load the mined reference pairs so metrics are aligned to the exact evaluated pairs.
    out_path = run_root / f"refs_n{cfg.n_samples}.parquet"
    ref_df = load_parquet(out_path)

    rows = []
    
    # For each (explainer, pair_type): load saved saliency artifact, compute sanity + 
    # faithfulness + complexity and save per-sample vectors + aggregate summary row
    for explainer_name in explainer_names:
        for pair_type in cfg.pair_types:
            
            ref_df_pair = ref_df[ref_df["pair_type"] == pair_type].reset_index(drop=True)
            sal_path = run_root / "explanations" / f"{explainer_name}" / f"sal__{pair_type}.pt"
            artifact = torch.load(sal_path, map_location="cpu")

            sanity = run_sanity_metric(explainer_name, artifact, torch_ds, model_wrapper, ref_df_pair, cfg)
            faith = run_faithfulness_metric(artifact, torch_ds, model_wrapper)
            complexity = run_complexity_metric(artifact, model_wrapper)

            rows.append({
                "explainer": explainer_name,
                "pair_type": pair_type,

                "sanity_mean": sanity["sanity_mean"],
                "sanity_median": sanity["sanity_median"],
                "sanity_std": sanity["sanity_std"],
                "sanity_metric": sanity["meta"]["metric"],
                "sanity_vec_path": save_vec(metrics_path, "sanity", explainer_name, pair_type, sanity["sanity_vec"]),

                "faith_mean": faith["faith_mean"],
                "faith_median": faith["faith_median"],
                "faith_std": faith["faith_std"],
                "faith_metric": faith["meta"]["metric"],
                "faith_vec_path": save_vec(metrics_path, "faith", explainer_name, pair_type, faith["faith_vec"]),

                "complexity_mean": complexity["complexity_mean"],
                "complexity_median": complexity["complexity_median"],
                "complexity_std": complexity["complexity_std"],
                "complexity_metric": complexity["meta"]["metric"],
                "complexity_vec_path": save_vec(metrics_path, "complexity", explainer_name, pair_type, complexity["complexity_vec"])
            })
    # write a compact CSV summary for comparisons across explainers/pair types.
    return pd.DataFrame(rows).to_csv(metrics_path / f"xai_summary_metrics.csv", index=False)



if __name__ == "__main__":

    cfg = XAIConfig(
        dataset_name="jaguar_xai",   
        split_name="val",             
        n_samples=10,
        pair_types=("easy_pos", "hard_neg"),
        seed=SEED,
        ig_steps=10,
        ig_internal_bs=32,
        ig_batch_size=32,
        out_root=PATHS.runs / "xai/similarity",
    )

    summary_df = run_xai_metrics(cfg)

    