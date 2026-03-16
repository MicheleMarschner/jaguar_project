"""
XAI Metric Evaluation for Jaguar Re-ID.

Project role:
- Loads saved pair-level saliency artifacts from the XAI analysis stage.
- Computes a small set of explanation-quality metrics.
- Stores per-sample metric vectors and aggregate summaries.

Procedure:
- Reuse saved saliency maps for each explainer and pair type.
- Evaluate sanity via parameter randomization.
- Evaluate faithfulness via deletion-based similarity drop.
- Evaluate complexity via saliency sparsity.
- Save summary tables and reusable metric artifacts.

Purpose:
- Assess the quality of post-hoc explanations rather than retrieval performance.
- Support comparison of explainers and pair types in a reproducible way.
"""

import gc
from pathlib import Path
from jaguar.experiments.experiment_output import save_requested_outputs
from jaguar.utils.utils_xai_class import compute_saliency_gradcam_class, compute_saliency_ig_class
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import quantus
import copy
import pandas as pd
import torch.nn.functional as F

from typing import Any, Callable, Dict

from jaguar.xai.xai_similarity import build_emb_row_sample_resolver, compute_saliency_gradcam_for_pair_type, compute_saliency_ig_for_pair_type
from jaguar.config import EXPERIMENTS_STORE, PATHS
from jaguar.logging.wandb_logger import init_wandb_run, log_wandb_xai_metrics_results
from jaguar.utils.utils import ensure_dir, load_parquet, resolve_path
from jaguar.utils.utils_xai_similarity import SimilarityForward
from jaguar.utils.utils_xai import format_n_samples_tag, save_vec
from jaguar.logging.wandb_logger import log_wandb_xai_metrics_results
from jaguar.utils.utils_evaluation import build_eval_context
from jaguar.utils.utils_experiments import load_toml_from_path, resolve_xai_metrics_paths


# ============================================================
# Faithfulness
# ============================================================

def _get_faithfulness_config(config: dict) -> dict:
    """
    Read faithfulness metric settings from config with safe defaults.
    """
    faith_cfg = config.get("xai_metrics", {}).get("faithfulness", {})
    return {
        "steps": int(faith_cfg.get("steps", 20)),
        "baseline": str(faith_cfg.get("baseline", "zeros")),
        "use_abs": bool(faith_cfg.get("use_abs", True)),
    }

def _mask_pixels_from_order(
    x: torch.Tensor,
    order: torch.Tensor,
    k: int,
    baseline_value: float = 0.0,
) -> torch.Tensor:
    """Mask the first k flattened pixel positions in a single [C, H, W] image tensor."""
    x_masked = x.clone()
    flat = x_masked.view(x_masked.shape[0], -1)
    flat[:, order[:k]] = baseline_value
    return x_masked


def _get_pixel_order_for_class_faithfulness(
    saliency_2d: torch.Tensor,
    order_mode: str,
    rng: np.random.Generator | None = None,
    use_abs: bool = True,
) -> torch.Tensor:
    """Return a flattened pixel ranking for masking, either by saliency strength or a random permutation."""
    s = saliency_2d
    if use_abs:
        s = s.abs()

    n_pix = s.numel()

    if order_mode == "saliency":
        return torch.argsort(s.reshape(-1), descending=True)

    if order_mode == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        perm = rng.permutation(n_pix)
        return torch.as_tensor(perm, dtype=torch.long)

    raise ValueError(f"Unknown order_mode: {order_mode}")


@torch.no_grad()
def faithfulness_topk_vs_random_class(
    artifact: dict,
    resolve_sample,
    model,
    topk_frac: float = 0.1,
    use_abs: bool = True,
    random_seed: int = 0,
    baseline_value: float = 0.0,
) -> dict:
    """Compare gold-class confidence drops after masking top-saliency pixels versus random pixels of the same size."""
    rng = np.random.default_rng(random_seed)
    model.eval()

    sample_indices = artifact["sample_indices"].cpu().numpy().astype(np.int64)
    gold_idx = artifact["gold_idx"].cpu().numpy().astype(np.int64)
    sal = artifact["saliency"].cpu()  # [N,H,W]

    topk_drop = np.empty((len(sample_indices),), dtype=np.float32)
    random_drop = np.empty((len(sample_indices),), dtype=np.float32)

    rows = []

    for i in tqdm(range(len(sample_indices)), desc="Class faithfulness"):
        sample_idx = int(sample_indices[i])
        target = int(gold_idx[i])

        ds, local_idx, _ = resolve_sample(sample_idx)
        sample = ds[local_idx]
        x0 = sample["img"].to(model.device)  # [C,H,W]

        logits0 = model(x0.unsqueeze(0))
        logp0 = F.log_softmax(logits0, dim=1)[0, target].item()

        H, W = sal[i].shape
        n_pix = H * W
        k = max(1, int(topk_frac * n_pix))

        order_topk = _get_pixel_order_for_class_faithfulness(
            saliency_2d=sal[i],
            order_mode="saliency",
            rng=rng,
            use_abs=use_abs,
        )
        order_rand = _get_pixel_order_for_class_faithfulness(
            saliency_2d=sal[i],
            order_mode="random",
            rng=rng,
            use_abs=use_abs,
        )

        x_topk = _mask_pixels_from_order(
            x=x0,
            order=order_topk,
            k=k,
            baseline_value=baseline_value,
        )
        x_rand = _mask_pixels_from_order(
            x=x0,
            order=order_rand,
            k=k,
            baseline_value=baseline_value,
        )

        logp_topk = F.log_softmax(model(x_topk.unsqueeze(0)), dim=1)[0, target].item()
        logp_rand = F.log_softmax(model(x_rand.unsqueeze(0)), dim=1)[0, target].item()

        drop_topk = float(logp0 - logp_topk)
        drop_rand = float(logp0 - logp_rand)

        topk_drop[i] = drop_topk
        random_drop[i] = drop_rand

        rows.append({
            "sample_idx": sample_idx,
            "gold_idx": target,
            "orig_gold_logp": float(logp0),
            "topk_gold_logp": float(logp_topk),
            "random_gold_logp": float(logp_rand),
            "faith_topk_drop": drop_topk,
            "faith_random_drop": drop_rand,
            "faith_gap": float(drop_topk - drop_rand),
        })

    gap = topk_drop - random_drop
    details_df = pd.DataFrame(rows)

    return {
        "faith_topk_vec": topk_drop,
        "faith_random_vec": random_drop,
        "faith_gap_vec": gap,
        "faith_topk_mean": float(topk_drop.mean()),
        "faith_random_mean": float(random_drop.mean()),
        "faith_gap_mean": float(gap.mean()),
        "details_df": details_df,
        "meta": {
            "metric": "gold_class_logp_drop",
            "topk_frac": float(topk_frac),
            "use_abs": bool(use_abs),
            "random_seed": int(random_seed),
            "baseline_value": float(baseline_value),
        },
    }


def _auc_trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compute a trapezoidal AUC with compatibility for NumPy versions with or without `trapezoid`."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _build_deletion_baseline(x: torch.Tensor, baseline: str) -> torch.Tensor:
    """
    Build replacement image for deletion-based faithfulness.
    Returns a tensor with the same shape as x: [C, H, W].
    """
    if baseline == "zeros":
        return torch.zeros_like(x)

    if baseline == "blur":
        # x: [C, H, W] -> [1, C, H, W] for pooling
        xb = x.unsqueeze(0)

        # simple blur via average pooling
        # kernel/stride chosen to preserve size and keep implementation minimal
        xb = F.avg_pool2d(xb, kernel_size=21, stride=1, padding=10)

        return xb.squeeze(0)

    raise ValueError(f"Unknown baseline: {baseline}")


def _get_deletion_order(
    saliency_2d: torch.Tensor,
    order_mode: str,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Return a flattened pixel deletion order based on saliency ranking or a random permutation."""
    n_pix = saliency_2d.numel()

    if order_mode == "saliency":
        return torch.argsort(saliency_2d.reshape(-1), descending=True)

    if order_mode == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        perm = rng.permutation(n_pix)
        return torch.as_tensor(perm, dtype=torch.long)

    raise ValueError(f"Unknown order_mode: {order_mode}")


@torch.no_grad()
def faithfulness_deletion_auc_similarity(
    artifact: dict,
    resolve_sample,
    model_wrapper,
    steps: int = 20,
    baseline: str = "zeros",
    use_abs: bool = True,
    order_mode: str = "saliency",
    random_seed: int = 51,
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
    groups = defaultdict(list)
    for pos, ref_idx in enumerate(r):
        groups[int(ref_idx)].append(int(pos))

    fracs = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)
    aucs = np.empty((N,), dtype=np.float32)

    rng = np.random.default_rng(random_seed)

    for ref_idx, positions in tqdm(groups.items(), desc=f"Faithfulness ({order_mode}) refs"):
        ref_ds, ref_local_idx, _ = resolve_sample(int(ref_idx))
        ref_sample = ref_ds[ref_local_idx]
        ref_x = ref_sample["img"].unsqueeze(0).to(device)
        ref_emb = model_wrapper.get_embeddings_tensor(ref_x).squeeze(0)

        sim_model = SimilarityForward(model_wrapper, ref_emb, maximize=True).to(device).eval()

        for pos in positions:
            qi = int(q[pos])
            query_ds, query_local_idx, _ = resolve_sample(int(qi))
            x0 = query_ds[query_local_idx]["img"].to(device)
            x_base = _build_deletion_baseline(x0, baseline)

            order = _get_deletion_order(
                saliency_2d=sal[pos],
                order_mode=order_mode,
                rng=rng,
            )

            sims = []
            for t in range(steps + 1):
                k = int(fracs[t] * (H * W))
                x = x0.clone()
                if k > 0:
                    flat = x.view(3, -1)
                    flat[:, order[:k]] = x_base.view(3, -1)[:, order[:k]]
                s = sim_model(x.unsqueeze(0)).detach().float().cpu().item()
                sims.append(s)

            sims = np.asarray(sims, dtype=np.float32)
            if sims[0] != 0:
                sims_norm = sims / sims[0]
            else:
                sims_norm = sims

            aucs[pos] = _auc_trapz(sims_norm, fracs)

    return {
        "faith_vec": aucs,
        "faith_mean": float(np.mean(aucs)),
        "faith_median": float(np.median(aucs)),
        "faith_std": float(np.std(aucs)),
        "meta": {
            "metric": "deletion_auc",
            "steps": int(steps),
            "baseline": baseline,
            "use_abs": bool(use_abs),
            "order_mode": order_mode,
            "random_seed": int(random_seed),
        }
    }

def run_faithfulness_metric(
    artifact,
    resolve_sample,
    model_wrapper,
    config: dict,
) -> dict:
    """Run the faithfulness metric with saliency-based and random deletion, then summarize both scores and their gap."""
    faith_cfg = _get_faithfulness_config(config)

    res_topk = faithfulness_deletion_auc_similarity(
        artifact=artifact,
        resolve_sample=resolve_sample,
        model_wrapper=model_wrapper,
        steps=faith_cfg["steps"],
        baseline=faith_cfg["baseline"],
        use_abs=faith_cfg["use_abs"],
        order_mode="saliency",
        random_seed=0,
    )
    gc.collect(); torch.cuda.empty_cache()

    res_random = faithfulness_deletion_auc_similarity(
        artifact=artifact,
        resolve_sample=resolve_sample,
        model_wrapper=model_wrapper,
        steps=faith_cfg["steps"],
        baseline=faith_cfg["baseline"],
        use_abs=faith_cfg["use_abs"],
        order_mode="random",
        random_seed=0,
    )
    gc.collect(); torch.cuda.empty_cache()

    gap_vec = res_random["faith_vec"] - res_topk["faith_vec"]

    return {
        "faith_topk_vec": res_topk["faith_vec"],
        "faith_topk_mean": float(np.mean(res_topk["faith_vec"])),
        "faith_topk_median": float(np.median(res_topk["faith_vec"])),
        "faith_topk_std": float(np.std(res_topk["faith_vec"])),

        "faith_random_vec": res_random["faith_vec"],
        "faith_random_mean": float(np.mean(res_random["faith_vec"])),
        "faith_random_median": float(np.median(res_random["faith_vec"])),
        "faith_random_std": float(np.std(res_random["faith_vec"])),

        "faith_gap_vec": gap_vec,
        "faith_gap_mean": float(np.mean(gap_vec)),
        "faith_gap_median": float(np.median(gap_vec)),
        "faith_gap_std": float(np.std(gap_vec)),

        "meta": {
            "metric": "deletion_auc_with_random_control",
            "steps": faith_cfg["steps"],
            "baseline": faith_cfg["baseline"],
            "use_abs": faith_cfg["use_abs"],
        }
    }


def run_faithfulness_metric_class(
    artifact: dict,
    resolve_sample,
    model,
    config: dict,
) -> dict:
    """
    Wrapper matching the similarity metric style.
    """
    faith_cfg = config.get("xai_metrics", {}).get("faithfulness", {})

    res = faithfulness_topk_vs_random_class(
        artifact=artifact,
        resolve_sample=resolve_sample,
        model=model,
        topk_frac=float(faith_cfg.get("topk_frac", 0.1)),
        use_abs=bool(faith_cfg.get("use_abs", True)),
        random_seed=int(faith_cfg.get("random_seed", 0)),
        baseline_value=float(faith_cfg.get("baseline_value", 0.0)),
    )

    return res

# ============================================================
# Sanity / Robustness
# ============================================================

def _spearman_corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a Spearman-style correlation on two flattened arrays via rank-transformed cosine similarity."""
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
    resolve_sample,
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
    sal0 = artifact["saliency"].cpu().numpy()  # [N,H,W]
    N = sal0.shape[0]

    mw_rand = copy.deepcopy(model_wrapper)
    _randomize_model_params_(mw_rand.model)
    mw_rand.eval()

    art_rand = compute_saliency_fn(resolve_sample, mw_rand, ref_df_pair, cfg)
    sal1 = art_rand["saliency"].cpu().numpy()  # [N,H,W]

    cors = np.empty((N,), dtype=np.float32)
    for i in range(N):
        cors[i] = _spearman_corr_flat(sal0[i].reshape(-1), sal1[i].reshape(-1))

    return {
        "sanity_vec": cors,
        "sanity_mean": float(np.mean(cors)),
        "sanity_median": float(np.median(cors)),
        "sanity_std": float(np.std(cors)),
        "randomized_artifact": art_rand,
        "meta": {"metric": "parameter_randomization_spearman"},
    }


def run_sanity_metric(
    explainer_name: str,
    artifact: Dict[str, Any],
    resolve_sample,
    model_wrapper,
    ref_df_pair,
    cfg,
):
    """Run the parameter-randomization sanity metric for the requested explainer and return the resulting summary."""
    if explainer_name == "IG":
        compute_saliency_fn = compute_saliency_ig_for_pair_type
    elif explainer_name == "GradCAM":
        compute_saliency_fn = compute_saliency_gradcam_for_pair_type
    else:
        raise NotImplementedError(f"Sanity metric not implemented for explainer={explainer_name}")

    if compute_saliency_fn is not None:
        res_sanity = sanity_parameter_randomization(
            artifact=artifact,
            resolve_sample=resolve_sample,
            model_wrapper=model_wrapper, 
            compute_saliency_fn=compute_saliency_fn, 
            ref_df_pair=ref_df_pair, 
            cfg=cfg
        )
        gc.collect(); torch.cuda.empty_cache()

    return res_sanity


def run_sanity_metric_class(
    explainer_name: str,
    artifact: dict,
    resolve_sample,
    model,
    cfg,
):
    """
    Sanity metric for class attribution.
    Recomputes class saliency after parameter randomization and compares it
    to the trained saliency via Spearman correlation.
    """
    if explainer_name == "IG":
        compute_saliency_fn = compute_saliency_ig_class   # provide this
    elif explainer_name == "GradCAM":
        compute_saliency_fn = compute_saliency_gradcam_class   # provide this
    else:
        raise NotImplementedError(f"Sanity metric not implemented for explainer={explainer_name}")

    sal0 = artifact["saliency"].cpu().numpy()  # [N,H,W]
    n = sal0.shape[0]

    model_rand = copy.deepcopy(model)
    _randomize_model_params_(model_rand)
    model_rand.eval()

    art_rand = compute_saliency_fn(
        resolve_sample=resolve_sample,
        model=model_rand,
        artifact=artifact,   # or cfg/sample metadata if your compute fn needs that
        cfg=cfg,
    )
    sal1 = art_rand["saliency"].cpu().numpy()

    cors = np.empty((n,), dtype=np.float32)
    for i in range(n):
        cors[i] = _spearman_corr_flat(sal0[i].reshape(-1), sal1[i].reshape(-1))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "sanity_vec": cors,
        "sanity_mean": float(np.mean(cors)),
        "sanity_median": float(np.median(cors)),
        "sanity_std": float(np.std(cors)),
        "randomized_artifact": art_rand,
        "meta": {"metric": "parameter_randomization_spearman"},
    }


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


def run_xai_metrics(
    items: list[dict],
    metrics_path: Path,
    randomized_root: Path,
    run_sanity_fn: Callable[..., dict],
    run_faithfulness_fn: Callable[..., dict],
    run_complexity_fn: Callable[..., dict] | None,
    build_sanity_kwargs_fn: Callable[[dict], dict],
    build_faithfulness_kwargs_fn: Callable[[dict], dict],
    build_complexity_kwargs_fn: Callable[[dict], dict] | None,
    summary_filename: str,
    slice_key: str,
) -> pd.DataFrame:
    """
    Generic XAI metric runner for both similarity and class attribution.
    """
    ensure_dir(metrics_path)
    rows = []

    for item in items:
        explainer_name = item["explainer"]
        slice_value = item[slice_key]
        artifact = item["artifact"]

        sanity = run_sanity_fn(**build_sanity_kwargs_fn(item))
        faith = run_faithfulness_fn(**build_faithfulness_kwargs_fn(item))

        complexity = None
        if run_complexity_fn is not None and build_complexity_kwargs_fn is not None:
            complexity = run_complexity_fn(**build_complexity_kwargs_fn(item))

        rand_sal_path = randomized_root / explainer_name / f"sal__{slice_value}.pt"
        ensure_dir(rand_sal_path.parent)
        torch.save(sanity["randomized_artifact"], rand_sal_path)

        row = {
            "explainer": explainer_name,
            slice_key: slice_value,

            "sanity_mean": sanity["sanity_mean"],
            "sanity_median": sanity["sanity_median"],
            "sanity_std": sanity["sanity_std"],
            "sanity_metric": sanity["meta"]["metric"],
            "sanity_randomized_sal_path": str(rand_sal_path),
            "sanity_vec_path": save_vec(metrics_path, "sanity", explainer_name, slice_value, sanity["sanity_vec"]),

            "faith_metric": faith["meta"]["metric"],
        }

        if "faith_steps" in faith["meta"]:
            row["faith_steps"] = faith["meta"]["steps"]
        if "baseline" in faith["meta"]:
            row["faith_baseline"] = faith["meta"]["baseline"]
        if "use_abs" in faith["meta"]:
            row["faith_use_abs"] = faith["meta"]["use_abs"]

        if "faith_topk_vec" in faith:
            row["faith_topk_mean"] = faith["faith_topk_mean"]
            row["faith_topk_median"] = faith.get("faith_topk_median")
            row["faith_topk_std"] = faith.get("faith_topk_std")
            row["faith_topk_vec_path"] = save_vec(metrics_path, "faith_topk", explainer_name, slice_value, faith["faith_topk_vec"])

        if "faith_random_vec" in faith:
            row["faith_random_mean"] = faith["faith_random_mean"]
            row["faith_random_median"] = faith.get("faith_random_median")
            row["faith_random_std"] = faith.get("faith_random_std")
            row["faith_random_vec_path"] = save_vec(metrics_path, "faith_random", explainer_name, slice_value, faith["faith_random_vec"])

        if "faith_gap_vec" in faith:
            row["faith_gap_mean"] = faith["faith_gap_mean"]
            row["faith_gap_median"] = faith.get("faith_gap_median")
            row["faith_gap_std"] = faith.get("faith_gap_std")
            row["faith_gap_vec_path"] = save_vec(metrics_path, "faith_gap", explainer_name, slice_value, faith["faith_gap_vec"])

        if complexity is not None:
            row["complexity_mean"] = complexity["complexity_mean"]
            row["complexity_median"] = complexity["complexity_median"]
            row["complexity_std"] = complexity["complexity_std"]
            row["complexity_metric"] = complexity["meta"]["metric"]
            row["complexity_vec_path"] = save_vec(metrics_path, "complexity", explainer_name, slice_value, complexity["complexity_vec"])

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(metrics_path / summary_filename, index=False)
    return summary_df


def run_xai_similarity_metrics(config: dict, cfg) -> pd.DataFrame:
    """Load saved pairwise XAI artifacts, run the configured evaluation metrics, and return the summary table."""
    
    checkpoint_dir = PATHS.checkpoints / config["evaluation"]["checkpoint_dir"]
    train_config = load_toml_from_path(checkpoint_dir / "config_leaderboard_exp.toml")

    ctx = build_eval_context(config, train_config, checkpoint_dir, eval_val_setting="original")

    explainer_names = list(cfg.explainer_names)
    run_root, run_root_write, metrics_path, randomized_root = resolve_xai_metrics_paths(config)

    run = init_wandb_run(
        config=config,
        run_dir=metrics_path,
        exp_name=config["evaluation"]["experiment_name"],
        experiment_group=config.get("output", {}).get("experiment_group"),
        job_type="explain_eval",
    )

    # Load the same dataset + model backbone used for scoring similarity 
    # (needed for faithfulness and sanity recomputation).
    model_wrapper = ctx.model.backbone_wrapper
    model_wrapper.eval()
    resolve_sample = build_emb_row_sample_resolver(ctx)

    # Load the mined reference pairs so metrics are aligned to the exact evaluated pairs.
    n_tag = format_n_samples_tag(cfg.n_samples)
    out_path = run_root / f"refs_n{n_tag}.parquet"
    ref_df = load_parquet(out_path)

    
    items = []
    for explainer_name in explainer_names:
        for pair_type in cfg.pair_types:
            ref_df_pair = ref_df[ref_df["pair_type"] == pair_type].reset_index(drop=True)
            sal_path = run_root / "explanations" / explainer_name / f"sal__{pair_type}.pt"
            artifact = torch.load(sal_path, map_location="cpu")

            items.append({
                "explainer": explainer_name,
                "pair_type": pair_type,
                "artifact": artifact,
                "resolve_sample": resolve_sample,
                "model_wrapper": model_wrapper,
                "ref_df_pair": ref_df_pair,
                "cfg": cfg,
                "config": config,
            })

    out = run_xai_metrics(
        items=items,
        metrics_path=metrics_path,
        randomized_root=randomized_root,
        run_sanity_fn=run_sanity_metric,
        run_faithfulness_fn=run_faithfulness_metric,
        run_complexity_fn=run_complexity_metric,
        build_sanity_kwargs_fn=lambda item: dict(
            explainer_name=item["explainer"],
            artifact=item["artifact"],
            resolve_sample=item["resolve_sample"],
            model_wrapper=item["model_wrapper"],
            ref_df_pair=item["ref_df_pair"],
            cfg=item["cfg"],
        ),
        build_faithfulness_kwargs_fn=lambda item: dict(
            artifact=item["artifact"],
            resolve_sample=item["resolve_sample"],
            model_wrapper=item["model_wrapper"],
            config=item["config"],
        ),
        build_complexity_kwargs_fn=lambda item: dict(
            artifact=item["artifact"],
            model_wrapper=item["model_wrapper"],
        ),
        summary_filename="xai_summary_metrics.csv",
        slice_key="pair_type",
    )

    artifacts = {
        "run_dir": metrics_path,
        "config": config,
    }
    save_requested_outputs(config, artifacts)

    log_wandb_xai_metrics_results(run, out)
    if run is not None:
        run.finish()
        
    return out

        
def run_xai_class_metrics(config: dict, cfg) -> pd.DataFrame:
    """Load saved class-attribution artifacts, run the configured evaluation metrics, and return the summary table."""
    run_root, run_root_write, metrics_path, randomized_root = resolve_xai_metrics_paths(config)

    run = init_wandb_run(
        config=config,
        run_dir=metrics_path,
        exp_name=config["evaluation"]["experiment_name"],
        experiment_group=config.get("output", {}).get("experiment_group"),
        job_type="explain_eval",
    )

    items = []
    for explainer_name in cfg.explainer_names:
        for group_name in cfg.groups:   # e.g. ("all", "orig_correct", "orig_wrong")
            sal_path = run_root / "explanations" / explainer_name / f"sal__{group_name}.pt"
            artifact = torch.load(sal_path, map_location="cpu")

            items.append({
                "explainer": explainer_name,
                "group": group_name,
                "artifact": artifact,
                "resolve_sample": resolve_sample,
                "model": model,
                "cfg": cfg,
                "config": config,
            })

    out = run_xai_metrics(
        items=items,
        metrics_path=metrics_path,
        randomized_root=randomized_root,
        run_sanity_fn=run_sanity_metric_class,          # class-specific wrapper
        run_faithfulness_fn=faithfulness_topk_vs_random_class,  # class-specific wrapper
        run_complexity_fn=run_complexity_metric,
        build_sanity_kwargs_fn=lambda item: dict(
            explainer_name=item["explainer"],
            artifact=item["artifact"],
            resolve_sample=item["resolve_sample"],
            model=item["model"],
            cfg=item["cfg"],
        ),
        build_faithfulness_kwargs_fn=lambda item: dict(
            artifact=item["artifact"],
            resolve_sample=item["resolve_sample"],
            model=item["model"],
            config=item["config"],
        ),
        build_complexity_kwargs_fn=lambda item: dict(
            artifact=item["artifact"],
            model_wrapper=item["model"],   # only if compatible
        ),
        summary_filename="xai_class_summary_metrics.csv",
        slice_key="group",
    )

    log_wandb_xai_metrics_results(run, out)
    if run is not None:
        run.finish()
        
    return out