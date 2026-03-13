"""
XAI Pair Explanations for Jaguar Re-ID.

Project role:
- Selects a small, reproducible subset of curated validation queries.
- Mines informative query-reference pairs from a curated gallery.
- Generates pair-level saliency maps (IG, GradCAM) for qualitative analysis.

Procedure:
- Use curated validation queries and a curated train+val gallery.
- Mine reference pairs from retrieval rankings (e.g., easy_pos, hard_neg, hard_pos).
- Exclude trivial burst matches during pair selection.
- Compute saliency maps that explain query-reference similarity.
- Save explanations as reusable analysis artifacts.

Purpose:
- Support qualitative debugging and reporting of retrieval behavior.
- Analyze which image regions drive similarity decisions for different pair types.
"""
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from captum.attr import IntegratedGradients
from pytorch_grad_cam import GradCAM

from typing import Dict, Any, Sequence, Tuple, List

from jaguar.config import PATHS 
from jaguar.utils.utils_models import load_or_extract_jaguarid_embeddings
from jaguar.utils.utils import ensure_dir, save_parquet
from jaguar.models.foundation_models import FoundationModelWrapper  
from jaguar.utils.utils_xai import format_n_samples_tag, ig_saliency_batched_similarity, CosineSimilarityTarget, EmbeddingForwardWrapper, SimilarityForward, find_module_name, get_val_query_indices, resolve_n_samples  
from jaguar.utils.utils_evaluation import RetrievalState, build_eval_context, build_query_gallery_retrieval_state, get_ranked_candidates_for_query, map_emb_rows_to_local_indices
from jaguar.logging.wandb_logger import init_wandb_run, log_wandb_xai_similarity_results
from jaguar.utils.utils_experiments import load_toml_from_path

# ============================================================
# Config: defines one reproducible XAI run
# ============================================================
@dataclass(frozen=True)
class XAIConfig:
    """
    Minimal run configuration
    """
    dataset_name: str = "jaguar_init"
    split_name: str = "val"
    n_samples: int | str | None = 100
    seed: int = 51

    explainer_names: Tuple[str, ...] = ("GradCAM", "IG")

    # IG hyperparameters (tradeoff: speed vs smoother attributions)
    ig_steps: int = 10
    ig_internal_bs: int = 32
    ig_batch_size: int = 32

    # Retrieval pair taxonomy
    pair_types: Tuple[str, ...] = ("easy_pos", "hard_neg", "hard_pos")

    # Output root; each run creates its own subfolder under here
    out_root: Path = Path()


# ============================================================
# Mine references for pair types 
# ============================================================
def mine_reference_pairs_from_retrieval(
    retrieval: RetrievalState,
    out_root: Path,
    pair_types: Sequence[str],
) -> pd.DataFrame:
    """
    Mines XAI reference pairs directly from a prepared query-gallery retrieval state.

    This keeps XAI pair selection consistent with the shared evaluation logic:
    ranking, self-exclusion, and same-burst exclusion are all inherited from
    get_ranked_candidates_for_query(...).

    Pair taxonomy:
    - easy_pos: first valid same-id match in the ranked gallery
    - hard_neg: first valid different-id match in the ranked gallery
    - hard_pos: last valid same-id match in the ranked gallery
    """
    n_queries = len(retrieval.q_global)
    out_path = out_root / f"refs_n{n_queries}.parquet"

    if out_path.exists():
        print(f"[Info] Loading existing refs from {out_path}")
        return pd.read_parquet(out_path)

    rows = []

    for i in tqdm(range(n_queries), desc="Mining refs from retrieval"):
        q_idx_global, q_label, ranked_candidates = get_ranked_candidates_for_query(retrieval, i)

        found_pairs = {pt: False for pt in pair_types}
        hard_pos_candidate = None
        need_tail_scan = "hard_pos" in pair_types

        for cand in ranked_candidates:
            g_idx_global = int(cand["gallery_global_idx"])
            g_label = cand["gallery_label"]
            is_same_id = bool(cand["is_same_id"])
            pair_sim = float(cand["sim"])
            rank_in_gallery = int(cand["rank_in_gallery"])

            if is_same_id and "easy_pos" in pair_types and not found_pairs["easy_pos"]:
                rows.append({
                    "pair_type": "easy_pos",
                    "query_idx": q_idx_global,
                    "query_label": q_label,
                    "ref_idx": g_idx_global,
                    "ref_label": g_label,
                    "pair_sim": pair_sim,
                    "rank_in_gallery": rank_in_gallery,
                    "is_same_id": True,
                })
                found_pairs["easy_pos"] = True

            if (not is_same_id) and "hard_neg" in pair_types and not found_pairs["hard_neg"]:
                rows.append({
                    "pair_type": "hard_neg",
                    "query_idx": q_idx_global,
                    "query_label": q_label,
                    "ref_idx": g_idx_global,
                    "ref_label": g_label,
                    "pair_sim": pair_sim,
                    "rank_in_gallery": rank_in_gallery,
                    "is_same_id": False,
                })
                found_pairs["hard_neg"] = True

            if is_same_id and "hard_pos" in pair_types:
                hard_pos_candidate = {
                    "pair_type": "hard_pos",
                    "query_idx": q_idx_global,
                    "query_label": q_label,
                    "ref_idx": g_idx_global,
                    "ref_label": g_label,
                    "pair_sim": pair_sim,
                    "rank_in_gallery": rank_in_gallery,
                    "is_same_id": True,
                }

            if not need_tail_scan:
                done = all(found_pairs[pt] for pt in pair_types)
                if done:
                    break

        if "hard_pos" in pair_types and hard_pos_candidate is not None:
            rows.append(hard_pos_candidate)

    ref_df = pd.DataFrame(rows)

    expected_max = n_queries * len(pair_types)
    if len(ref_df) < expected_max:
        print(
            f"[Warning] Mined {len(ref_df)} pairs, expected up to {expected_max}. "
            "Some queries likely have no valid positive after exclusions."
        )

    ensure_dir(out_path.parent)
    save_parquet(out_path, ref_df)
    return ref_df


# ============================================================
# Compute saliency maps for a given pair_type + explainer
# ============================================================
def build_emb_row_sample_resolver(ctx):
    """
    Builds a resolver from global emb_row ids to the correct dataset sample
    (train or val) and its local index.
    """
    train_map = {
        int(emb_row): int(local_idx)
        for local_idx, emb_row in enumerate(ctx.train_local_to_emb_row)
    }
    val_map = {
        int(emb_row): int(local_idx)
        for local_idx, emb_row in enumerate(ctx.val_local_to_emb_row)
    }

    def resolve_sample(emb_row: int):
        emb_row = int(emb_row)

        if emb_row in train_map:
            local_idx = train_map[emb_row]
            return ctx.train_ds, local_idx, "train"

        if emb_row in val_map:
            local_idx = val_map[emb_row]
            return ctx.val_ds, local_idx, "val"

        raise KeyError(f"emb_row {emb_row} not found in train or val resolver.")

    return resolve_sample


def compute_saliency_ig_for_pair_type(
    resolve_sample,
    model_wrapper: FoundationModelWrapper,
    ref_df_pair: pd.DataFrame,
    cfg: XAIConfig,
) -> Dict[str, Any]:
    """
    Computes IG saliency maps for the pairs in ref_df_pair.
    IG here explains similarity: “which pixels in the *query* increase cosine similarity to this fixed 
    reference embedding?” (pair explanation, not class explanation).
    """
    # Align arrays
    query = ref_df_pair["query_idx"].astype(int).to_numpy()
    ref = ref_df_pair["ref_idx"].astype(int).to_numpy()
    pair_sims = ref_df_pair["pair_sim"].astype(float).to_numpy()

    # Memory Efficiency: group by reference index to reuse reference embedding (SimilarityForward + IG object)
    groups = defaultdict(list)
    for pos, ref_idx in enumerate(ref):
        groups[int(ref_idx)].append(pos)

    saliency_out: List[torch.Tensor] = [None] * len(query)
    x_query_out: List[torch.Tensor] = [None] * len(query)

    model_wrapper.eval()

    # iterate over all references
    for ref_idx, positions in tqdm(groups.items(), desc="IG refs", total=len(groups)):
        # get reference embedding
        ref_ds, ref_local_idx, _ = resolve_sample(int(ref_idx))
        ref_sample = ref_ds[ref_local_idx]
        ref_tensor = ref_sample["img"].unsqueeze(0).to(model_wrapper.device)
        with torch.no_grad():
            ref_emb = model_wrapper.get_embeddings_tensor(ref_tensor).squeeze(0)  # [D]

        # similarity forward + IG for this reference
        sim_model = SimilarityForward(model_wrapper, ref_emb, maximize=True).to(model_wrapper.device)
        ig = IntegratedGradients(sim_model)

        # gather queries for this ref-group
        idx_of_queries_matching_ref = query[np.asarray(positions, dtype=np.int64)]
        x_batch = []
        for idx in idx_of_queries_matching_ref:
            query_ds, query_local_idx, _ = resolve_sample(int(idx))
            sample = query_ds[query_local_idx]
            x = sample["img"]
            x_batch.append(x)
        X_batch = torch.stack(x_batch, dim=0)  # [B,3,H,W] CPU

        sal_group = ig_saliency_batched_similarity(
            X_batch,
            explainer=ig,
            device=model_wrapper.device,
            steps=cfg.ig_steps,
            internal_bs=cfg.ig_internal_bs,
            batch_size=cfg.ig_batch_size,
        )  # [B,H,W] CPU

        # maps back to global array
        for local_k, global_pos in enumerate(positions):
            saliency_out[global_pos] = sal_group[local_k]
            x_query_out[global_pos] = X_batch[local_k].cpu()

    artifact = {
        "meta": {
            "dataset_name": cfg.dataset_name,
            "split": cfg.split_name,
            "model_name": model_wrapper.name,
            "explainer": "IG",
            "pair_type": str(ref_df_pair["pair_type"].iloc[0]),
            "similarity_target": "cosine",
            "maximize": True,
            "ig_steps": cfg.ig_steps,
            "ig_internal_bs": cfg.ig_internal_bs,
            "ig_batch_size": cfg.ig_batch_size,
            "baseline_type": "zero_in_normalized_space",
            "ig_method": "riemann_trapezoid",
            "saliency_reduction": "sum_channels_signed"
        },
        "query_indices": torch.tensor(query, dtype=torch.long),
        "ref_indices": torch.tensor(ref, dtype=torch.long),
        "pair_sims": torch.tensor(pair_sims, dtype=torch.float32),
        "saliency": torch.stack(saliency_out, dim=0),  # [N,H,W]
    }
    return artifact


def compute_saliency_gradcam_for_pair_type(
    resolve_sample,
    model_wrapper: FoundationModelWrapper,
    ref_df_pair: pd.DataFrame,
    cfg: XAIConfig,
) -> Dict[str, Any]:
    """
    Computes GradCam saliency maps for the pairs in ref_df_pair.
    GradCAM target is “cosine similarity to the reference embedding”.
    This makes GradCAM comparable to IG: both explain the *pair similarity* decision.
    """

    query = ref_df_pair["query_idx"].astype(int).to_numpy()
    ref = ref_df_pair["ref_idx"].astype(int).to_numpy()
    pair_sims = ref_df_pair["pair_sim"].astype(float).to_numpy()

    device = model_wrapper.device
    model_wrapper.eval()

    # GradCAM setup
    # Target layer is model-dependent; FoundationModelWrapper provides the chosen layer
    # so results are consistent within each backbone.
    target_layers, reshape_transform = model_wrapper.get_grad_cam_config()
    cam_model = EmbeddingForwardWrapper(model_wrapper.model).to(device).eval()

    cam = GradCAM(
        model=cam_model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    saliency_out: List[torch.Tensor] = [None] * len(query)
    x_query_out: List[torch.Tensor] = [None] * len(query)

    for k in tqdm(range(len(query)), desc="GradCAM pairs"):
        qi = int(query[k])
        ri = int(ref[k])

        q_ds, q_local_idx, _ = resolve_sample(qi)
        r_ds, r_local_idx, _ = resolve_sample(ri)

        q_sample = q_ds[q_local_idx]
        r_sample = r_ds[r_local_idx]

        q_x = q_sample["img"].unsqueeze(0).to(device)  # [1,3,H,W]
        r_x = r_sample["img"].unsqueeze(0).to(device)  # [1,3,H,W]

        q_x.requires_grad_(True)

        with torch.no_grad():
            ref_emb = model_wrapper.get_embeddings_tensor(r_x)  # [1,D]

        targets = [CosineSimilarityTarget(ref_emb, maximize=True)]

        # enable gradient
        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=q_x, targets=targets)

        grayscale_cam = cam(input_tensor=q_x, targets=targets)  # np array [1,h,w]
        cam_t = torch.as_tensor(grayscale_cam[0], dtype=torch.float32)  # [h,w] CPU

        # resize to input size if needed
        H, W = q_x.shape[-2], q_x.shape[-1]
        if cam_t.shape != (H, W):
            cam_t = torch.nn.functional.interpolate(
                cam_t[None, None, ...], size=(H, W), mode="bilinear", align_corners=False
            )[0, 0]  # [H,W]

        saliency_out[k] = cam_t
        x_query_out[k] = q_x[0].detach().cpu()

    target_layers, reshape_transform = model_wrapper.get_grad_cam_config()
    layer0 = target_layers[0]
    layer_name = find_module_name(model_wrapper.model, layer0)

    artifact = {
        "meta": {
            "dataset_name": cfg.dataset_name,
            "split": cfg.split_name,
            "model_name": model_wrapper.name,
            "explainer": "GradCAM",
            "pair_type": str(ref_df_pair["pair_type"].iloc[0]),
            "similarity_target": "cosine",
            "maximize": True,
            "saliency_reduction": "gradcam_relu",
            "saliency_normalized": False, 
            "gradcam_target_layer_name": layer_name,
            "gradcam_target_layer_type": layer0.__class__.__name__,
            "gradcam_reshape_transform": None if reshape_transform is None else getattr(reshape_transform, "__name__", str(reshape_transform)),
        },
        "query_indices": torch.tensor(query.copy(), dtype=torch.long),
        "ref_indices": torch.tensor(ref.copy(), dtype=torch.long),
        "pair_sims": torch.tensor(pair_sims.copy(), dtype=torch.float32),
        "saliency": torch.stack(saliency_out, dim=0),  # [N,H,W]
    }
    return artifact

# ============================================================
# Load-or-create artifacts per (explainer, pair_type)
# ============================================================

def load_or_create_saliency_maps(
    resolve_sample,
    run_root: Path,
    model_wrapper: FoundationModelWrapper,
    ref_df: pd.DataFrame,
    cfg: XAIConfig,
    explainer_names: Sequence[str],
) -> Dict[Tuple[str, str], Path]:
    """
    create one artifact per pair_type (and per explainer).
    """
    out: Dict[Tuple[str, str], Path] = {}
    
    for explainer_name in explainer_names:
        for pair_type in cfg.pair_types:
            sal_path = run_root / "explanations" / f"{explainer_name}" / f"sal__{pair_type}.pt"
            ensure_dir(sal_path.parent)
            
            if sal_path.exists():
                out[(explainer_name, pair_type)] = sal_path
                continue
            else:
                ref_df_pair = ref_df[ref_df["pair_type"] == pair_type].reset_index(drop=True)
                if len(ref_df_pair) == 0:
                    raise RuntimeError(f"No rows for pair_type={pair_type}")
                if explainer_name == "IG":
                    artifact = compute_saliency_ig_for_pair_type(resolve_sample, model_wrapper, ref_df_pair, cfg)
                elif explainer_name == "GradCAM":
                    artifact = compute_saliency_gradcam_for_pair_type(resolve_sample, model_wrapper, ref_df_pair, cfg)
                else:
                    raise NotImplementedError(
                        f"Explainer '{explainer_name}' not implemented here yet."
                    )

                torch.save(artifact, sal_path)
                out[(explainer_name, pair_type)] = sal_path

    return out



def build_curated_train_val_gallery(
    split_df: pd.DataFrame,
    ctx,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds one curated train+val gallery and returns:
    - gallery_embeddings
    - gallery_labels
    - gallery_global_indices
    """
    gallery_df = split_df[
        split_df["split_final"].isin(["train", "val"])
        & split_df["keep_curated"].fillna(False).astype(bool)
    ].copy()

    train_gallery_emb_rows = gallery_df.loc[
        gallery_df["split_final"] == "train", "emb_row"
    ].astype(np.int64).to_numpy()

    val_gallery_emb_rows = gallery_df.loc[
        gallery_df["split_final"] == "val", "emb_row"
    ].astype(np.int64).to_numpy()

    train_gallery_local_idx = map_emb_rows_to_local_indices(
        train_gallery_emb_rows,
        ctx.train_local_to_emb_row,
    )
    val_gallery_local_idx = map_emb_rows_to_local_indices(
        val_gallery_emb_rows,
        ctx.val_local_to_emb_row,
    )

    gallery_embeddings = np.concatenate(
        [
            train_embeddings[train_gallery_local_idx],
            val_embeddings[val_gallery_local_idx],
        ],
        axis=0,
    )

    gallery_labels = np.concatenate(
        [
            np.asarray(ctx.train_ds.labels)[train_gallery_local_idx],
            np.asarray(ctx.val_ds.labels)[val_gallery_local_idx],
        ],
        axis=0,
    )

    gallery_global_indices = np.concatenate(
        [train_gallery_emb_rows, val_gallery_emb_rows],
        axis=0,
    )

    return gallery_embeddings, gallery_labels, gallery_global_indices


def run_xai(config, cfg: XAIConfig) -> Dict[Tuple[str, str], Path]:

    checkpoint_dir = PATHS.checkpoints / config["evaluation"]["checkpoint_dir"]
    train_config = load_toml_from_path(checkpoint_dir / "config_leaderboard_exp.toml")

    ctx = build_eval_context(config, train_config, checkpoint_dir, eval_val_setting="original")

    n_tag = format_n_samples_tag(cfg.n_samples)
    run_root = cfg.out_root / f"{ctx.model.backbone_wrapper.name}__{cfg.split_name}__n{n_tag}__seed{cfg.seed}"
    ensure_dir(run_root)
    split_df = ctx.split_df
    explainer_names = cfg.explainer_names
    #exp_name = config["training"]["experiment_name"]

    run = init_wandb_run(
        config=config,
        run_dir=cfg.out_root,
        exp_name=config["evaluation"]["experiment_name"],
        experiment_group=config.get("output", {}).get("experiment_group"),
        job_type="explain",
    )

    train_embeddings = load_or_extract_jaguarid_embeddings(
        model=ctx.model,
        torch_ds=ctx.train_ds,
        split="train",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix=f"xai_{ctx.model.backbone_wrapper.name}_train"
    )

    val_embeddings = load_or_extract_jaguarid_embeddings(
        model=ctx.model,
        torch_ds=ctx.val_ds,
        split="val",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix=f"xai_{ctx.model.backbone_wrapper.name}_val"
    )

    # Choose deterministic validation queries (small subset N for explainability runs)
    resolved_n_samples = resolve_n_samples(cfg.n_samples)
    val_query_emb_rows = get_val_query_indices(
        split_df=split_df,
        out_root=run_root,
        n_samples=resolved_n_samples,
        seed=cfg.seed,
    )
    val_query_local_idx = map_emb_rows_to_local_indices(val_query_emb_rows, ctx.val_local_to_emb_row)

    query_embeddings = val_embeddings[val_query_local_idx]
    query_labels = np.asarray(ctx.val_ds.labels)[val_query_local_idx]
    query_global_indices = val_query_emb_rows
    
    # Define a curated gallery pool (bigger pool => more meaningful negatives)
    gallery_embeddings, gallery_labels, gallery_global_indices = build_curated_train_val_gallery(
        split_df=split_df,
        ctx=ctx,
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
    )

    retrieval = build_query_gallery_retrieval_state(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        query_global_indices=query_global_indices,
        gallery_global_indices=gallery_global_indices,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        split_df=split_df,
    )

    # Mine references from the curated train+val gallery
    ref_df = mine_reference_pairs_from_retrieval(
        retrieval=retrieval,
        out_root=run_root,
        pair_types=cfg.pair_types,
    )

    print(f"[Result] Mined {len(ref_df)} pairs.")
    print(ref_df.groupby("pair_type")["pair_sim"].describe())

    resolve_sample = build_emb_row_sample_resolver(ctx)

    paths = load_or_create_saliency_maps(
        resolve_sample=resolve_sample,
        run_root=run_root,
        model_wrapper=ctx.model.backbone_wrapper,
        ref_df=ref_df,
        cfg=cfg,
        explainer_names=explainer_names,
    )

    log_wandb_xai_similarity_results(run, ref_df, explainer_names, cfg.pair_types)
    if run is not None:
        run.finish()

    return paths