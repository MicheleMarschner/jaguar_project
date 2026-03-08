"""
XAI pair-explanations for the Jaguar Re-ID project.

Purpose in the project:
- picks a small, reproducible subset of curated validation queries
- mines “meaningful” reference pairs from a curated gallery (easy_pos, hard_neg, ...)
- generates saliency maps explaining *why the model considers query ↔ reference similar*
  (IG + GradCAM) and saves them as reusable artifacts

This script defines *analysis artifacts* for qualitative debugging and reporting,
not training or evaluation metrics.

Important project assumptions:
1) emb_row alignment:
   - split_df['emb_row'] indexes into torch_ds AND into the embeddings returned by load_or_extract_embeddings.
   - i.e., embeddings[emb_row] corresponds to torch_ds[emb_row].
2) Curation contract:
   - downstream analysis uses keep_curated == True to avoid dropped/duplicate-heavy samples.
3) Burst leakage avoidance:
   - while mining references, we skip pairs from the same burst_group_id (trivial near-duplicates).
4) Reference pool train + val: 
   - we keep queries from val dataset, but expand the reference pool to avoid singleton positives and to mine stronger imposters. 
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

from jaguar.config import DATA_STORE, EXPERIMENTS_STORE, PATHS, DEVICE, SEED 
from jaguar.utils.utils import ensure_dir, resolve_path, save_npy, save_parquet
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export, load_or_extract_embeddings 
from jaguar.models.foundation_models import FoundationModelWrapper  
from jaguar.utils.utils_xai import CosineSimilarityTarget, EmbeddingForwardWrapper, SimilarityForward, find_module_name  
from jaguar.utils.utils_xai import ig_saliency_batched_similarity 


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
    n_samples: int = 100
    seed: int = 51

    # IG hyperparameters (tradeoff: speed vs smoother attributions)
    ig_steps: int = 10
    ig_internal_bs: int = 32
    ig_batch_size: int = 32

    # Retrieval pair taxonomy
    pair_types: Tuple[str, ...] = ("easy_pos", "hard_neg", "hard_pos")

    # Output root; each run creates its own subfolder under here
    out_root: Path = Path()

# ============================================================
# Step 1: deterministic query selection (curated val subset)
# ============================================================


def get_curated_indices(split_df: pd.DataFrame, splits: Sequence[str]) -> np.ndarray:
    # training/eval/XAI should only use kept images.
    df = split_df[
        split_df["split_final"].isin(list(splits))
        & split_df["keep_curated"].fillna(False).astype(bool)
    ]

    return df["emb_row"].astype(np.int64).to_numpy()


def sample_indices(indices: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    # Deterministic sampling
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    
    if len(indices) == 0:
        return indices
    
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)

    return np.sort(chosen)


def get_val_query_indices(
    split_df: pd.DataFrame,
    out_root: Path,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Cache the exact query subset so explainers/models can be compared on the same images 
    across repeated runs (reproducible qualitative analysis)
    """
    idx_path = out_root / f"xai_val_idx_n{n_samples}.npy"
    if idx_path.exists():
        return np.load(idx_path)

    val_pool = get_curated_indices(split_df, splits=["val"])
    val_chosen = sample_indices(val_pool, n_samples=n_samples, seed=seed)

    ensure_dir(idx_path.parent)
    save_npy(idx_path, val_chosen)
    return val_chosen


# ============================================================
# Step 2: mine references for each pair_type (easy_pos/hard_neg)
# ============================================================

def mine_references_from_gallery(
    torch_ds,
    query_indices: np.ndarray,   # [N] Global indices of queries
    gallery_indices: np.ndarray, # [M] Global indices of the pool (e.g. all Val)
    split_df,
    model_wrapper: FoundationModelWrapper,
    out_root: Path,
    split: str,
    pair_types: Sequence[str],
) -> pd.DataFrame:
    """
    Mines reference pairs by searching for Queries within the Gallery.
    
    This stage defines the *cases we want to explain* (pair taxonomy):
    - easy_pos: “model should clearly match” (sanity case)
    - hard_neg: “model is likely to confuse” (failure-analysis case)
    - hard_pos: 
    """
    n_queries = len(query_indices)
    out_path = out_root / f"refs_n{n_queries}.parquet"

    if out_path.exists():
        print(f"[Info] Loading existing refs from {out_path}")
        return pd.read_parquet(out_path)
    
    print(f"[Mining] Queries: {n_queries} vs Gallery: {len(gallery_indices)}")

    # Load Embeddings
    all_embeddings = load_or_extract_embeddings(model_wrapper, torch_ds, num_workers=0)
    
    # Slice specific sets
    q_global = np.asarray(query_indices, dtype=np.int64)
    g_global = np.asarray(gallery_indices, dtype=np.int64)
    
    emb_q = all_embeddings[q_global] # [N, D]
    emb_g = all_embeddings[g_global] # [M, D]

    # Normalize
    emb_q = emb_q / (np.linalg.norm(emb_q, axis=1, keepdims=True) + 1e-12)
    emb_g = emb_g / (np.linalg.norm(emb_g, axis=1, keepdims=True) + 1e-12)

    # Rectangular Similarity [N, M]: Row i = query i, Col j = gallery item j
    sim_matrix = emb_q @ emb_g.T 
    
    # Filter labels
    all_labels = np.asarray(torch_ds.labels)
    labels_q = all_labels[q_global]
    labels_g = all_labels[g_global]

    # burst filtering: Skip same-burst matches so “easy positives” aren’t just near-duplicate frames.
    bg = split_df.set_index("emb_row")["burst_group_id"]  
    burst_q = bg.reindex(q_global).fillna(-1).to_numpy()
    burst_g = bg.reindex(g_global).fillna(-1).to_numpy()

    rows = []

    # Iterate over queries
    for i in tqdm(range(n_queries), desc="Mining Refs"):
        
        # Get ranks for specific query (descending similarity); processes row i of sim_matrix
        sims_i = sim_matrix[i]
        ranked_g_indices = np.argsort(-sims_i) # These are indices into G (0..M-1)

        q_idx_global = int(q_global[i])
        q_label = labels_q[i]

        found_pairs = {pt: False for pt in pair_types}

        hard_pos_candidate = None  # keep last same-ID encountered (lowest sim)
        # if hard_pos is requested, we must scan full ranked list (no early break)
        need_tail_scan = ("hard_pos" in pair_types)
        
        # --- Mining Logic ---
        # iterate through the ranked gallery items
        valid_rank = 0
        for g_idx in ranked_g_indices:
            # early break is only safe if we don't need hard_pos
            if (not need_tail_scan) and all(found_pairs.values()):
                break

            g_idx_global = int(g_global[g_idx])
            
            # Skip Self-Match
            if q_idx_global == g_idx_global:
                continue

            # Skip same burst group (avoid trivial near-duplicates)
            if burst_q[i] != -1 and burst_g[g_idx] == burst_q[i]:
                continue
            
            valid_rank += 1

            current_sim = float(sims_i[g_idx])
            g_label = labels_g[g_idx]
            is_same_id = (q_label == g_label)

            # Check Pair Types
            # Easy Positive: First (highest sim) match with same identity (TP, high-sim - success case)
            if "easy_pos" in pair_types and not found_pairs.get("easy_pos", False): 
                if is_same_id:
                    rows.append({
                        "pair_type": "easy_pos",
                        "query_idx": q_idx_global,
                        "ref_idx": g_idx_global,
                        "pair_sim": current_sim,
                        "rank_in_gallery": valid_rank
                    })
                    found_pairs["easy_pos"] = True

            # Hard Negative: First (highest sim) match with different identity (FP, high-sim - confusion case)
            if "hard_neg" in pair_types and not found_pairs.get("hard_neg", False):
                if not is_same_id:
                    rows.append({
                        "pair_type": "hard_neg",
                        "query_idx": q_idx_global,
                        "ref_idx": g_idx_global,
                        "pair_sim": current_sim,
                        "rank_in_gallery": valid_rank
                    })
                    found_pairs["hard_neg"] = True

            # Hard Positive: keep overwriting -> last same-ID (lowest sim)
            if "hard_pos" in pair_types and is_same_id:
                hard_pos_candidate = {
                    "pair_type": "hard_pos",
                    "query_idx": q_idx_global,
                    "ref_idx": g_idx_global,
                    "pair_sim": current_sim,
                    "rank_in_gallery": valid_rank
                }
                found_pairs["hard_pos"] = True  # means "exists at least one positive"

        # append hard_pos after loop (tail-based)
        if "hard_pos" in pair_types and hard_pos_candidate is not None:
            rows.append(hard_pos_candidate)

    
    ref_df = pd.DataFrame(rows)

    # Check for missing pairs
    expected_count = n_queries * len(pair_types)
    if len(ref_df) < expected_count:
        print(f"[Warning] Mined {len(ref_df)} pairs, expected {expected_count}. Some queries lack positives.")
        
    ensure_dir(out_path.parent)
    save_parquet(out_path, ref_df)
    return ref_df



# ============================================================
# Step 3: compute saliency maps for a given pair_type + explainer
# ============================================================

def compute_saliency_ig_for_pair_type(
    torch_ds,
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
        ref_sample = torch_ds[int(ref_idx)]
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
            sample = torch_ds[int(idx)]
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
    torch_ds,
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

        q_sample = torch_ds[qi]
        r_sample = torch_ds[ri]

        q_x = q_sample["img"].unsqueeze(0).to(device)  # [1,3,H,W]
        r_x = r_sample["img"].unsqueeze(0).to(device)  # [1,3,H,W]

        with torch.no_grad():
            ref_emb = model_wrapper.get_embeddings_tensor(r_x)  # [1,D]

        targets = [CosineSimilarityTarget(ref_emb, maximize=True)]

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
# Step 4: load-or-create artifacts per (explainer, pair_type)
# ============================================================

def load_or_create_saliency_maps(
    torch_ds,
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
                    artifact = compute_saliency_ig_for_pair_type(torch_ds, model_wrapper, ref_df_pair, cfg)
                elif explainer_name == "GradCAM":
                    artifact = compute_saliency_gradcam_for_pair_type(torch_ds, model_wrapper, ref_df_pair, cfg)
                else:
                    raise NotImplementedError(
                        f"Explainer '{explainer_name}' not implemented here yet."
                    )

                torch.save(artifact, sal_path)
                out[(explainer_name, pair_type)] = sal_path

    return out



def run_xai(cfg: XAIConfig, model_name: str, explainer_names: Sequence[str]) -> Dict[Tuple[str, str], Path]:
    
    run_root = cfg.out_root / f"{model_name}__{cfg.split_name}__n{cfg.n_samples}__seed{cfg.seed}"
    ensure_dir(run_root)

    # Load curated split artifacts (defines what counts as “kept” + provides burst IDs)
    artifacts_dir = resolve_path(
        "splits/jaguar_burst__str_closed_set__pol_drop_duplicates__k1",
        EXPERIMENTS_STORE,
    )
    split_df = pd.read_parquet(artifacts_dir / "full_split.parquet")

    # Load the dataset manifest used for indexing (emb_row is the common join key)
    _, torch_ds = load_jaguar_from_FO_export(
        resolve_path("fiftyone/splits_curated", DATA_STORE),
        dataset_name=cfg.dataset_name,
        processing_fn=None,
        overwrite_db=False, 
    )
    
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    torch_ds.transform = model_wrapper.preprocess 
    model_wrapper.eval()

    # Choose deterministic validation queries (small subset N for explainability runs)
    val_query_indices = get_val_query_indices(
        split_df=split_df,
        out_root=cfg.out_root,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
    )
    
    # Define a curated gallery pool (bigger pool => more meaningful negatives)
    gallery_df = split_df[
        split_df["split_final"].isin(["train", "val"]) &
        split_df["keep_curated"].fillna(False).astype(bool)
    ]
    gallery_indices = gallery_df["emb_row"].astype(np.int64).to_numpy()

    # Mine References (Query Subset vs. Full Val Gallery)
    ref_df = mine_references_from_gallery(
        torch_ds=torch_ds,
        query_indices=val_query_indices,     # N=100
        gallery_indices=gallery_indices,    # M=300+
        split_df=split_df,
        model_wrapper=model_wrapper,
        out_root=run_root,
        split=cfg.split_name,
        pair_types=cfg.pair_types,
    )

    print(f"[Result] Mined {len(ref_df)} pairs.")
    print(ref_df.groupby("pair_type")["pair_sim"].describe())

    
    # Compute Saliency Maps
    paths = load_or_create_saliency_maps(
        torch_ds=torch_ds,
        run_root=run_root,
        model_wrapper=model_wrapper,
        ref_df=ref_df,
        cfg=cfg,
        explainer_names=explainer_names,
    )

    return paths


if __name__ == "__main__":
    cfg = XAIConfig(
        dataset_name="jaguar_xai",   
        split_name="val",             
        n_samples=10,
        seed=SEED,
        ig_steps=10,
        ig_internal_bs=32,
        ig_batch_size=32,
        out_root=PATHS.runs / "xai/similarity",         ### TODO ISt das verhalten nur write oder auch read?
    )

    model_name = "MiewID"      # MiewID, ConvNeXt-V2
    explainer_names = ["IG", "GradCAM"]        

    artifact_paths = run_xai(cfg, model_name=model_name, explainer_names=explainer_names)
    for (expl, pair_type), p in artifact_paths.items():
        print(f"[Saved] {expl}/{pair_type}: {p}")