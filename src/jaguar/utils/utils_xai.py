from pathlib import Path
from typing import Sequence
from jaguar.utils.utils import ensure_dir, save_npy
import torch
import numpy as np
import pandas as pd
import re


def normalize_heatmap(h):
    """
    Normalize a heatmap to the [0, 1] range for stable downstream comparison or display.
    """
    if isinstance(h, torch.Tensor):
        h = h.detach().cpu().float().numpy()
    h = h.astype(np.float32)

    h_min = h.min()
    h_max = h.max()
    if h_max - h_min < 1e-12:
        return np.zeros_like(h, dtype=np.float32)
    return (h - h_min) / (h_max - h_min)


def find_module_name(model: torch.nn.Module, target_module: torch.nn.Module) -> str:
    """
    Return the dotted module name of a given submodule inside the model.
    """
    for name, m in model.named_modules():
        if m is target_module:
            return name
    return "<unnamed>"


def save_vec(save_dir: Path, prefix: str, expl: str, pt: str, vec: np.ndarray) -> str:
    """
    Save one metric vector as a .npy file and return the stored filename.
    """
    fname = f"{prefix}__{expl}__{pt}.npy"
    p = Path(save_dir) / fname
    np.save(p, np.asarray(vec, dtype=np.float32))
    return fname

# ============================================================
# Deterministic query selection (curated val subset)
# ============================================================

def get_curated_indices(split_df: pd.DataFrame, splits: Sequence[str]) -> np.ndarray:
    """
    Return global embedding-row indices for curated samples in the requested splits.
    """
    df = split_df[
        split_df["split_final"].isin(list(splits))
        & split_df["keep_curated"].fillna(False).astype(bool)
    ]
    return df["emb_row"].astype(np.int64).to_numpy()


def sample_indices(indices: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """
    Deterministically sample up to n_samples indices without replacement.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)

    if len(indices) == 0:
        return indices

    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=min(n_samples, len(indices)), replace=False)
    return np.sort(chosen)


def get_val_query_indices(
    split_df: pd.DataFrame,
    out_root: Path,
    n_samples: int | None,
    seed: int,
) -> np.ndarray:
    """
    Build or reload the fixed curated validation query subset used across repeated runs.
    """
    n_tag = "full" if n_samples is None else str(n_samples)
    idx_path = out_root / f"xai_val_idx_n{n_tag}.npy"

    if idx_path.exists():
        return np.load(idx_path)

    val_pool = get_curated_indices(split_df, splits=["val"])
    val_chosen = sample_indices(val_pool, n_samples=n_samples, seed=seed)

    ensure_dir(idx_path.parent)
    save_npy(idx_path, val_chosen)
    return val_chosen


def resolve_n_samples(n_samples: int | str | None) -> int | None:
    """
    Resolve the configured sample count, treating full/all as the full split.
    """
    if n_samples is None:
        return None

    if isinstance(n_samples, str):
        value = n_samples.strip().lower()
        if value in {"full", "all"}:
            return None
        return int(value)

    return int(n_samples)


def format_n_samples_tag(n_samples: int | str | None) -> str:
    """
    Convert the configured sample count into a stable filename/run tag.
    """
    resolved = resolve_n_samples(n_samples)
    return "full" if resolved is None else str(resolved)


def resolve_vec_path(vec_path_raw: str, metrics_dir: Path) -> Path:
    """
    Resolve a stored metric-vector path against the current metrics directory layout.

    Supports existing absolute paths, relative paths, and older absolute paths from
    the pre-similarity directory structure.
    """
    if not isinstance(vec_path_raw, str) or not vec_path_raw:
        raise FileNotFoundError(f"Empty vec path: {vec_path_raw}")

    p = Path(vec_path_raw)

    if p.is_absolute() and p.exists():
        return p

    cand = metrics_dir / p
    if cand.exists():
        return cand

    if p.is_absolute():
        parts = list(p.parts)
        try:
            xai_i = parts.index("xai")
        except ValueError:
            pass
        else:
            # if it already has similarity right after xai, nothing to rewrite
            if xai_i + 1 < len(parts) and parts[xai_i + 1] != "similarity":
                rewritten = Path(*parts[: xai_i + 1], "similarity", *parts[xai_i + 1 :])
                if rewritten.exists():
                    return rewritten

    raise FileNotFoundError(
        f"Could not resolve vec path:\n"
        f"  vec_path_raw: {vec_path_raw}\n"
        f"  metrics_dir : {metrics_dir}"
    )


_RUN_RE = re.compile(r"^(?P<model>.+)__(?P<split>.+)__n(?P<n>\d+)__seed(?P<seed>\d+)$")


def load_all_vectors(run_root: Path) -> pd.DataFrame:
    """
    Load all per-sample XAI metric vectors under a run root into one long dataframe.
    """
    out = []

    for summary_csv in run_root.rglob("xai_metrics/xai_summary_metrics.csv"):
        metrics_dir = summary_csv.parent          
        run_dir = metrics_dir.parent              
        m = _RUN_RE.match(run_dir.name)
        if m is None:
            continue

        model = m.group("model")
        split = m.group("split")
        n_samples = int(m.group("n"))
        seed = int(m.group("seed"))

        summary = pd.read_csv(summary_csv)

        for _, row in summary.iterrows():
            explainer = row["explainer"]
            pair_type = row["pair_type"]

            for metric_name, vec_col in [
                ("sanity", "sanity_vec_path"),
                ("faith_topk", "faith_topk_vec_path"),
                ("faith_random", "faith_random_vec_path"),
                ("faith_gap", "faith_gap_vec_path"),
                ("complexity", "complexity_vec_path"),
            ]:
                vec_path_raw = row.get(vec_col, "")
                vec_path = resolve_vec_path(vec_path_raw, metrics_dir=metrics_dir)

                v = np.load(vec_path)  # shape [N]

                for i, val in enumerate(v):
                    out.append({
                        "model": model,
                        "split": split,
                        "n_samples": n_samples,
                        "seed": seed,
                        "run_id": run_dir.name,
                        "explainer": explainer,
                        "pair_type": pair_type,
                        "metric": metric_name,
                        "sample_i": int(i),
                        "value": float(val),
                        "vec_path": str(vec_path),
                    })

    return pd.DataFrame(out)


def load_all_refs(xai_similarity_root: Path) -> pd.DataFrame:
    """
    Load and concatenate all reference parquet files and attach run metadata columns.
    """
    dfs = []

    for pq in xai_similarity_root.rglob("refs_n*.parquet"):
        # refs_n10.parquet is either directly under run_dir OR under run_dir/refs/
        run_dir = pq.parent
        m = _RUN_RE.match(run_dir.name)

        if m is None and pq.parent.parent is not None:
            run_dir = pq.parent.parent
            m = _RUN_RE.match(run_dir.name)

        if m is None:
            continue

        df = pd.read_parquet(pq)

        df["model"] = m.group("model")
        df["split"] = m.group("split")
        df["n_samples"] = int(m.group("n"))
        df["seed"] = int(m.group("seed"))
        df["run_id"] = run_dir.name
        df["refs_file"] = str(pq)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def summarize_retrieval_variant(df: pd.DataFrame, suffix: str) -> dict:
    """
    Summarize retrieval quality for one query variant identified by its column suffix.
    """
    return {
        "rank1": float(df[f"is_rank1_{suffix}"].mean()),
        "rank5": float(df[f"is_rank5_{suffix}"].mean()),
        "median_gold_rank": float(df[f"gold_rank_{suffix}"].median()),
        "mean_gold_rank": float(df[f"gold_rank_{suffix}"].mean()),
        "mean_margin": float(df[f"margin_gold_minus_impostor_{suffix}"].mean()),
    }


def summarize_bg_vs_jaguar(df: pd.DataFrame) -> dict:
    """
    Summarize whether background-only queries outperform jaguar-only queries.
    """
    return {
        "share_bg_better_rank": float(df["bg_better_than_jag_rank"].mean()),
        "share_bg_better_rank1": float(df["bg_better_than_jag_rank1"].mean()),
        "share_bg_better_rank5": float(df["bg_better_than_jag_rank5"].mean()),
        "share_bg_better_margin": float(df["bg_better_than_jag_margin"].mean()),
        "median_rank_delta_bg_minus_jag": float(df["gold_rank_delta_bg_minus_jag"].median()),
        "mean_margin_delta_bg_minus_jag": float(df["margin_delta_bg_minus_jag"].mean()),
    }


def summarize_embedding_stability(df: pd.DataFrame) -> dict:
    """
    Summarize how strongly original embeddings align with jaguar-only versus background-only queries.
    """
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


def build_bg_sensitivity_summaries(
    retrieval_df: pd.DataFrame,
    similarity_res: pd.DataFrame,
) -> dict:
    """
    Merge retrieval and embedding results and build the main foreground-vs-background summaries.
    """
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

    retrieval_summary = {
        "all": summarize_bg_vs_jaguar(analysis_df),
        "orig_rank1_correct": summarize_bg_vs_jaguar(analysis_correct),
        "orig_rank1_wrong": summarize_bg_vs_jaguar(analysis_wrong),
        "variant_summary": retrieval_variant_summary,
    }

    similarity_summary = {
        "all": summarize_embedding_stability(analysis_df),
        "orig_rank1_correct": summarize_embedding_stability(analysis_correct),
        "orig_rank1_wrong": summarize_embedding_stability(analysis_wrong),
    }

    return {
        "analysis_df": analysis_df,
        "retrieval_summary": retrieval_summary,
        "similarity_summary": similarity_summary,
    }