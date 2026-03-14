import pandas as pd
from jaguar.utils.utils_xai_similarity import load_all_refs, load_all_vectors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re
import torch
import cv2
from PIL import Image

from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.config import DATA_STORE, EXPERIMENTS_STORE, RESULTS_STORE
from jaguar.utils.utils import ensure_dir, resolve_path


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# ============================================================
# Generate summary table
# ============================================================

def build_pairwise_xai_main_table(df_vec: pd.DataFrame) -> pd.DataFrame:
    """
    Report table for pairwise XAI evaluation:
    explainer × pair_type with sanity, top-k, random, gap.
    """
    sub = df_vec[df_vec["metric"].isin(["sanity", "faith_topk", "faith_random", "faith_gap"])].copy()

    summary = (
        sub.groupby(["model", "explainer", "pair_type", "metric"])["value"]
        .mean()
        .reset_index()
        .pivot(
            index=["model", "explainer", "pair_type"],
            columns="metric",
            values="value",
        )
        .reset_index()
    )

    summary = summary.rename(columns={
        "sanity": "sanity_mean",
        "faith_topk": "mean_similarity_drop_topk_auc",
        "faith_random": "mean_similarity_drop_random_auc",
        "faith_gap": "faithfulness_gap_auc",
    })

    return summary.sort_values(["model", "explainer", "pair_type"]).reset_index(drop=True)


def save_pairwise_xai_main_table(df_vec: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    table = build_pairwise_xai_main_table(df_vec)
    table.to_csv(save_path / "pairwise_xai_main_table.csv", index=False)
    return table


def print_summary_table(df_vec: pd.DataFrame, save_path: Path):
    """
    RQ3 (GradCAM vs IG) + RQ4 (sanity/faithfulness compliance).

    Purpose: Compares metrics across model × explainer × pair_type
    Produce a compact “Mean ± Std” summary table from df_vec (per-sample metrics).
    Why: complements boxplots (shows exact values).
    """
    print("\n=== Summary Statistics (Mean ± Std) ===")

    summary = (
        df_vec
        .groupby(["model","explainer","pair_type","metric"])["value"]
        .agg(mean="mean", std="std")
        .assign(formatted=lambda d: d["mean"].map(lambda x: f"{x:.3f}") + " ± " + d["std"].map(lambda x: f"{x:.3f}"))
        .reset_index()
        .pivot(index=["model","explainer","pair_type"], columns="metric", values="formatted")
    )

    '''
    summary = pd.pivot_table(
        df_vec,
        index=["model", "explainer", "pair_type"],
        columns="metric",
        values="value",
        aggfunc=["mean", "std", "count"],
    )

    # pivot_table gives columns (stat, metric). Flip to (metric, stat)
    summary = summary.swaplevel(0, 1, axis=1)

    # rename + order
    summary = summary.rename(columns={
        "faith": "faith_mean",
        "sanity": "sanity_mean",
        "complexity": "complexity_mean",
    }, level=0)

    summary = summary.reindex(["faith_mean","sanity_mean","complexity_mean"], axis=1, level=0)
    summary = summary.reindex(["mean","std"], axis=1, level=1)
    '''
    print(summary)
    summary.to_csv(save_path / "xai_metrics_summary_table.csv")
    
    

def compute_failure_summary_all_models(refs_all: pd.DataFrame) -> pd.DataFrame:
    """
    RQ1 (Why wrong outranks right) – quantitative overview across models.

    Purpose: Compute per-model top-1 failure statistics from mined refs:
    - n_queries (number of unique queries)
    - n_failures and failure_rate (hard_neg at rank 1 AND easy_pos exists but not rank 1)
    - median rank of the best true match in failure cases
    - median similarity gap (wrong top-1 sim minus best true-match sim)
    Why: Provides the quantitative “breadth” for RQ1; complements qualitative failure panels.
    """
    out_rows = []

    for model, df_m in refs_all.groupby("model"):
        easy = df_m[df_m["pair_type"] == "easy_pos"][["query_idx","ref_idx","pair_sim","rank_in_gallery"]].copy()
        easy = easy.rename(columns={
            "ref_idx": "easy_ref_idx",
            "pair_sim": "easy_sim",
            "rank_in_gallery": "easy_rank",
        })

        hard = df_m[df_m["pair_type"] == "hard_neg"][["query_idx","ref_idx","pair_sim","rank_in_gallery"]].copy()
        hard = hard.rename(columns={
            "ref_idx": "hard_ref_idx",
            "pair_sim": "hard_sim",
            "rank_in_gallery": "hard_rank",
        })

        merged = hard.merge(easy, on="query_idx", how="left")

        n_queries = int(merged["query_idx"].nunique())

        # failures: top1 is negative (hard_rank==1), and a positive exists but isn't rank1 (easy_rank>1)
        failures = merged[
            (merged["hard_rank"] == 1) &
            (merged["easy_rank"].notna()) &
            (merged["easy_rank"] > 1)
        ].copy()

        n_failures = int(failures["query_idx"].nunique())
        failure_rate = float(n_failures / n_queries) if n_queries > 0 else 0.0

        if n_failures > 0:
            failures["sim_gap_wrong_minus_right"] = failures["hard_sim"] - failures["easy_sim"]
            med_easy_rank = float(failures["easy_rank"].median())
            med_gap = float(failures["sim_gap_wrong_minus_right"].median())
        else:
            med_easy_rank = float("nan")
            med_gap = float("nan")

        out_rows.append({
            "model": model,
            "n_queries": n_queries,
            "n_failures": n_failures,
            "failure_rate": failure_rate,
            "median_easy_rank_in_failures": med_easy_rank,
            "median_sim_gap_wrong_minus_right": med_gap,
        })

    return pd.DataFrame(out_rows).sort_values("model").reset_index(drop=True)


def get_top1_failures_for_model(refs_model: pd.DataFrame) -> pd.DataFrame:
    """
    RQ1 (Why does the model retrieve the wrong identity?).

    Purpose: Identify concrete top-1 failure cases for a single model/run:
    - hard_neg is the top-1 candidate (rank_in_gallery==1)
    - easy_pos exists but appears later (easy_rank>1)
    Also computes sim_gap_wrong_minus_right = hard_sim - easy_sim.
    Output: Failure DataFrame sorted by largest similarity gap (worst confusions first).
    Why: Selects the exact cases you should visualize/explain (query + wrong + true match).
    """
    easy = refs_model[refs_model["pair_type"]=="easy_pos"][["query_idx","ref_idx","pair_sim","rank_in_gallery"]].copy()
    easy = easy.rename(columns={"ref_idx":"easy_ref_idx","pair_sim":"easy_sim","rank_in_gallery":"easy_rank"})

    hard = refs_model[refs_model["pair_type"]=="hard_neg"][["query_idx","ref_idx","pair_sim","rank_in_gallery"]].copy()
    hard = hard.rename(columns={"ref_idx":"hard_ref_idx","pair_sim":"hard_sim","rank_in_gallery":"hard_rank"})

    merged = hard.merge(easy, on="query_idx", how="left")
    failures = merged[(merged["hard_rank"]==1) & (merged["easy_rank"].notna()) & (merged["easy_rank"]>1)].copy()
    failures["sim_gap_wrong_minus_right"] = failures["hard_sim"] - failures["easy_sim"]
    return failures.sort_values("sim_gap_wrong_minus_right", ascending=False)


def save_failure_panels(
    refs_all: pd.DataFrame,
    torch_ds,
    out_dir: Path,
    n_per_model: int = 10,
):
    """
    RQ1 – qualitative evidence.

    Purpose: Save concrete image triptychs for the worst top-1 failure cases per model:
    [Query | Wrong top-1 (hard_neg) | Best true match (easy_pos)] with ranks and similarities in titles.
    Output: PNG panels saved under output_dir/<model>/failure_XX__q<idx>.png
    """
    ensure_dir(out_dir)

    for model, df_m in refs_all.groupby("model"):
        failures = get_top1_failures_for_model(df_m).head(n_per_model)
        if failures.empty:
            print(f"[{model}] No top1-failure cases found.")
            continue

        model_dir = out_dir / model
        model_dir.mkdir(exist_ok=True)

        for k, row in enumerate(failures.itertuples(index=False), start=1):
            q = int(row.query_idx)
            wrong = int(row.hard_ref_idx)
            right = int(row.easy_ref_idx)

            q_img = Image.open(idx_to_imgpath(torch_ds, q)).convert("RGB")
            w_img = Image.open(idx_to_imgpath(torch_ds, wrong)).convert("RGB")
            r_img = Image.open(idx_to_imgpath(torch_ds, right)).convert("RGB")

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(q_img); axes[0].axis("off")
            axes[0].set_title(f"Query\nidx={q}")

            axes[1].imshow(w_img); axes[1].axis("off")
            axes[1].set_title(
                f"Wrong top1 (hard_neg)\n"
                f"idx={wrong}  sim={row.hard_sim:.3f}\n"
                f"rank={int(row.hard_rank)}"
            )

            axes[2].imshow(r_img); axes[2].axis("off")
            axes[2].set_title(
                f"Best true match (easy_pos)\n"
                f"idx={right}  sim={row.easy_sim:.3f}\n"
                f"rank={int(row.easy_rank)}"
            )

            plt.suptitle(f"{model} | sim_gap={row.sim_gap_wrong_minus_right:.3f}", y=1.02)
            plt.tight_layout()
            save_path = model_dir / f"failure_{k:02d}__q{q}.png"
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

        print(f"[{model}] Saved {len(failures)} panels to {model_dir}")


def add_easypos_rank_quantiles(pairs_df: pd.DataFrame, q_low=0.2, q_high=0.8) -> pd.DataFrame:
    """
    RQ3 (Easy vs hard true matches) – bucketing step.

    Purpose: Split easy_pos cases into difficulty buckets using rank quantiles of the first true match:
    - easy_pos_easy (low rank)
    - easy_pos_mid
    - easy_pos_hard (high rank)
    Why: Lets you compare what “hard positives” look like vs “easy positives”.
    """
    easy = pairs_df[pairs_df["pair_type"] == "easy_pos"].copy()
    if "rank_in_gallery" not in easy.columns:
        raise RuntimeError("Need rank_in_gallery. Fix mining first.")

    ranks = easy["rank_in_gallery"].astype(float).to_numpy()
    t_low = float(np.quantile(ranks, q_low))
    t_high = float(np.quantile(ranks, q_high))

    def bucket(r):
        if r <= t_low:
            return "easy_pos_easy"
        elif r >= t_high:
            return "easy_pos_hard"
        else:
            return "easy_pos_mid"

    easy["easypos_bucket"] = easy["rank_in_gallery"].apply(bucket)
    return easy, {"t_low": t_low, "t_high": t_high}



def idx_to_imgpath(torch_ds, idx: int) -> Path:
    s = torch_ds.samples[int(idx)]
    # prefer absolute path if stored
    if "filepath" in s:
        return Path(s["filepath"])
    if "path" in s:
        return Path(s["path"])
    # if only filename stored, resolve using dataset helper if available
    # fall back: assume it's already absolute
    return Path(s.get("filename", ""))



def build_saliency_lookup(artifact: dict) -> dict:
    """
    Returns dict mapping (query_idx, ref_idx) -> saliency_2d (H,W).
    """
    q = artifact["query_indices"]
    r = artifact["ref_indices"]
    sal = artifact["saliency"]

    if torch.is_tensor(q): q = q.cpu().numpy()
    if torch.is_tensor(r): r = r.cpu().numpy()
    if torch.is_tensor(sal): sal = sal.cpu().numpy()

    lookup = {}
    for i in range(len(q)):
        s = sal[i]
        # collapse channels if needed: (C,H,W) -> (H,W)
        if s.ndim == 3:
            s = np.max(np.abs(s), axis=0)
        lookup[(int(q[i]), int(r[i]))] = s
    return lookup


def select_pairtype_examples(
    refs_df: pd.DataFrame,
    pair_type: str,
    n: int = 3,
) -> pd.DataFrame:
    """
    Select a small qualitative subset for one pair type.
    - easy_pos: highest-sim same-id examples
    - hard_pos: lowest-sim same-id examples
    - hard_neg: highest-sim impostor examples
    """
    sub = refs_df[refs_df["pair_type"] == pair_type].copy()
    if sub.empty:
        return sub

    if pair_type == "easy_pos":
        sub = sub.sort_values("pair_sim", ascending=False)
    elif pair_type == "hard_pos":
        sub = sub.sort_values("pair_sim", ascending=True)
    elif pair_type == "hard_neg":
        sub = sub.sort_values("pair_sim", ascending=False)
    else:
        sub = sub.sort_values("pair_sim", ascending=False)

    return sub.head(n).reset_index(drop=True)


def save_pairtype_trained_vs_randomized_panels(
    torch_ds,
    run_dir: Path,
    out_dir: Path,
    explainer: str,
    pair_types: tuple[str, ...] = ("easy_pos", "hard_pos", "hard_neg"),
    n_per_type: int = 3,
) -> None:
    """
    Save qualitative panels for trained vs randomized saliency maps.

    Panel layout:
    [Query | Reference | Trained overlay on query | Randomized overlay on query]
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) / f"{run_dir.name}__{explainer}"
    ensure_dir(out_dir)

    refs_candidates = sorted(run_dir.glob("refs_n*.parquet"))
    if not refs_candidates:
        raise FileNotFoundError(f"No refs_n*.parquet found in {run_dir}")
    refs_df = pd.read_parquet(refs_candidates[0])

    for pair_type in pair_types:
        trained_path = run_dir / "explanations" / explainer / f"sal__{pair_type}.pt"
        randomized_path = run_dir / "explanations_randomized" / explainer / f"sal__{pair_type}.pt"

        if not trained_path.exists():
            print(f"[WARN] Missing trained artifact: {trained_path}")
            continue
        if not randomized_path.exists():
            print(f"[WARN] Missing randomized artifact: {randomized_path}")
            continue

        art_trained = torch.load(trained_path, map_location="cpu")
        art_randomized = torch.load(randomized_path, map_location="cpu")

        trained_lookup = build_saliency_lookup_from_artifact(art_trained)
        randomized_lookup = build_saliency_lookup_from_artifact(art_randomized)

        chosen = select_pairtype_examples(refs_df, pair_type=pair_type, n=n_per_type)
        if chosen.empty:
            print(f"[WARN] No rows for pair_type={pair_type}")
            continue

        pair_dir = out_dir / pair_type
        ensure_dir(pair_dir)

        for k, row in enumerate(chosen.itertuples(index=False), start=1):
            q = int(row.query_idx)
            r = int(row.ref_idx)

            q_img = np.array(Image.open(idx_to_imgpath(torch_ds, q)).convert("RGB"))
            r_img = np.array(Image.open(idx_to_imgpath(torch_ds, r)).convert("RGB"))

            sal_tr = trained_lookup.get((q, r), None)
            sal_rand = randomized_lookup.get((q, r), None)

            q_vis_tr = overlay_heatmap(q_img, sal_tr) if sal_tr is not None else q_img
            q_vis_rand = overlay_heatmap(q_img, sal_rand) if sal_rand is not None else q_img

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(q_img)
            axes[0].axis("off")
            axes[0].set_title(f"Query\nidx={q}")

            axes[1].imshow(r_img)
            axes[1].axis("off")
            axes[1].set_title(
                f"Reference\nidx={r}\nsim={float(row.pair_sim):.3f}"
            )

            axes[2].imshow(q_vis_tr)
            axes[2].axis("off")
            axes[2].set_title("Trained saliency")

            axes[3].imshow(q_vis_rand)
            axes[3].axis("off")
            axes[3].set_title("Randomized saliency")

            plt.suptitle(f"{run_dir.name} | {explainer} | {pair_type}", y=1.02)
            plt.tight_layout()
            save_path = pair_dir / f"{pair_type}_{k:02d}__q{q}__r{r}.png"
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

        print(f"[OK] Saved {len(chosen)} panels for {pair_type} to {pair_dir}")


def overlay_heatmap(img_rgb: np.ndarray, saliency_2d: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    s = np.abs(saliency_2d)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)

    if s.shape != img_rgb.shape[:2]:
        s = cv2.resize(s, (img_rgb.shape[1], img_rgb.shape[0]))

    heat = cv2.applyColorMap(np.uint8(255 * s), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 1)
        img_rgb = (img_rgb * 255).astype(np.uint8)

    return cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)


def build_saliency_lookup_from_artifact(artifact: dict) -> dict:
    """
    Maps (query_idx, ref_idx) -> saliency_2d (H,W).
    Artifact format matches your saved dict:
      query_indices: torch.LongTensor [N]
      ref_indices: torch.LongTensor [N]
      saliency: torch.FloatTensor [N,H,W]
    """
    q = artifact["query_indices"]
    r = artifact["ref_indices"]
    sal = artifact["saliency"]

    if torch.is_tensor(q): q = q.cpu().numpy()
    if torch.is_tensor(r): r = r.cpu().numpy()
    if torch.is_tensor(sal): sal = sal.cpu().numpy()

    lookup = {}
    for i in range(len(q)):
        lookup[(int(q[i]), int(r[i]))] = sal[i]  # [H,W]
    return lookup



def save_failure_panels_with_overlays(
    refs_all: pd.DataFrame,
    torch_ds,
    run_dir: Path,         
    out_dir: Path,
    model: str,
    explainer: str,
    n: int = 10,
):
    """
    RQ1 qualitative: Query | Wrong top-1 (hard_neg overlay) | Best true match (easy_pos overlay)
    """
    out_dir = Path(out_dir) / f"{model}__{explainer}"
    ensure_dir(out_dir)

    easy_pt = run_dir / "explanations" / explainer / "sal__easy_pos.pt"
    hard_pt = run_dir / "explanations" / explainer / "sal__hard_neg.pt"

    art_easy = torch.load(easy_pt, map_location="cpu")
    art_hard = torch.load(hard_pt, map_location="cpu")

    easy_lookup = build_saliency_lookup_from_artifact(art_easy)
    hard_lookup = build_saliency_lookup_from_artifact(art_hard)

    df_m = refs_all[refs_all["model"] == model]
    failures = get_top1_failures_for_model(df_m).head(n)
    if failures.empty:
        print(f"[{model}] No failures found.")
        return

    for k, row in enumerate(failures.itertuples(index=False), start=1):
        q = int(row.query_idx)
        wrong = int(row.hard_ref_idx)
        right = int(row.easy_ref_idx)

        q_img = np.array(Image.open(idx_to_imgpath(torch_ds, q)).convert("RGB"))
        w_img = np.array(Image.open(idx_to_imgpath(torch_ds, wrong)).convert("RGB"))
        r_img = np.array(Image.open(idx_to_imgpath(torch_ds, right)).convert("RGB"))

        sal_w = hard_lookup.get((q, wrong), None)
        sal_r = easy_lookup.get((q, right), None)

        w_vis = overlay_heatmap(w_img, sal_w) if sal_w is not None else w_img
        r_vis = overlay_heatmap(r_img, sal_r) if sal_r is not None else r_img

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(q_img); axes[0].axis("off")
        axes[0].set_title(f"Query\nidx={q}")

        axes[1].imshow(w_vis); axes[1].axis("off")
        axes[1].set_title(f"Wrong top1 (hard_neg)\nidx={wrong} sim={row.hard_sim:.3f} rank={int(row.hard_rank)}")

        axes[2].imshow(r_vis); axes[2].axis("off")
        axes[2].set_title(f"Best true match (easy_pos)\nidx={right} sim={row.easy_sim:.3f} rank={int(row.easy_rank)}")

        plt.suptitle(f"{model} | {explainer} | sim_gap={row.sim_gap_wrong_minus_right:.3f}", y=1.02)
        plt.tight_layout()
        plt.savefig(out_dir / f"failure_overlay_{k:02d}__q{q}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"[{model}] Saved {len(failures)} overlay panels to {out_dir}")


if __name__ == "__main__":
    run_root = resolve_path("xai/similarity", EXPERIMENTS_STORE)
    save_root = resolve_path("xai/similarity", RESULTS_STORE)
    ensure_dir(save_root)
    dataset_name = "jaguar_xai" 
    backbone_name = "EVA-02" #TODO go over all folder by yourself
    n_samples = 10
    seed = 42
    explainer = "IG"

    df_vec = load_all_vectors(run_root)
    refs_all = load_all_refs(run_root)
    print(refs_all.head())

    main_table = save_pairwise_xai_main_table(df_vec, save_root)
    print(main_table.head())
    
    
    # RQ 3 + RQ 4 (and boxplot per model for complexity is RQ")
    print_summary_table(df_vec, save_root)
    
    summary_fail = compute_failure_summary_all_models(refs_all)
    summary_fail_display = summary_fail.copy()
    summary_fail_display[["median_easy_rank_in_failures","median_sim_gap_wrong_minus_right"]] = \
    summary_fail_display[["median_easy_rank_in_failures","median_sim_gap_wrong_minus_right"]].fillna("—")
    print(summary_fail_display)
    summary_fail.to_csv(save_root / "failure_summary_all_models.csv", index=False)

    _, torch_ds = load_jaguar_from_FO_export(
        resolve_path("fiftyone/splits_curated", DATA_STORE),
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=False, 
    )

    save_failure_panels(refs_all, torch_ds, save_root / "failure_panels", n_per_model=n_samples)

    runs_dir = resolve_path(
        f"xai/similarity/{backbone_name}__val__n{n_samples}__seed{seed}",
        EXPERIMENTS_STORE,
    )
    refs_path = runs_dir / f"refs_n{n_samples}.parquet"

    pairs_df = pd.read_parquet(refs_path)

    rank_col="rank_in_gallery"
    
    # failures list (for RQ1)
    failures = get_top1_failures_for_model(pairs_df)
    print(failures.head(20))
    print("n_failures:", len(failures))

    if not failures.empty:
        save_failure_panels_with_overlays(
            refs_all=refs_all,
            torch_ds=torch_ds,
            run_dir=runs_dir,
            out_dir=save_root / "failure_overlays",
            model=backbone_name,
            explainer=explainer,
            n=n_samples,
        )
    
    save_pairtype_trained_vs_randomized_panels(
        torch_ds=torch_ds,
        run_dir=runs_dir,
        out_dir=save_root / "trained_vs_randomized_panels",
        explainer=explainer,
        pair_types=("easy_pos", "hard_pos", "hard_neg"),
        n_per_type=3,
    )

    print(f"\nAnalysis complete. Results saved to {save_root}")
    