"""
1) Why does the model retrieve the wrong identity?

RQ: In top-1 failures, what evidence makes a wrong gallery image outrank the best true match?
Need: Your mined pairs per query: top-1 wrong candidate (usually hard_neg, valid_rank=1) + the best true match 
(easy_pos, valid_rank>1). Run similarity-target explanations (BiLRP / similarity attribution) on both pairs 
and compare. Quantus: masking faithfulness on the similarity score (top-relevance vs random/bottom), plus 
sanity randomization. Report qualitative side-by-side explanations + quantitative faithfulness curves.


3) What separates easy from hard true matches?

RQ: When the true match exists, why is it sometimes ranked high (easy) and sometimes far down (hard)?
Need: Use easy_pos.valid_rank (quantiles) or explicit hard_pos (lowest-sim same-ID). Compare easy positives 
vs hard positives with similarity-target explanations. Quantus: same masking faithfulness; optionally 
compare explanation concentration/entropy across buckets. Show that hard positives either lack identity 
evidence (occlusion/blur) or drift to non-identity regions.


4) Do explanations pass sanity and faithfulness requirements?

RQ: Are your explanations actually tied to the learned model and causally related to the score?
Need: Pick a small, representative set: ~10 easy_pos + ~10 hard_neg (and optionally hard positives). 
Run (a) sanity: randomize weights (or last block) and show maps degrade + faithfulness drops to random. 
Run (b) faithfulness: top-k relevance masking causes larger similarity/log-prob drop than random/bottom 
masking. This is your “Q2 compliance” section.


RQ 2: The Anatomy of Failure (Hard Negatives Analysis)
Importance: ⭐⭐⭐⭐ (Explains why Re-ID fails)
Easiness: ⭐⭐⭐⭐ (Requires correlating two of your outputs)
The Question: When the model gets confused (Hard Negatives), is it because it is looking at the wrong place (e.g., background), or because the features are too complex?
Hypothesis: Hard Negatives have higher Background Sensitivity (the model matched the trees, not the cat) OR higher Saliency Complexity (the model looked everywhere and got confused) compared to Easy Positives.
How to Analyse:
Filter: Take your Hard Negatives (incorrect high-score matches) and Easy Positives (correct high-score matches).
Compare: Look at the distribution of your Faithfulness and Complexity metrics for these two groups.
Check: Do Hard Negatives have a significantly higher drop_bg than Easy Positives?
The Plot:
Box Plot: X-axis = Categories (Easy Pos, Hard Pos, Hard Neg). Y-axis = drop_bg.
Finding: If Hard Negatives have higher drop_bg, you prove that background bias is the primary cause of false positives.


RQ 3: Evaluation Methodology (GradCAM vs IG)
Importance: ⭐⭐⭐ (Technical validation)
Easiness: ⭐⭐⭐⭐⭐ (You already ran the metrics)
The Question: Which explanation method is more faithful to the model's decision making for wildlife Re-ID?
Hypothesis: Integrated Gradients (IG) might be noisier (higher complexity) but arguably more faithful than GradCAM, or vice-versa depending on the architecture (CNN vs ViT).
How to Analyse:
Metric: Compare average Faithfulness and Randomization scores between GradCAM and IG.
Sanity Check: If Randomization score is low (meaning the map doesn't change when model weights are scrambled), that method is invalid.
The Plot:
Radar Chart or Table: Columns = [Faithfulness, Complexity, Randomization]. Rows = [GradCAM, IG].
Use: This justifies which visual you show in the paper. "We display GradCAM images because they achieved 0.85 Faithfulness vs 0.70 for IG."


RQ 4: Complexity vs. Performance Trade-off
Importance: ⭐⭐⭐ (Theoretical insight)
Easiness: ⭐⭐⭐ (Requires correlating ranking with XAI metrics)
The Question: Do better-performing models (High mAP) produce simpler, more focused explanations (Low Complexity)?
Hypothesis: "Good" models focus tightly on the rosette patterns (Low Complexity, High Saliency on Foreground). "Bad" models look at the whole image scattershot (High Complexity).
How to Analyse:
Correlation: Scatter plot of Model mAP (or Rank-1) vs. Average Saliency Complexity.
Within-Model: For a single model, does the Complexity of the explanation correlate with the Confidence of the prediction?
The Plot:
Scatter Plot: X-axis = Prediction Confidence. Y-axis = Saliency Complexity.
Trend: You expect a negative correlation (Higher confidence = Simpler, more focused explanation).

"""
import pandas as pd
from jaguar.analysis.xai_background_sensitivity import run_xai_background_sensitivity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re
import torch
import cv2
from PIL import Image

from jaguar.config import DATA_STORE, EXPERIMENTS_STORE, PATHS, RESULTS_STORE
from jaguar.utils.utils import ensure_dir, resolve_path
from jaguar.utils.utils_datasets import load_full_jaguar_from_FO_export
from jaguar.analysis.xai_metrics_analysis import run_xai_metrics_analysis

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


_RUN_RE = re.compile(r"^(?P<model>.+)__(?P<split>.+)__n(?P<n>\d+)__seed(?P<seed>\d+)$")


def load_all_refs(xai_similarity_root: Path) -> pd.DataFrame:
    """
    Loads and concatenates all refs_n*.parquet under xai_similarity_root.
    Adds run metadata columns: model, split, n_samples, seed, run_id.
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    Requires artifacts saved at:
      run_dir/explanations/<explainer>/sal__easy_pos.pt
      run_dir/explanations/<explainer>/sal__hard_neg.pt
    """
    out_dir = Path(out_dir) / f"{model}__{explainer}"
    out_dir.mkdir(parents=True, exist_ok=True)

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


def run_xai_similarity_analysis(
    run_root: Path,
    save_dir: Path,
    manifest_dir: Path,
    dataset_name: str = "jaguar_xai",
    overlay_model_name: str | None = None,
    overlay_explainer: str = "IG",
    n_per_model: int = 10,
) -> dict[str, Path]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    refs_all = load_all_refs(run_root)
    if refs_all.empty:
        print(f"[ANALYSIS][WARN] No XAI refs found under: {run_root}")
        return {}

    summary_fail = compute_failure_summary_all_models(refs_all)
    summary_fail_path = save_dir / "failure_summary_all_models.csv"
    summary_fail.to_csv(summary_fail_path, index=False)

    _, torch_ds = load_full_jaguar_from_FO_export(
        manifest_dir,
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=False,
    )

    failure_panels_dir = save_dir / "failure_panels"
    save_failure_panels(
        refs_all=refs_all,
        torch_ds=torch_ds,
        out_dir=failure_panels_dir,
        n_per_model=n_per_model,
    )

    out = {
        "failure_summary_csv": summary_fail_path,
        "failure_panels_dir": failure_panels_dir,
    }

    if overlay_model_name is not None:
        run_dir = run_root / f"{overlay_model_name}__val__n10__seed51"
        refs_path = run_dir / "refs_n10.parquet"

        if refs_path.exists():
            pairs_df = pd.read_parquet(refs_path)
            failures = get_top1_failures_for_model(pairs_df)

            if not failures.empty:
                overlay_dir = save_dir / "failure_overlays"
                save_failure_panels_with_overlays(
                    refs_all=refs_all,
                    torch_ds=torch_ds,
                    run_dir=run_dir,
                    out_dir=overlay_dir,
                    model=overlay_model_name,
                    explainer=overlay_explainer,
                    n=n_per_model,
                )
                out["failure_overlays_dir"] = overlay_dir

    return out


if __name__ == "__main__":
    run_root = resolve_path("xai/similarity", EXPERIMENTS_STORE)
    save_root = resolve_path("xai/similarity", RESULTS_STORE)
    ensure_dir(save_root)
    dataset_name = "jaguar_xai"

    run_xai_metrics_analysis(
        run_root=run_root,
        save_dir=save_root,
    )

    xai_similarity_outputs = run_xai_similarity_analysis(
        run_root=run_root,
        save_dir=save_root,
        manifest_dir=resolve_path("fiftyone/splits_curated", DATA_STORE),
        dataset_name=dataset_name,
        overlay_model_name="MiewID",
        overlay_explainer="IG",
        n_per_model=10,
    )

    print(f"[ANALYSIS] XAI similarity outputs: {xai_similarity_outputs}")

    run_xai_background_sensitivity(
        experiments_dir=PATHS.runs / "xai/background_sensitivity",
    )