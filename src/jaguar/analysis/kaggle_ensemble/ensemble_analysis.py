import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


from jaguar.config import PATHS
from jaguar.utils.utils_evaluation import get_ranked_candidates_for_query


def _build_embrow_to_filename(split_df: pd.DataFrame) -> dict[int, str]:
    """Map global emb_row ids to filenames."""
    if "emb_row" not in split_df.columns or "filename" not in split_df.columns:
        raise ValueError("split_df must contain 'emb_row' and 'filename'.")
    return dict(zip(split_df["emb_row"].astype(int), split_df["filename"].astype(str)))


def _topk_strings_from_query_df(
    query_df: pd.DataFrame,
    topk_df: pd.DataFrame | None,
    embrow_to_filename: dict[int, str],
    prefix: str,
) -> pd.DataFrame:
    """
    Build compact top-k string columns for one method.
    Expects:
      - query_df with query_idx, ap, rank1_correct, first_pos_rank, top1_idx, top1_label
      - topk_df optionally with columns: query_idx, rank_in_gallery, gallery_global_idx, gallery_label, sim
    """
    out = query_df[
        ["query_idx", "ap", "rank1_correct", "first_pos_rank", "top1_idx", "top1_label", "top1_sim"]
    ].copy()

    out = out.rename(columns={
        "ap": f"ap__{prefix}",
        "rank1_correct": f"rank1__{prefix}",
        "first_pos_rank": f"first_pos_rank__{prefix}",
        "top1_idx": f"top1_idx__{prefix}",
        "top1_label": f"top1_label__{prefix}",
        "top1_sim": f"top1_sim__{prefix}",
    })

    out[f"top1_file__{prefix}"] = out[f"top1_idx__{prefix}"].map(embrow_to_filename)

    if topk_df is not None and len(topk_df) > 0:
        topk_sub = topk_df[topk_df["rank_in_gallery"] <= 3].copy()
        topk_sub["gallery_file"] = topk_sub["gallery_global_idx"].map(embrow_to_filename)
        topk_sub["entry"] = topk_sub.apply(
            lambda r: f"{int(r['rank_in_gallery'])}:{r['gallery_label']}|{r['gallery_file']}|{r['sim']:.4f}",
            axis=1,
        )
        top3 = (
            topk_sub.groupby("query_idx")["entry"]
            .apply(lambda s: " || ".join(s.tolist()))
            .reset_index()
            .rename(columns={"entry": f"top3__{prefix}"})
        )
        out = out.merge(top3, on="query_idx", how="left")
    else:
        out[f"top3__{prefix}"] = None

    return out


def build_qualitative_review_df(
    per_query_comparison_df: pd.DataFrame,
    split_df: pd.DataFrame,
    method_query_dfs: dict[str, pd.DataFrame],
    method_topk_dfs: dict[str, pd.DataFrame] | None = None,
    target_method: str = "score_fusion",
    worst_n: int = 10,
) -> pd.DataFrame:
    """
    Build a compact qualitative review table for the worst losses of one fusion method.

    method_query_dfs:
      dict like {
        "EVA-02": eva_query_df,
        "MiewID": miewid_query_df,
        "score_fusion": score_query_df,
      }

    method_topk_dfs:
      optional dict with precomputed ranked-candidate rows per method.
      Each df should contain:
        query_idx, rank_in_gallery, gallery_global_idx, gallery_label, sim
    """
    method_topk_dfs = method_topk_dfs or {}
    embrow_to_filename = _build_embrow_to_filename(split_df)

    if f"ap__{target_method}" not in per_query_comparison_df.columns:
        raise ValueError(f"Missing ap__{target_method} in per_query_comparison_df")

    review = per_query_comparison_df.copy()

    query_file_map = embrow_to_filename
    review["query_file"] = review["query_idx"].map(query_file_map)

    loss_col = f"{target_method}_delta_vs_best_model"
    if loss_col not in review.columns:
        raise ValueError(f"Missing column: {loss_col}")

    review = review.sort_values(loss_col).head(worst_n).copy()

    keep_cols = [
        "query_idx",
        "query_label",
        "query_file",
        "best_model",
        "best_model_ap",
        f"ap__{target_method}",
        loss_col,
        f"{target_method}_delta_vs_oracle",
    ]
    review = review[keep_cols]

    for method_name, query_df in method_query_dfs.items():
        method_block = _topk_strings_from_query_df(
            query_df=query_df,
            topk_df=method_topk_dfs.get(method_name),
            embrow_to_filename=embrow_to_filename,
            prefix=method_name,
        )
        review = review.merge(method_block, on="query_idx", how="left")

    return review

def build_topk_candidates_df(
    retrieval,
    top_k: int = 3,
) -> pd.DataFrame:
    """Export top-k ranked candidates per query from a RetrievalState."""
    rows = []

    for i in range(len(retrieval.q_global)):
        q_idx_global, q_label, ranked_candidates = get_ranked_candidates_for_query(retrieval, i)

        for cand in ranked_candidates[:top_k]:
            rows.append({
                "query_idx": q_idx_global,
                "query_label": q_label,
                "rank_in_gallery": cand["rank_in_gallery"],
                "gallery_global_idx": cand["gallery_global_idx"],
                "gallery_label": cand["gallery_label"],
                "sim": cand["sim"],
            })

    return pd.DataFrame(rows)


def build_singles_group_summary(experiment_root_dir: str | Path) -> pd.DataFrame:
    """
    Aggregate protocol_comparison.csv files from all single-model runs in one
    experiment group into one summary table with one row per model.
    """
    experiment_root_dir = Path(experiment_root_dir)

    rows = []

    for run_dir in sorted(experiment_root_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        comp_path = run_dir / "protocol_comparison.csv"
        if not comp_path.exists():
            continue

        df = pd.read_csv(comp_path)

        # Keep only the actual single-model row, not oracle/fusion rows
        # For single runs this should be the row where method_type == "single"
        df = df[df["method_type"] == "single"].copy()
        if df.empty:
            continue

        if len(df) != 1:
            print(f"[WARN] Expected exactly 1 single row in {comp_path}, found {len(df)}")
            continue

        row = df.iloc[0].to_dict()

        row["run_name"] = run_dir.name
        row["model_name"] = row.get("method_name")
        rows.append(row)

    if not rows:
        raise ValueError(f"No usable protocol_comparison.csv files found in {experiment_root_dir}")

    out_df = pd.DataFrame(rows)

    # Nice default ordering for inspection
    sort_cols = [
        c for c in [
            "mAP__valonly_gallery",
            "mAP__mean_across_protocols",
            "mAP__delta_protocols",
        ]
        if c in out_df.columns
    ]
    ascending = [False, False, True][:len(sort_cols)]

    if sort_cols:
        out_df = out_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out_df
    

def run(
    config: dict, 
    save_dir: Path, 
    root_dir: Path | None = None, 
    run_dir: Path | None = None, 
    **kwargs
) -> None:
    img_root = PATHS.data_train
    singles_df = build_singles_group_summary(root_dir)

    print(singles_df.to_string(index=False))
    singles_df.to_csv(root_dir / "singles_group_summary.csv", index=False)

    cols = [
        "run_name",
        "model_name",
        "mAP__trainval_gallery",
        "mAP__valonly_gallery",
        "mAP__mean_across_protocols",
        "mAP__delta_protocols",
        "rank1__trainval_gallery",
        "rank1__valonly_gallery",
    ]

    cols = [c for c in cols if c in singles_df.columns]
    print(singles_df[cols].to_string(index=False))

    """
    Fall: gute Query, aber schlechte Positives
    22 Bilder aus 3 Bursts, nur 1 Burst gut - alle Modelle falsch, aber anders falsch
    """

    paths = {
        "Query\nBororo": img_root / "train_0118.png",
        "EVA-02 wrong Top1\nJaju": img_root / "train_0128.png",
        "MiewID wrong Top1\nOusado": img_root / "train_1394.png",
        "Score fusion wrong Top1\nTi": img_root / "train_0160.png",
        "Correct Bororo ref": img_root / "train_0018.png",
    }

    fig, axes = plt.subplots(1, len(paths), figsize=(20, 5))

    for ax, (title, path) in zip(axes, paths.items()):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{title}\n{path.name}", fontsize=10)
        ax.axis("off")
    out_path = PATHS.results /"img"/"query.png"
    #ensure_dir(out_path)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.tight_layout()
    plt.show()


    """
    Fall: schwache, unvollständige Query-Information
    Wenn EVA-02 bei Jaju besser war als die anderen, könnte das heißen:
    es nutzt lokale Texturmuster besser
    es ist robuster gegen starke partielle Sichtbarkeit
    es braucht weniger globalen Kontext

    Jaju appears to be a partial-visibility case: the query images contain only fragmented flank regions with little global body context. The main difficulty is therefore incomplete query evidence rather than poor gallery support.
    """

    paths = {
        "Query\nJaju": img_root / "train_0556.png",
        "ConvNeXt wrong Top1\nSaseka": img_root / "train_1673.png",
        "Score fusion Top1\nJaju": img_root / "train_0558.png",
    }

    fig, axes = plt.subplots(1, len(paths), figsize=(18, 6), dpi=200)

    for ax, (title, path) in zip(axes, paths.items()):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{title}\n{path.name}", fontsize=11)
        ax.axis("off")
    out_path = PATHS.results /"img"/ "jaju_case_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    print(f"Saved to: {out_path.resolve()}")
    

    """
    Also:
    MiewID erkennt Medrosa extrem stabil und praktisch perfekt.

    EVA-02

    Für mehrere Medrosa-Queries:

    first_pos_rank = 3, 6, 14

    AP nur ~0.38 bis 0.49, EVA-02 verwechselt Medrosa teils systematisch mit Marcela.
    Medrosa is a model-specific confusion case rather than a general hard identity. MiewID retrieves correct positives at rank 1 consistently with near-perfect AP, whereas EVA-02 often confuses Medrosa with Marcela and ranks the first correct positive lower. Fixed score fusion remains correct at rank 1 but slightly dilutes the stronger MiewID ranking.
    """

    paths = {
        "Query\nMedrosa": img_root / "train_1214.png",
        "MiewID Top1\nMedrosa": img_root / "train_1268.png",
        "Score fusion Top1\nMedrosa": img_root / "train_1248.png",
        "Wrong other model\nMarcela": img_root / "train_1074.png",
    }

    fig, axes = plt.subplots(1, len(paths), figsize=(20, 6), dpi=200)

    for ax, (title, path) in zip(axes, paths.items()):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{title}\n{path.name}", fontsize=11)
        ax.axis("off")
    out_path = PATHS.results /"img"/ "medrosa_case_comparison.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    print(f"Saved to: {out_path.resolve()}")