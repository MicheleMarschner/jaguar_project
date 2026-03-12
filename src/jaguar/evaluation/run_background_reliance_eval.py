"""
Background-reliance evaluation runner for Jaguar Re-ID.

Purpose:
- loads one trained retrieval model
- evaluates the same validation retrieval task under multiple background settings
- compares retrieval performance across settings
- stores both aggregate and per-query outputs

This file is the main orchestration entrypoint for the background-reliance RQ.
It should stay thin: shared retrieval logic lives in retrieval_core,
and setting-specific image/embedding logic lives in background_reliance_core.
"""
import os

from jaguar.utils.utils_models import load_or_extract_jaguarid_embeddings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import pandas as pd
import numpy as np

from jaguar.config import PATHS
from jaguar.logging.wandb_logger import init_wandb_run
from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_datasets import build_eval_processing_fn, get_transforms, load_split_jaguar_from_FO_export
from jaguar.utils.utils_evaluation import build_eval_context, build_local_to_emb_row, build_query_gallery_retrieval_state, evaluate_query_gallery_retrieval



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


def build_query_for_setting(
    config: dict,
    ctx_orig,
    setting: str,
) -> dict:
    """
    Builds the query side for one background setting so retrieval can be compared
    against the same fixed original gallery.
    """
    val_processing_fn = build_eval_processing_fn(setting, config)

    _, _, val_ds = load_split_jaguar_from_FO_export(
        PATHS.data_export / "splits_curated",
        overwrite_db=False,
        parquet_path=ctx_orig.parquet_root,
        dataset_name="jaguar_splits_curated",
        train_processing_fn=None,
        val_processing_fn=val_processing_fn,
        include_duplicates=config["split"]["include_duplicates"],
    )

    val_ds.transform = get_transforms(
        config,
        ctx_orig.model.backbone_wrapper,
        is_training=False,
    )

    val_local_to_emb_row = build_local_to_emb_row(
        val_ds,
        ctx_orig.split_df,
        split="val",
    )

    query_embeddings = load_or_extract_jaguarid_embeddings(
        model=ctx_orig.model,
        torch_ds=val_ds,
        split="val",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix=f"{setting}_val",
    )

    query_labels = np.asarray(val_ds.labels)
    query_global_indices = val_local_to_emb_row


    #query_local_idx_setting = map_emb_rows_to_local_indices(
    #    emb_rows=query_emb_rows,
    #    local_to_emb_row=ctx_setting.val_local_to_emb_row,
    #)

    #query_embeddings = val_embeddings_setting[query_local_idx_setting]
    #query_labels = np.asarray(ctx_setting.val_ds.labels)[query_local_idx_setting]
    #query_global_indices = query_emb_rows

    return {
        "query_embeddings": query_embeddings,
        "query_labels": query_labels,
        "query_global_indices": query_global_indices,
        "val_ds": val_ds,
    }


def build_original_gallery_base(config: dict, checkpoint_dir: Path):
    """
    Builds the shared original gallery used as the fixed reference for background-reliance
    comparisons across query settings.
    """
    ctx_orig = build_eval_context(
        config=config,
        checkpoint_dir=checkpoint_dir,
        eval_val_setting="original",
    )

    train_embeddings_orig = load_or_extract_jaguarid_embeddings(
        model=ctx_orig.model,
        torch_ds=ctx_orig.train_ds,
        split="train",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix="original_train",
    )

    ##  !TODO klären welchen background die haben sollten
    val_embeddings_orig = load_or_extract_jaguarid_embeddings(
        model=ctx_orig.model,
        torch_ds=ctx_orig.val_ds,
        split="val",
        batch_size=config["inference"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        use_tta=config["inference"]["use_tta"],
        cache_prefix="original_val",
    )

    gallery_embeddings_orig = np.concatenate(
        [train_embeddings_orig, val_embeddings_orig],
        axis=0,
    )

    gallery_labels_orig = np.concatenate(
        [np.asarray(ctx_orig.train_ds.labels), np.asarray(ctx_orig.val_ds.labels)],
        axis=0,
    )

    gallery_global_indices_orig = np.concatenate(
        [ctx_orig.train_local_to_emb_row, ctx_orig.val_local_to_emb_row],
        axis=0,
    )

    train_files = [str(s["filename"]) for s in ctx_orig.train_ds.samples]
    val_files = [str(s["filename"]) for s in ctx_orig.val_ds.samples]
    gallery_files_orig = train_files + val_files

    #query_emb_rows = get_val_query_indices(
    #    split_df=ctx_orig.split_df,
    #    out_root=save_dir,
    #    n_samples=config["evaluation"]["n_queries"],
    #    seed=config["evaluation"]["seed"],
    #)

    return {
        "ctx_orig": ctx_orig,
        "train_embeddings_orig": train_embeddings_orig,
        "val_embeddings_orig": val_embeddings_orig,
        "gallery_embeddings_orig": gallery_embeddings_orig,
        "gallery_labels_orig": gallery_labels_orig,
        "gallery_global_indices_orig": gallery_global_indices_orig,
        "gallery_files_orig": gallery_files_orig,
    }


def run_bg_vs_jaguar_stress_test(config, save_dir):
    """
    Runs the Jaguar-only versus background-only stress test to see which signal better
    supports retrieval for validation queries.
    """
    checkpoint_dir = Path(config["evaluation"]["checkpoint_dir"])
    ensure_dir(save_dir)

    base = build_original_gallery_base(config=config, checkpoint_dir=checkpoint_dir)
    ctx_orig = base["ctx_orig"]

    query_emb_orig = base["val_embeddings_orig"]
    gallery_emb = base["gallery_embeddings_orig"]
    gallery_ids = list(base["gallery_labels_orig"])
    gallery_files = list(base["gallery_files_orig"])

    query_jag = build_query_for_setting(
        config=config,
        ctx_orig=ctx_orig,
        setting="black_bg", # !TODO placeholder: replace with your jaguar-only processor name
    )
    query_emb_jaguar_only = query_jag["query_embeddings"]

    query_bg = build_query_for_setting(
        config=config,
        ctx_orig=ctx_orig,
        setting="bg_only", # !TODO placeholder: replace with your jaguar-only processor name
    )
    query_emb_bg_only = query_bg["query_embeddings"]

    query_ids = list(ctx_orig.val_ds.labels)
    query_files = [str(s["filename"]) for s in ctx_orig.val_ds.samples]

    retrieval_df = compute_retrieval_bg_vs_jaguar(
        query_emb_orig=query_emb_orig,
        query_emb_jaguar_only=query_emb_jaguar_only,
        query_emb_bg_only=query_emb_bg_only,
        query_ids=query_ids,
        query_files=query_files,
        gallery_emb=gallery_emb,
        gallery_ids=gallery_ids,
        gallery_files=gallery_files,
    )

    out_path = save_dir / "background_ablation_orig_vs_jag_bg.parquet"
    retrieval_df.to_parquet(out_path, index=False)

    return {
        "df": retrieval_df,
        "path": out_path,
    }


def save_retrieval_results(
    save_dir: Path,
    setting: str,
    query_df: pd.DataFrame,
    summary: dict,
) -> dict:
    """
    Saves per-query and summary retrieval outputs for one background setting.
    """
    query_df = query_df.copy()
    query_df["setting"] = setting

    summary = dict(summary)
    summary["setting"] = setting

    setting_dir = save_dir / setting
    ensure_dir(setting_dir)

    query_path = setting_dir / "retrieval_per_query.parquet"
    summary_path = setting_dir / "retrieval_summary.parquet"

    query_df.to_parquet(query_path, index=False)
    pd.DataFrame([summary]).to_parquet(summary_path, index=False)

    return {
        "query_df": query_df, 
        "summary": summary, 
        "query_path": query_path, 
        "summary_path": summary_path,
    }


def aggregate_and_save_background_results(
    all_query_dfs: list[pd.DataFrame],
    all_summaries: list[dict],
    save_dir: Path,
) -> dict:
    """
    Aggregates all background-setting results, computes deltas against the original
    setting, and saves the combined outputs.
    """
    per_query_all = pd.concat(all_query_dfs, ignore_index=True)
    summary_all = pd.DataFrame(all_summaries)

    orig_row = summary_all.loc[summary_all["setting"] == "original"].iloc[0]
    summary_all["delta_mAP_vs_original"] = summary_all["mAP"] - float(orig_row["mAP"])
    summary_all["delta_rank1_vs_original"] = summary_all["rank1"] - float(orig_row["rank1"])

    orig_per_query = (
        per_query_all[per_query_all["setting"] == "original"][
            ["query_idx", "ap", "rank1_correct", "first_pos_rank", "top1_idx", "top1_label", "top1_sim"]
        ]
        .rename(
            columns={
                "ap": "ap_original",
                "rank1_correct": "rank1_correct_original",
                "first_pos_rank": "first_pos_rank_original",
                "top1_idx": "top1_idx_original",
                "top1_label": "top1_label_original",
                "top1_sim": "top1_sim_original",
            }
        )
        .copy()
    )

    per_query_delta = per_query_all.merge(orig_per_query, on="query_idx", how="left")

    per_query_delta["delta_ap_vs_original"] = (
        per_query_delta["ap"] - per_query_delta["ap_original"]
    )

    per_query_delta["rank1_flip_vs_original"] = (
        per_query_delta["rank1_correct"] != per_query_delta["rank1_correct_original"]
    )

    per_query_delta["delta_first_pos_rank_vs_original"] = (
        per_query_delta["first_pos_rank"] - per_query_delta["first_pos_rank_original"]
    )

    per_query_path = save_dir / "background_per_query_all.parquet"
    summary_path = save_dir / "background_summary_all.parquet"
    per_query_delta_path = save_dir / "background_per_query_delta_vs_original.parquet"

    per_query_all.to_parquet(per_query_path, index=False)
    summary_all.to_parquet(summary_path, index=False)
    per_query_delta.to_parquet(per_query_delta_path, index=False)

    return {
        "per_query_all": per_query_all,
        "per_query_delta": per_query_delta,
        "summary_all": summary_all,
        "summary_path": summary_path,
        "per_query_path": per_query_path,
        "per_query_delta_path": per_query_delta_path,
    }


def run_background_reliance_eval(config, save_dir):
    """
    Runs the full background-reliance evaluation by comparing several query background
    settings against the same original gallery.
    """
    checkpoint_dir = Path(config["evaluation"]["checkpoint_dir"])
    ensure_dir(save_dir)


    #! TODO ight go into evaluation runner
    exp_name = config["evaluation"]["experiment_name"]
    experiment_group = config.get("output", {}).get("experiment_group")

    wandb_run = init_wandb_run(
        config=config,
        run_dir=save_dir,
        exp_name=exp_name,
        experiment_group=experiment_group,
        job_type="eval",
    )

    base = build_original_gallery_base(config=config, checkpoint_dir=checkpoint_dir)

    ctx_orig = base["ctx_orig"]
    gallery_embeddings_orig = base["gallery_embeddings_orig"]
    gallery_labels_orig = base["gallery_labels_orig"]
    gallery_global_indices_orig = base["gallery_global_indices_orig"]

    settings = ["original", "gray_bg", "blur_bg", "black_bg"]

    if "original" not in settings:
        raise ValueError(
            "'original' must be included in settings because all summary and per-query deltas are computed against it."
        )

    all_query_dfs = []
    all_summaries = []

    for setting in settings:
        query = build_query_for_setting(
            config=config,
            ctx_orig=ctx_orig,
            setting=setting,
            #query_emb_rows=query_emb_rows,
        )

        retrieval = build_query_gallery_retrieval_state(
                query_embeddings=query["query_embeddings"],
                gallery_embeddings=gallery_embeddings_orig,
                query_global_indices=query["query_global_indices"],
                gallery_global_indices=gallery_global_indices_orig,
                query_labels=query["query_labels"],
                gallery_labels=gallery_labels_orig,
                split_df=ctx_orig.split_df,
            )
        
        query_df, summary = evaluate_query_gallery_retrieval(retrieval)
        result = save_retrieval_results(save_dir, setting, query_df, summary)
        
        all_query_dfs.append(result["query_df"])
        all_summaries.append(result["summary"])

    result = aggregate_and_save_background_results(
        all_query_dfs=all_query_dfs,
        all_summaries=all_summaries,
        save_dir=save_dir,
    )

    return result


if __name__ == "__main__":
    config = ""
    save_dir = ""
    main_result = run_background_reliance_eval(config, save_dir)
    stress_result = run_bg_vs_jaguar_stress_test(config, save_dir)