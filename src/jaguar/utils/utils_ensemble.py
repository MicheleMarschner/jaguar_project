import pandas as pd

def build_ensemble_results_long_df(
    exp_name: str,
    experiment_group: str | None,
    gallery_protocol: str,
    per_model_metrics: dict[str, dict],
    fusion_metrics: dict[str, dict],
    oracle_summary: dict,
) -> pd.DataFrame:
    """Build one long-form results table for a single ensemble run."""
    rows = []

    member_names = list(per_model_metrics.keys())
    member_set = "+".join(member_names)
    n_members = len(member_names)

    best_single_map = max(float(metrics["mAP"]) for metrics in per_model_metrics.values())
    oracle_map = float(oracle_summary["oracle_mAP"])
    oracle_rank1 = float(oracle_summary["oracle_rank1"])

    for model_name, metrics in per_model_metrics.items():
        model_map = float(metrics["mAP"])
        model_rank1 = float(metrics["rank1"])

        rows.append({
            "experiment_name": exp_name,
            "experiment_group": experiment_group,
            "gallery_protocol": gallery_protocol,
            "method_name": model_name,
            "method_type": "single",
            "member_set": member_set,
            "n_members": n_members,
            "mAP": model_map,
            "rank1": model_rank1,
            "best_single_mAP": best_single_map,
            "gain_vs_best_single": model_map - best_single_map,
            "oracle_mAP": oracle_map,
            "oracle_rank1": oracle_rank1,
            "oracle_gap_mAP": oracle_map - model_map,
        })

    for fusion_name, metrics in fusion_metrics.items():
        fusion_map = float(metrics["mAP"])
        fusion_rank1 = float(metrics["rank1"])

        rows.append({
            "experiment_name": exp_name,
            "experiment_group": experiment_group,
            "gallery_protocol": gallery_protocol,
            "method_name": fusion_name,
            "method_type": "fusion",
            "member_set": member_set,
            "n_members": n_members,
            "mAP": fusion_map,
            "rank1": fusion_rank1,
            "best_single_mAP": best_single_map,
            "gain_vs_best_single": fusion_map - best_single_map,
            "oracle_mAP": oracle_map,
            "oracle_rank1": oracle_rank1,
            "oracle_gap_mAP": oracle_map - fusion_map,
        })

    rows.append({
        "experiment_name": exp_name,
        "experiment_group": experiment_group,
        "gallery_protocol": gallery_protocol,
        "method_name": "oracle",
        "method_type": "oracle",
        "member_set": member_set,
        "n_members": n_members,
        "mAP": oracle_map,
        "rank1": oracle_rank1,
        "best_single_mAP": best_single_map,
        "gain_vs_best_single": oracle_map - best_single_map,
        "oracle_mAP": oracle_map,
        "oracle_rank1": oracle_rank1,
        "oracle_gap_mAP": 0.0,
    })

    return pd.DataFrame(rows)

def build_protocol_comparison_df(results_long_df: pd.DataFrame) -> pd.DataFrame:
    """Build a protocol comparison table from long-form ensemble results."""
    metrics_to_compare = [
        "mAP",
        "rank1",
        "gain_vs_best_single",
        "oracle_gap_mAP",
    ]

    pieces = []

    for metric_name in metrics_to_compare:
        pivot = results_long_df.pivot_table(
            index=["member_set", "method_name", "method_type"],
            columns="gallery_protocol",
            values=metric_name,
            aggfunc="first",
        ).reset_index()

        value_cols = [
            c for c in pivot.columns
            if c not in {"member_set", "method_name", "method_type"}
        ]
        pivot = pivot.rename(
            columns={col: f"{metric_name}__{col}" for col in value_cols}
        )

        trainval_col = f"{metric_name}__trainval_gallery"
        valonly_col = f"{metric_name}__valonly_gallery"

        if trainval_col in pivot.columns and valonly_col in pivot.columns:
            pivot[f"{metric_name}__mean_across_protocols"] = (
                pivot[trainval_col] + pivot[valonly_col]
            ) / 2.0
            pivot[f"{metric_name}__delta_protocols"] = (
                pivot[trainval_col] - pivot[valonly_col]
            )

        pieces.append(pivot)

    comparison_df = pieces[0]
    for piece in pieces[1:]:
        comparison_df = comparison_df.merge(
            piece,
            on=["member_set", "method_name", "method_type"],
            how="outer",
        )

    return comparison_df