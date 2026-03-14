import pandas as pd
from pathlib import Path

from jaguar.analysis.xai_metrics_analysis import run_xai_class_metrics_analysis
from jaguar.config import PATHS

def build_class_xai_main_table(df_vec: pd.DataFrame) -> pd.DataFrame:
    """
    Main report table:
    explainer × group with sanity / topk / random / gap means
    """
    sub = df_vec[df_vec["metric"].isin(["sanity", "faith_topk", "faith_random", "faith_gap"])].copy()

    summary = (
        sub.groupby(["run_id", "explainer", "group", "metric"])["value"]
        .mean()
        .reset_index()
        .pivot(
            index=["run_id", "explainer", "group"],
            columns="metric",
            values="value",
        )
        .reset_index()
    )

    return summary.rename(columns={
        "sanity": "sanity_mean",
        "faith_topk": "mean_topk_drop",
        "faith_random": "mean_random_drop",
        "faith_gap": "faithfulness_gap",
    }).sort_values(["run_id", "explainer", "group"]).reset_index(drop=True)


def save_class_xai_main_table(df_vec: pd.DataFrame, save_dir: Path) -> Path:
    table = build_class_xai_main_table(df_vec)
    out = save_dir / "xai_class_main_table.csv"
    table.to_csv(out, index=False)
    return out


def run(
    config: dict,
    save_dir: Path, 
    root_dir: Path | None = None,
    run_dir: Path | None = None,
    exemplar_run_dir: Path | None = None,
    **kwargs,
) -> None:
    
    save_root = PATHS.results / "xai_metrics"
        
    run_xai_class_metrics_analysis(
        run_root=run_dir,
        save_root=save_root,
    )