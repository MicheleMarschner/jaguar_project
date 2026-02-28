import json
from pathlib import Path
from typing import Literal
import pandas as pd

from jaguar.utils.utils import json_default, save_parquet

# ============================================================
# Load exported manifest into metadata table (one row per image)
# ============================================================
def build_split_table_from_torch_dataset(torch_ds) -> pd.DataFrame:
    """
    Build one-row-per-image table from JaguarDataset manifest (source of truth).
    Flatten per-sample manifest entries into a tabular split table so later split logic works with explicit metadata 
    columns (identity, dedup tags, file references).
    """
    rows = []

    for i, s in enumerate(torch_ds.samples):
        gt = s.get("ground_truth", {}) if isinstance(s, dict) else {}
        label = gt.get("label", None)

        row = {
            "emb_row": i,  # stable row index to align with dataset-order-based artifacts if needed
            "filepath": str(torch_ds._resolve_path(s["filepath"])),
            "filename": s.get("filename", Path(s["filepath"]).name),
            "identity_id": (str(label) if label is not None else None),
            "split_original": s.get("split", None),

            # dedup fields if present
            "burst_group_id": s.get("burst_group_id", None),
            "burst_cluster_size": s.get("burst_cluster_size", None),
            "burst_role": s.get("burst_role", None),  # singleton / burst_member
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # normalize missing burst roles (if field absent)
    if "burst_role" in df.columns:
        df["burst_role"] = df["burst_role"].fillna("singleton")

    return df


# ============================================================
# logging helpers
# ============================================================

def print_keep_drop_summary(df: pd.DataFrame, split_col="split_tmp", keep_col="keep_curated"):
    base = df.groupby(split_col).size().rename("n_total")
    
    kept = df[df[keep_col]].groupby(split_col).size().rename("n_kept")
    dropped = df[~df[keep_col]].groupby(split_col).size().rename("n_dropped")
    out = pd.concat([base, kept, dropped], axis=1).fillna(0).astype(int)
    out["keep_rate"] = (out["n_kept"] / out["n_total"]).round(4)
    
    print("\n[Summary] Keep vs Drop per split")
    print(out.sort_index()) 

    return out


def build_burst_delta_table(
    df: pd.DataFrame,
    split_col="split_tmp",
    burst_col="burst_group_id",
    keep_col="keep_curated",
    role_col="burst_role",
) -> pd.DataFrame:
    # only burst rows (exclude singletons)
    burst_df = df[(df[role_col] != "singleton") & (df[burst_col].notna())].copy()

    g = burst_df.groupby([split_col, burst_col], dropna=False)

    out = g.agg(
        n_total=("filepath", "size"),
        n_kept=(keep_col, "sum"),
    ).reset_index()

    out["n_dropped"] = out["n_total"] - out["n_kept"]
    out["keep_rate"] = (out["n_kept"] / out["n_total"]).round(4)
    return out.sort_values(["split_tmp", "n_total"], ascending=[True, False]).reset_index(drop=True)

def print_top_changed_bursts(df: pd.DataFrame, top_n: int = 20):
    burst_delta = build_burst_delta_table(df)
    print("\n[Bursts] Top bursts by #dropped")
    print(burst_delta.sort_values("n_dropped", ascending=False).head(top_n))
    
# ============================================================
# Save artifacts
# ============================================================

def summarize_splits(df, split_col="split_tmp", keep_col="keep_curated", id_col="identity_id"):
    def get_stats(subset_df):
        total_samples = len(subset_df)
        total_ids = subset_df[id_col].nunique()
        
        splits_info = {}
        # Ensure we only iterate over present splits
        for split in subset_df[split_col].unique():
            split_df = subset_df[subset_df[split_col] == split]
            count = len(split_df)
            splits_info[split] = {
                "samples": int(count),
                "sample_percentage": round(count / total_samples, 4) if total_samples > 0 else 0,
                "identities": int(split_df[id_col].nunique())
            }
        return total_samples, total_ids, splits_info

    # 1. Raw Stats
    raw_total, raw_ids, raw_splits = get_stats(df)

    # 2. Curated Stats
    curated_df = df[df[keep_col] == True]
    curated_total, curated_ids, curated_splits = get_stats(curated_df)

    # 3. Combine into JSON structure
    summary_config = {
        "raw_data": {
            "total_samples": raw_total,
            "total_identities": raw_ids,
            "splits": raw_splits
        },
        "curated_data": {
            "total_samples": curated_total,
            "total_identities": curated_ids,
            "splits": curated_splits
        }
    }
    
    with open('summary_config.json', 'w') as f:
        json.dump(summary_config, f, indent=4)
    
    return summary_config


def save_split_bundle(out_root: Path, final_df: pd.DataFrame, config: dict):
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) FULL audit table
    full_path = out_root / f"full_split.parquet"
    save_parquet(full_path, final_df)

    # 2) Curated training manifest
    curated_df = final_df[final_df["keep_curated"]].copy()
    curated_path = out_root / f"curated_split.parquet"
    save_parquet(curated_path, curated_df)

    # 3) Burst delta summary table (small + super useful)
    burst_delta = build_burst_delta_table(final_df)
    burst_path = out_root / f"burst_delta.parquet"
    save_parquet(burst_path, burst_delta)

    # 4) Config JSON
    cfg_path = out_root / f"config.json"
    cfg_path.write_text(json.dumps(config, indent=2, default=json_default))