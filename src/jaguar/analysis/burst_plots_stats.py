from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from jaguar.config import PATHS


def get_largest_burst_group(artifacts_dir):
    df = pd.read_parquet(artifacts_dir / "burst_assignments.parquet")

    # only burst members
    x = df[df["burst_group_id"].notna()].copy()

    # pick one burst (or change [0] -> [1], etc.)
    gid = x["burst_group_id"].value_counts().index[0]
    burst_df = x[x["burst_group_id"] == gid].copy()

    # put representative first (leftmost), then others
    burst_df["is_rep"] = (burst_df["burst_role"] == "representative").astype(int)
    burst_df = burst_df.sort_values(["is_rep", "filepath"], ascending=[False, True]).copy()

    # jaguar name + burst members
    jaguar_name = burst_df["identity_id"].iloc[0] if "identity_id" in burst_df.columns else "unknown"
    print("Jaguar:", jaguar_name)
    print("Burst group:", gid)

    return burst_df


def plot_burst(data_root, df, save_dir, filename=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    jaguar_name = str(df["identity_id"].iloc[0]) if "identity_id" in df.columns and len(df) else "unknown"
    burst_size = len(df)

    show = df.head(5).copy()
    n = len(show)

    plt.figure(figsize=(2.0 * n, 2.8))  

    for i, r in enumerate(show.itertuples(index=False), start=1):
        fp = Path(r.filepath)
        if not fp.exists() and data_root is not None:
            fp = Path(data_root) / fp.name  

        with Image.open(fp) as img:
            plt.subplot(1, n, i)
            plt.imshow(img.convert("RGB"))
            role = "REP" if r.burst_role == "representative" else "DUP"

            # put jaguar + burst size into first image title (no extra top gap)
            if i == 1:
                plt.title(
                    f"{jaguar_name} | n={burst_size}\n{role}\n{Path(r.filepath).name}",
                    fontsize=8, pad=2
                )
            else:
                plt.title(f"{role}\n{Path(r.filepath).name}", fontsize=8, pad=2)

            plt.axis("off")

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0.02)

    if filename is None:
        filename = f"burst_top5__{jaguar_name}__n{burst_size}.png"

    out_fp = save_dir / filename
    plt.savefig(out_fp, dpi=200, bbox_inches="tight")
    plt.close()
    return out_fp


def compute_dedup_stats(artifacts_dir, save_dir):
    df = pd.read_parquet(artifacts_dir / "burst_assignments.parquet").copy()

    # ---- Flags ----
    # in any burst group (all members incl. representative)
    df["is_in_burst"] = df["burst_group_id"].notna()

    # actual duplicate-to-remove candidates (exclude representative)
    df["is_burst_duplicate"] = df["is_in_burst"] & (df["burst_role"] != "representative")

    # Choose identity column + filename column used in your artifacts
    id_col = "identity_id"      # or "ground_truth" if your artifact has that instead
    file_col = "filepath"       # or "filename" depending on artifact schema

    # Group by jaguar
    jaguar_stats = (
        df.groupby(id_col)
        .agg(
            total_images=(file_col, "count"),
            duplicate_images=("is_burst_duplicate", "sum") # duplicates excluding kept rep
        )
        .reset_index()
    )
    jaguar_stats["kept_images"] = jaguar_stats["total_images"] - jaguar_stats["duplicate_images"]
    jaguar_stats = jaguar_stats.sort_values("total_images", ascending=False)

    out_fp = save_dir / "dedup_stats_per_jaguar.csv"
    jaguar_stats.to_csv(out_fp, index=False)
    print(f"Saved: {out_fp}")

    return jaguar_stats


def dedup_stats_plot(jaguar_stats, save_dir, filename="dedup_stats_per_jaguar.png"):
    save_dir = Path(save_dir)

    id_col = "identity_id"

    plt.figure(figsize=(14, 8))
    plt.bar(jaguar_stats[id_col], jaguar_stats["kept_images"], label="Kept Images", color="skyblue")
    plt.bar(
        jaguar_stats[id_col],
        jaguar_stats["duplicate_images"],
        bottom=jaguar_stats["kept_images"],
        label="Burst Duplicates (removed)",
        color="orange",
    )

    plt.title("Kept Images vs Burst Duplicates per Jaguar", fontsize=16)
    plt.xlabel("Jaguar Identity", fontsize=12)
    plt.ylabel("Image Count", fontsize=12)
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()

    out_fp = save_dir / filename
    plt.savefig(out_fp, dpi=200, bbox_inches="tight")

    return out_fp


if __name__ == "__main__":
    model_name = "Megadescriptor-L".lower()
    artifacts_dir = PATHS.runs / "deduplication" / f"dedup__{model_name}" / f"dedup_final__{model_name}__sim0.95_ph4__or__minSize2"
    img_root = PATHS.data_train
    save_dir = PATHS.results / "bursts"

    burst_df = get_largest_burst_group(artifacts_dir)
    burst_plot = plot_burst(data_root=img_root, df=burst_df, save_dir=save_dir)

    jaguar_stats = compute_dedup_stats(artifacts_dir, save_dir)
    stats_plot = dedup_stats_plot(jaguar_stats, save_dir)