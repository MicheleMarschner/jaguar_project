from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from jaguar.config import EXPERIMENTS_STORE, PATHS
from jaguar.utils.utils import ensure_dir, resolve_path


def get_largest_burst_group(artifacts_dir):
    df = pd.read_parquet(artifacts_dir / "burst_assignments.parquet")

    # only burst members
    x = df[df["burst_group_id"].notna()].copy()

    # pick one burst (or change [0] -> [1], etc.)
    gid = x["burst_group_id"].value_counts().index[0]
    burst_df = x[x["burst_group_id"] == gid].copy()

    burst_df = burst_df.sort_values(["filename"], ascending=[True]).copy()

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
        fp = Path(r.filename)
        if not fp.exists() and data_root is not None:
            fp = Path(data_root) / Path(r.filename).name  

        with Image.open(fp) as img:
            plt.subplot(1, n, i)
            plt.imshow(img.convert("RGB"))

            # put jaguar + burst size into first image title (no extra top gap)
            if i == 1:
                plt.title(
                    f"{jaguar_name} | n={burst_size}\n{Path(r.filename).name}",
                    fontsize=8, pad=2
                )
            else:
                plt.title(f"{Path(r.filename).name}", fontsize=8, pad=2)

            plt.axis("off")

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0.02)

    if filename is None:
        filename = f"burst_top5__{jaguar_name}__n{burst_size}.png"

    out_fp = save_dir / filename
    plt.savefig(out_fp, dpi=200, bbox_inches="tight")
    plt.close()
    return out_fp


def compute_burst_stats(artifacts_dir: Path, save_dir: Path):
    df = pd.read_parquet(artifacts_dir / "burst_assignments.parquet").copy()
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    df["is_in_burst"] = df["burst_group_id"].notna()

    id_col = "identity_id"
    file_col = "filename"

    jaguar_stats = (
        df.groupby(id_col)
          .agg(
              total_images=(file_col, "count"),
              burst_images=("is_in_burst", "sum"),
          )
          .reset_index()
    )
    jaguar_stats["singleton_images"] = jaguar_stats["total_images"] - jaguar_stats["burst_images"]
    jaguar_stats = jaguar_stats.sort_values("total_images", ascending=False)

    out_fp = save_dir / "burst_stats_per_jaguar.csv"
    jaguar_stats.to_csv(out_fp, index=False)
    print(f"Saved: {out_fp}")

    return jaguar_stats, out_fp


def plot_burst_stats(jaguar_stats: pd.DataFrame, save_dir: Path, filename="burst_per_jaguar.png"):
    save_dir = Path(save_dir)

    id_col = "identity_id"
    plt.figure(figsize=(14, 8))
    plt.bar(jaguar_stats[id_col], jaguar_stats["singleton_images"], label="Singleton Images")
    plt.bar(
        jaguar_stats[id_col],
        jaguar_stats["burst_images"],
        bottom=jaguar_stats["singleton_images"],
        label="Burst Members",
    )

    plt.title("Singleton vs Burst-Member Images per Jaguar", fontsize=16)
    plt.xlabel("Jaguar Identity", fontsize=12)
    plt.ylabel("Image Count", fontsize=12)
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()

    out_fp = save_dir / filename
    plt.savefig(out_fp, dpi=200, bbox_inches="tight")
    plt.close()

    return out_fp


def run_burst_analysis(
    artifacts_dir: Path,
    save_dir: Path,
    img_root: Path,
) -> dict[str, Path]:
    
    ensure_dir(save_dir)

    burst_df = get_largest_burst_group(artifacts_dir)
    burst_plot_path = plot_burst(
        data_root=img_root,
        df=burst_df,
        save_dir=save_dir,
    )

    jaguar_stats, stats_csv_path = compute_burst_stats(
        artifacts_dir=artifacts_dir,
        save_dir=save_dir,
    )
    stats_plot_path = plot_burst_stats(
        jaguar_stats=jaguar_stats,
        save_dir=save_dir,
    )

    return {
        "burst_top5_plot": burst_plot_path,
        "burst_stats_csv": stats_csv_path,
        "burst_stats_plot": stats_plot_path,
    }