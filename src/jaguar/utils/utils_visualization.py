from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import math
from typing import Any, Optional, Sequence

from jaguar.utils.utils import denormalize_image, ensure_dir, resolve_path, tensor_img_to_hwc01
from jaguar.utils.utils_xai_similarity import normalize_heatmap
from jaguar.config import DATA_STORE, IMGNET_MEAN, IMGNET_STD, PATHS, USE_FIFTYONE
from jaguar.utils.utils_datasets import load_full_jaguar_from_FO_export

sns.set_theme(style="whitegrid", palette="muted")

# ============================================================
# EDA Plots
# ============================================================

def plot_image_dimensions(stats_df: pd.DataFrame, save_path: Optional[Path] = None, show: bool = False) -> None:
    fig = plt.figure(figsize=(8, 4))
    plt.hist(stats_df["width"], bins=50, alpha=0.6, label="width")
    plt.hist(stats_df["height"], bins=50, alpha=0.6, label="height")
    plt.legend()
    plt.title("Distribution of image widths and heights (train)")
    plt.xlabel("Pixels")
    plt.ylabel("Count")
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# Left: per-identity counts (sorted) to reveal long-tail imbalance.
# Right: histogram of class sizes to summarize the distribution shape.
def plot_identity_distribution(counts: pd.Series, save_path: Path) -> None:
    plt.figure(figsize=(12, 6))

    # Bar chart
    ax = plt.subplot(1, 2, 1)
    color_palette = ['red' if x < 20 else 'skyblue' for x in counts.values]
    sns.barplot(x=counts.values, y=counts.index, palette=color_palette, ax=ax)

    ax.set_title("Training Data: Image Count per Jaguar", fontsize=14)
    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Class index (sorted)")
    ax.tick_params(axis='y', rotation=30)

    # for horizontal bars, median count should be a VERTICAL line (x-axis), not hline
    ax.axvline(x=counts.median(), color='red', linestyle='--', label=f'Median: {counts.median():.1f}')
    for p in ax.patches:
        w = p.get_width()   # horizontal bar length
        y = p.get_y() + p.get_height() / 2
        ax.annotate(
            f"{int(w)}",
            (w, y),
            ha="left",
            va="center",
            fontsize=8,
            xytext=(3, 0),
            textcoords="offset points",
        )
    ax.legend()

    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(counts.values, bins=15, kde=True, color="darkgreen", alpha=0.5)
    plt.title("Distribution of Class Sizes", fontsize=14)
    plt.xlabel("Images per Jaguar")
    plt.ylabel("Count")

    plt.tight_layout()

    ensure_dir(save_path.parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


# Sharpness spans orders of magnitude, so we use a log-scaled x-axis and log-spaced bins.
def sharpness_histogramm(artifacts_dir, save_dir, filename="sharpness_histogram.png"):
    df = pd.read_parquet(artifacts_dir / "meta_img_features.parquet").copy()

    x = df["sharpness"].dropna()
    x = x[x >= 0]  
    median = x.median()
    p95 = x.quantile(0.95)

    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 70)  

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(x, bins=bins, kde=False, ax=ax)

    ax.set_xscale("log")
    ax.set_title("Sharpness Histogram (log x-scale)")
    ax.set_xlabel("Sharpness (log scale)")
    ax.set_ylabel("Count")

    ax.axvline(median, color="green", linestyle="--", linewidth=2, label=f"median={median:.1f}")
    ax.axvline(p95, color="green", linestyle=":", linewidth=2, label=f"p95={p95:.1f}")
    ax.legend(frameon=False)

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:g}"))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 3, 5)))
    ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, pos: f"{v:g}"))

    ax.tick_params(axis="x", which="major", labelsize=10, length=4)
    ax.tick_params(axis="x", which="minor", labelsize=10, length=2)

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_dir/filename, dpi=200, bbox_inches="tight")


# Resolution also spans a wide range; log-x avoids compressing low-resolution bins.
def plot_resolution_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    save_path: str | Path | None = None,
    bins_n: int = 50,
    add_quantile_lines: bool = False,
):
    """
    Log-x histogram with log-spaced bins.
    """
    x = pd.to_numeric(values, errors="coerce").dropna()
    x = x[x > 0]
    if len(x) == 0:
        raise ValueError("No positive values to plot on log scale.")

    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), bins_n)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(x, bins=bins, kde=False, ax=ax)

    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    # log ticks
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, pos: f"{int(v):,}" if v >= 1 else f"{v:g}")
    )
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 3, 5)))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.grid(False)

    if add_quantile_lines:
        median = float(x.median())
        p95 = float(x.quantile(0.95))
        ax.axvline(median, color="green", linestyle="--", linewidth=2, label=f"median={median:.1f}")
        ax.axvline(p95, color="green", linestyle=":", linewidth=2, label=f"p95={p95:.1f}")
        ax.legend(frameon=False)

    plt.tight_layout()

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[Plot] Saved to: {save_path}")

    return fig, ax


# Qualitative inspection helper: visualize extreme examples with identity + size metadata.
def show_image_gallery(
    df_subset: pd.DataFrame,
    image_root: str | Path | None = None,   # NEW
    title: str = "",
    n_cols: int = 5,
    figsize_per_img: float = 3.2,
    save_path: str | Path | None = None,
):
    df_subset = df_subset.reset_index(drop=True)
    n_rows = math.ceil(len(df_subset) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_img, n_rows * figsize_per_img))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()

    for ax in axes[len(df_subset):]:
        ax.axis("off")

    for i, (_, row) in enumerate(df_subset.iterrows()):
        ax = axes[i]
        fname = str(row.get("filename"))
        fp = image_root / fname

        with Image.open(fp) as img:
            img = img.convert("RGB")
            ax.imshow(img)

        fname = str(row.get("filename", fp.name))
        jag_id = str(row.get("identity_id", "NA"))

        title_line = jag_id
        title_line += f" | {int(row['width'])}×{int(row['height'])}"
        title_line += f" | {int(row['resolution_px']):,} px"

        ax.set_title(title_line, fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(top=0.88, wspace=0.05, hspace=0.35)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[Gallery] Saved to: {save_path}")



# ============================================================
# Interpretability Plots
# ============================================================


def plot_ig_triplets_from_pt(
    pt_path,
    indices=None,            # list of positions INSIDE saved tensor, e.g. [0,1,2]
    max_rows=3,
    figsize_per_row=(12, 4),
    mean=IMGNET_MEAN,
    std=IMGNET_STD,
    cmap="jet",
    alpha=0.45,
):
    """
    Expected saved dict keys (your format):
      - 'saliency': [N,H,W]
      - 'X_query' : [N,3,H,W] normalized tensors
      - optional: 'query_indices', 'ref_indices', 'pair_sims'
    """
    data = torch.load(pt_path, map_location="cpu")

    # support your older names too, if needed
    sal = data.get("saliency", data.get("sal"))
    X = data.get("X_query", data.get("X"))

    if sal is None or X is None:
        raise KeyError("Could not find saliency/X_query in .pt file (also checked sal/X).")

    N = sal.shape[0]
    if indices is None:
        indices = list(range(min(N, max_rows)))
    else:
        indices = list(indices)[:max_rows]

    n_rows = len(indices)
    if n_rows == 0:
        raise ValueError("No indices selected for plotting.")

    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows),
        squeeze=False
    )

    query_idx_tensor = data.get("query_indices", None)
    ref_idx_tensor = data.get("ref_indices", None)
    pair_sims_tensor = data.get("pair_sims", None)

    for row, k in enumerate(indices):
        img = denormalize_image(X[k], mean=mean, std=std)     # [H,W,3]
        h = normalize_heatmap(sal[k])                         # [H,W]
        ov = overlay_heatmap_on_image(img, h, alpha=alpha, cmap=cmap)

        # titles with metadata if available
        meta = []
        if query_idx_tensor is not None:
            meta.append(f"q={int(query_idx_tensor[k])}")
        if ref_idx_tensor is not None:
            meta.append(f"r={int(ref_idx_tensor[k])}")
        if pair_sims_tensor is not None:
            meta.append(f"sim={float(pair_sims_tensor[k]):.3f}")
        meta_str = " | ".join(meta)

        # col 1: original
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Original\n{meta_str}" if meta_str else "Original")
        axes[row, 0].axis("off")

        # col 2: heatmap
        axes[row, 1].imshow(h, cmap=cmap)
        axes[row, 1].set_title("IG Heatmap")
        axes[row, 1].axis("off")

        # col 3: overlay
        axes[row, 2].imshow(ov)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()


def overlay_heatmap_on_image(img_rgb, heatmap, alpha=0.45, cmap="jet"):
    """
    img_rgb: [H,W,3] in [0,1]
    heatmap: [H,W] in [0,1]
    returns overlay [H,W,3] in [0,1]
    """
    cmap_fn = plt.get_cmap(cmap)
    heat_rgb = cmap_fn(heatmap)[..., :3]  # drop alpha channel
    overlay = (1 - alpha) * img_rgb + alpha * heat_rgb
    return np.clip(overlay, 0, 1)


def _resolve_rows_to_plot(
    n_available: int,
    indices: Optional[Sequence[int]] = None,
    max_rows: int = 3,
) -> list[int]:
    """
    indices are positions in the saved .pt arrays (0..N-1), not dataset indices.
    """
    if n_available <= 0:
        raise ValueError("No rows available in saved .pt file.")

    if indices is None:
        return list(range(min(max_rows, n_available)))

    rows = [int(i) for i in indices]
    for r in rows:
        if r < 0 or r >= n_available:
            raise IndexError(f"Requested row {r} out of range [0, {n_available-1}]")
    return rows


def plot_similarity_ig_quad_from_pt(
    pt_path: str | Path,
    torch_ds: Optional[Any] = None,
    indices: Optional[Sequence[int]] = None,   # positions in saved arrays
    max_rows: int = 3,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    heatmap_cmap: str = "jet",
    overlay_alpha: float = 0.35,
    normalize_each_heatmap: bool = True,
    fig_scale: float = 4.2,
    title: Optional[str] = None,
):
    """
    Plots rows of:
      Query | Reference | IG Heatmap | Overlay

    Expected saved fields (best case):
      - 'saliency'   : [N,H,W] tensor
      - 'X_query'    : [N,3,H,W] normalized tensor (optional if torch_ds provided)
      - 'query_indices': [N] tensor (optional but recommended)
      - 'ref_indices'  : [N] tensor (optional but recommended)
      - 'pair_sims'    : [N] tensor (optional)

    If X_query is missing, query image is loaded from torch_ds using query_indices.
    Reference image is loaded from torch_ds using ref_indices (if available).
    """
    pt_path = Path(pt_path)
    data = torch.load(pt_path, map_location="cpu")

    # --- read keys robustly ---
    sal = data.get("saliency", data.get("sal"))
    if sal is None:
        raise KeyError("Saved .pt must contain 'saliency' (or legacy 'sal').")

    sal = torch.as_tensor(sal).cpu()  # [N,H,W]
    if sal.ndim != 3:
        raise ValueError(f"Expected saliency [N,H,W], got {tuple(sal.shape)}")

    N = int(sal.shape[0])

    X_query = data.get("X_query", data.get("X"))
    if X_query is not None:
        X_query = torch.as_tensor(X_query).cpu()

        # Handle accidental single-sample saves like [3,H,W] with saliency [1,H,W]
        if X_query.ndim == 3 and N == 1:
            X_query = X_query.unsqueeze(0)

        if X_query.ndim != 4:
            raise ValueError(f"Expected X_query [N,3,H,W], got {tuple(X_query.shape)}")

        if X_query.shape[0] != N:
            raise ValueError(
                f"Mismatch: saliency has N={N}, but X_query has N={X_query.shape[0]}"
            )

    q_idx = data.get("query_indices", None)
    r_idx = data.get("ref_indices", None)
    pair_sims = data.get("pair_sims", None)

    if q_idx is not None:
        q_idx = torch.as_tensor(q_idx).cpu().numpy().astype(int).reshape(-1)
        if len(q_idx) != N:
            raise ValueError(f"query_indices length {len(q_idx)} != N={N}")

    if r_idx is not None:
        r_idx = torch.as_tensor(r_idx).cpu().numpy().astype(int).reshape(-1)
        if len(r_idx) != N:
            raise ValueError(f"ref_indices length {len(r_idx)} != N={N}")

    if pair_sims is not None:
        pair_sims = torch.as_tensor(pair_sims).cpu().numpy().astype(float).reshape(-1)
        if len(pair_sims) != N:
            raise ValueError(f"pair_sims length {len(pair_sims)} != N={N}")

    rows = _resolve_rows_to_plot(N, indices=indices, max_rows=max_rows)
    n_rows = len(rows)

    # --- figure + axes (always 2D) ---
    fig_w = 4 * fig_scale + 1.2  # extra space for cbar
    fig_h = max(1, n_rows) * fig_scale
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=False
    )

    # Keep reference to one heatmap image for shared colorbar
    hm_im = None

    for row_pos, k in enumerate(rows):
        # ---------- query image ----------
        if X_query is not None:
            q_img = denormalize_image(X_query[k], mean=mean, std=std)  # [H,W,3]
        else:
            if torch_ds is None or q_idx is None:
                raise ValueError(
                    "Need either saved 'X_query' OR (torch_ds + query_indices) to render query image."
                )
            q_sample = torch_ds[int(q_idx[k])]
            q_img = tensor_img_to_hwc01(q_sample["img"])

        # ---------- reference image ----------
        ref_img = None
        if torch_ds is not None and r_idx is not None:
            r_sample = torch_ds[int(r_idx[k])]
            ref_img = tensor_img_to_hwc01(r_sample["img"])

        # If no reference image available, show blank panel with text
        if ref_img is None:
            ref_img = np.ones_like(q_img) * 0.95

        # ---------- heatmap + overlay ----------
        heat = sal[k].detach().cpu().numpy().astype(np.float32)
        heat_vis = normalize_heatmap(heat) if normalize_each_heatmap else heat
        overlay = overlay_heatmap_on_image(q_img, heat_vis, cmap=heatmap_cmap, alpha=overlay_alpha)

        # ---------- plot row ----------
        ax0, ax1, ax2, ax3 = axes[row_pos]

        ax0.imshow(q_img)
        ax0.set_title("Query", fontsize=12)

        ax1.imshow(ref_img)
        ax1.set_title("Reference", fontsize=12)

        hm_im = ax2.imshow(heat_vis, cmap=heatmap_cmap, vmin=0.0, vmax=1.0)
        ax2.set_title("IG Heatmap", fontsize=12)

        ax3.imshow(overlay)
        ax3.set_title("Overlay", fontsize=12)

        for ax in (ax0, ax1, ax2, ax3):
            ax.axis("off")

        # Row annotation on left / title on query axis
        q_text = f"q={int(q_idx[k])}" if q_idx is not None else f"row={k}"
        r_text = f"r={int(r_idx[k])}" if r_idx is not None else "r=?"
        s_text = f"sim={pair_sims[k]:.3f}" if pair_sims is not None else ""
        meta = " | ".join([t for t in [q_text, r_text, s_text] if t])

        # Put metadata above query image
        ax0.set_title(f"Query\n{meta}", fontsize=12)

        # If no ref image, indicate it
        if torch_ds is None or r_idx is None:
            ax1.text(
                0.5, 0.5, "Reference image\nnot available",
                ha="center", va="center", fontsize=10, transform=ax1.transAxes
            )

    # Shared colorbar on the far right (manual axis so it doesn't overlap/cut)
    fig.subplots_adjust(
        left=0.04,
        right=0.90,   # reserve room for cbar
        top=0.92 if title else 0.96,
        bottom=0.04,
        wspace=0.08,
        hspace=0.18,
    )

    cax = fig.add_axes([0.92, 0.14, 0.015, 0.72])  # [left, bottom, width, height]
    if hm_im is not None:
        cbar = fig.colorbar(hm_im, cax=cax)
        cbar.set_label("Normalized IG intensity", rotation=90)

    if title is None:
        title = f"Similarity IG (N shown = {n_rows})"
    fig.suptitle(title, fontsize=15, y=0.985)

    plt.show()
    return fig, axes
        

if __name__ == "__main__":
    pt_path = resolve_path("saliency_maps/ig_similarity_pair_specific__ConvNeXt-V2__N1.pt", DATA_STORE)
    plot_ig_triplets_from_pt(pt_path, indices=[0], max_rows=3)

    fo_ds, torch_ds = load_full_jaguar_from_FO_export(
        PATHS.data_export / "init",
        dataset_name="jaguar_init",
        processing_fn=None,
        overwrite_db=False,
        use_fiftyone=USE_FIFTYONE
    )

    # show first 3 rows from saved file
    plot_similarity_ig_quad_from_pt(
        pt_path=pt_path,
        torch_ds=torch_ds,
        indices=[0],   # positions in saved arrays, not necessarily dataset idx
        max_rows=3,
        title="Pair-specific similarity IG sanity check"
    )