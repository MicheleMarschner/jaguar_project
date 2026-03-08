"""
2) Are similarity matches driven by jaguar identity cues or by background shortcuts?

RQ: Does background contribute more than the animal to similarity and/or classification confidence?
Need: Your RGBA alpha-mask ablations: orig vs jaguar-only vs background-only. Run for (a) embedding similarity 
stability (sim_bg, sim_fg, margin sim_fg-sim_bg) and (b) classification head confidence (log-prob or prob of 
target class; margin bg-only − jaguar-only). Summarize distributions and % “spurious”. Include a few Grad-CAM/heatmap 
examples for high-spurious-margin cases.


RQ 1: The "Clever Hans" Check (Spurious Correlations)
The Question: Do state-of-the-art Re-ID models actually learn the jaguar's identity, or do they rely on the background environment (context)?
Hypothesis: Foundation models (like DINOv2/MegaDescriptor) are robust to background removal, whereas older or overfitted models suffer high performance drops when the background is masked.
How to Analyse:
Metric: Use your drop_bg (Score drop when background is masked) vs. drop_fg (Score drop when jaguar is masked).
Aggregation: Calculate the Spuriousness Ratio: Percentage of samples where drop_bg > drop_fg.
Comparison: Compare this ratio across your different models.
The Plot:
A Grouped Bar Chart: X-axis = Models. Y-axis = Average Probability Drop. Two bars per model: "Background Removed" vs "Jaguar Removed".
Interpretation: If the "Background Removed" bar is high, the model is cheating.

"""

import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from jaguar.XAI.run_xai_classification import select_random_datasubset_balanced
from jaguar.config import DEVICE, EXPERIMENTS_STORE, IMGNET_MEAN, IMGNET_STD, PATHS
from jaguar.datasets.JaguarDataset import MaskAwareJaguarDataset
from jaguar.models.jaguarid_models import JaguarIDModel
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export

_RUN_RE = re.compile(r"^(?P<model>.+)__(?P<split>.+)__n(?P<n>\d+)__seed(?P<seed>\d+)$")

def load_all_background_sensitivity(bg_root: Path) -> pd.DataFrame:
    """
    Loads all similarity_stability__*.parquet/csv and adds model/run metadata.
    Expected columns inside each file (minimum):
      filepath, stability_jaguar_only, stability_bg_only, spurious_margin, is_spurious
    """
    dfs = []

    # parquet preferred, but allow csv
    files = list(bg_root.rglob("similarity_stability__*.parquet")) + list(bg_root.rglob("similarity_stability__*.csv"))

    for fp in files:
        run_dir = fp.parent  # adjust if nested
        m = _RUN_RE.match(run_dir.name)
        if m is None and run_dir.parent is not None:
            m = _RUN_RE.match(run_dir.parent.name)
            if m is not None:
                run_dir = run_dir.parent

        if m is None:
            continue

        df = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)

        df["model"] = m.group("model")
        df["split"] = m.group("split")
        df["n_samples"] = int(m.group("n"))
        df["seed"] = int(m.group("seed"))
        df["run_id"] = run_dir.name
        df["source_file"] = str(fp)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def summarize_spuriousness_generic(
    df: pd.DataFrame,
    value_bg_removed_col: str,   # “jaguar only” (background removed)
    value_fg_removed_col: str,   # “bg only” (jaguar removed)
    model_col: str = "model",
    id_col: str = "filepath",
    margin_col_name: str = "spurious_margin",
) -> pd.DataFrame:
    """
    RQ: Background shortcuts / Clever Hans check (Spurious correlations).

    Goal:
    Quantify whether a model relies more on background context than on the jaguar itself.

    Method (alpha-mask ablations):
    You provide two scalar scores per image:
        - value_bg_removed_col: score when background is removed (jaguar-only)
        - value_fg_removed_col: score when jaguar is removed (background-only)

    We compute:
    margin = (background-only score) - (jaguar-only score)
    is_spurious = margin > 0   (background contributes more than jaguar)

    Outputs:
    1) Per-model summary table with:
        - mean_bg_removed, mean_fg_removed
        - mean_margin
        - spurious_ratio (% samples where margin > 0)
    2) A long dataframe with computed margin + is_spurious per sample.

    Interpretation:
    - High spurious_ratio or positive mean_margin suggests the model is "cheating" using background cues.
    - Low spurious_ratio and negative mean_margin suggests identity cues dominate.
    """
    d = df.copy()

    d["val_bg_removed"] = d[value_bg_removed_col]
    d["val_fg_removed"] = d[value_fg_removed_col]
    d[margin_col_name] = d["val_fg_removed"] - d["val_bg_removed"]
    d["is_spurious"] = d[margin_col_name] > 0

    summary = (
        d.groupby(model_col)
         .agg(
            n=(id_col, "count"),
            mean_bg_removed=("val_bg_removed", "mean"),
            mean_fg_removed=("val_fg_removed", "mean"),
            mean_margin=(margin_col_name, "mean"),
            spurious_ratio=("is_spurious", "mean"),
         )
         .reset_index()
         .sort_values("spurious_ratio", ascending=False)
    )
    return summary, d


def plot_margin_box(df_long: pd.DataFrame, out_path: Path, margin_col: str = "spurious_margin", title: str = ""):
    """
    RQ: Background shortcuts / Clever Hans check (Spurious correlations).

    Goal:
    Visualize the distribution of spuriousness margins per model.

    What it plots:
    Boxplot of 'spurious_margin' (or a provided margin column) by model, with a zero baseline line.
    Positive margins => background-only produces higher score than jaguar-only (more spurious).
    Negative margins => jaguar-only dominates (desired).

    Use:
    This is the main distribution plot supporting claims like
    "Model A shows stronger background reliance than Model B."
    """
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_long, x="model", y=margin_col)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(title or f"{margin_col} by model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_mean_bars(summary: pd.DataFrame, out_path: Path, title: str = ""):
    """
    RQ: Background shortcuts / Clever Hans check (Spurious correlations).

    Goal:
    Provide an easy-to-interpret summary view (two bars per model).

    What it plots:
    Grouped bar chart showing mean score under:
        - Jaguar only (background removed)
        - Background only (jaguar removed)

    Interpretation:
    - If background-only mean is close to (or higher than) jaguar-only mean, the model is likely using context cues.
    - If jaguar-only mean is much higher, the model is robust to background removal.

    Note:
    This matches the "Clever Hans" narrative: does removing background break the model?
    """
    plot_df = summary.melt(
        id_vars=["model"],
        value_vars=["mean_bg_removed","mean_fg_removed"],
        var_name="condition",
        value_name="mean_value",
    )
    plot_df["condition"] = plot_df["condition"].map({
        "mean_bg_removed": "Jaguar only (BG removed)",
        "mean_fg_removed": "Background only (Jaguar removed)",
    })

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="model", y="mean_value", hue="condition")
    plt.title(title or "Mean values under masking")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_spurious_ratio(summary: pd.DataFrame, out_path: Path, title: str = ""):
    """
    RQ: Background shortcuts / Clever Hans check (Spurious correlations).

    Goal:
    Summarize the shortcut behavior as a single interpretable number per model.

    What it plots:
    Bar chart of spurious_ratio per model:
        spurious_ratio = % samples where (background-only score) > (jaguar-only score)

    Interpretation:
    Higher spurious_ratio => more evidence the model relies on background.
    Lower spurious_ratio => more evidence the model relies on jaguar identity cues.
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(data=summary, x="model", y="spurious_ratio")
    plt.ylim(0, 1)
    plt.title(title or "Spuriousness ratio")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()







_RUN_RE = re.compile(r"^(?P<model>.+)__(?P<split>.+)__n(?P<n>\d+)__seed(?P<seed>\d+)$")

def load_all_classification_sensitivity(bg_root: Path) -> pd.DataFrame:
    """
    Loads all classification_sensitivity__*.parquet/csv and adds model/run metadata.
    Expected columns (minimum):
      filepath, score_jaguar_only_logp, score_bg_only_logp (or *_p)
    """
    dfs = []
    files = list(bg_root.rglob("classification_sensitivity__*.parquet")) + list(bg_root.rglob("classification_sensitivity__*.csv"))

    for fp in files:
        run_dir = fp.parent
        m = _RUN_RE.match(run_dir.name)
        if m is None and run_dir.parent is not None:
            m = _RUN_RE.match(run_dir.parent.name)
            if m is not None:
                run_dir = run_dir.parent
        if m is None:
            continue

        df = pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp)

        df["model"] = m.group("model")
        df["split"] = m.group("split")
        df["n_samples"] = int(m.group("n"))
        df["seed"] = int(m.group("seed"))
        df["run_id"] = run_dir.name
        df["source_file"] = str(fp)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

from jaguar.utils.utils import ensure_dir, resolve_path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def unnorm_img(t: torch.Tensor, mean, std) -> np.ndarray:
    """t: [3,H,W] normalized -> returns float RGB [H,W,3] in [0,1]."""
    x = t.detach().cpu().clone()
    for c, (m, s) in enumerate(zip(mean, std)):
        x[c] = x[c] * s + m
    x = x.permute(1,2,0).numpy()
    return np.clip(x, 0, 1)

def save_bg_panels_with_gradcam(
    model,
    dataloader,
    out_dir: Path,
    target_layer,
    mean, std,
    n: int = 10,
    pick: str = "most_spurious",  # "most_spurious" or "random"
):
    """
    Saves panels: Orig, Jaguar-only, BG-only + GradCAM overlays (same target class).
    Requires batch contains: t_orig, t_bg_masked, t_fg_masked, label_idx, filepath.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])

    saved = 0
    for batch in dataloader:
        bs = len(batch["filepath"])
        for i in range(bs):
            if saved >= n:
                return

            x0 = batch["t_orig"][i].unsqueeze(0).to(next(model.parameters()).device)
            xJ = batch["t_bg_masked"][i].unsqueeze(0).to(next(model.parameters()).device)
            xB = batch["t_fg_masked"][i].unsqueeze(0).to(next(model.parameters()).device)

            cls = int(batch["label_idx"][i].item())
            targets = [ClassifierOutputTarget(cls)]

            # GradCAM maps
            cam0 = cam(x0, targets=targets)[0]
            camJ = cam(xJ, targets=targets)[0]
            camB = cam(xB, targets=targets)[0]

            # Unnorm for display
            img0 = unnorm_img(batch["t_orig"][i], mean, std)
            imgJ = unnorm_img(batch["t_bg_masked"][i], mean, std)
            imgB = unnorm_img(batch["t_fg_masked"][i], mean, std)

            # simple overlay
            def overlay(img, heat, a=0.45):
                import cv2
                h = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                h = cv2.resize(h, (img.shape[1], img.shape[0]))
                hm = cv2.applyColorMap(np.uint8(255*h), cv2.COLORMAP_JET)
                hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB) / 255.0
                return np.clip((1-a)*img + a*hm, 0, 1)

            vis0 = overlay(img0, cam0)
            visJ = overlay(imgJ, camJ)
            visB = overlay(imgB, camB)

            fname = Path(batch["filepath"][i]).stem

            fig, ax = plt.subplots(2, 3, figsize=(12, 7))
            ax[0,0].imshow(img0); ax[0,0].set_title("Orig"); ax[0,0].axis("off")
            ax[0,1].imshow(imgJ); ax[0,1].set_title("Jaguar-only"); ax[0,1].axis("off")
            ax[0,2].imshow(imgB); ax[0,2].set_title("BG-only"); ax[0,2].axis("off")

            ax[1,0].imshow(vis0); ax[1,0].set_title("GradCAM Orig"); ax[1,0].axis("off")
            ax[1,1].imshow(visJ); ax[1,1].set_title("GradCAM Jaguar-only"); ax[1,1].axis("off")
            ax[1,2].imshow(visB); ax[1,2].set_title("GradCAM BG-only"); ax[1,2].axis("off")

            plt.tight_layout()
            plt.savefig(out_dir / f"bg_panel_gradcam__{fname}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            saved += 1

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def save_bg_panels_no_cam(dataloader, out_dir: Path, mean, std, n: int = 10):
    """
    Saves panels: Orig, Jaguar-only, BG-only (no CAM).
    Useful for showing what the alpha-mask ablation is doing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def unnorm(t):
        x = t.detach().cpu().clone()
        for c, (m, s) in enumerate(zip(mean, std)):
            x[c] = x[c] * s + m
        return np.clip(x.permute(1,2,0).numpy(), 0, 1)

    saved = 0
    for batch in dataloader:
        bs = len(batch["filepath"])
        for i in range(bs):
            if saved >= n:
                return

            img0 = unnorm(batch["t_orig"][i])
            imgJ = unnorm(batch["t_bg_masked"][i])
            imgB = unnorm(batch["t_fg_masked"][i])

            fname = Path(batch["filepath"][i]).stem

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(img0); ax[0].set_title("Orig"); ax[0].axis("off")
            ax[1].imshow(imgJ); ax[1].set_title("Jaguar-only"); ax[1].axis("off")
            ax[2].imshow(imgB); ax[2].set_title("BG-only"); ax[2].axis("off")

            plt.tight_layout()
            plt.savefig(out_dir / f"bg_panel__{fname}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            saved += 1

import pandas as pd

def select_topk_filepaths(df: pd.DataFrame, k: int = 10, mode: str = "most_spurious") -> set[str]:
    """
    mode:
      - "most_spurious": largest spurious_margin
      - "least_spurious": smallest spurious_margin
      - "spurious_only": top-k among is_spurious==True
    Returns a set of filenames (Path(...).name) to filter the dataloader loop.
    """
    d = df.copy()
    d = d.dropna(subset=["filepath", "spurious_margin"])
    d["filepath"] = d["filepath"].astype(str)

    if mode == "spurious_only":
        d = d[d["is_spurious"] == True]

    if d.empty:
        return set()

    asc = (mode == "least_spurious")
    top = d.sort_values("spurious_margin", ascending=asc).head(k)
    return set(top["filepath"].tolist())


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def save_mask_triptychs(
    dataloader,
    out_dir: Path,
    mean, std,
    target_files: set[str],
    n_max: int = 10,
):
    """
    Saves panels: [Orig | Jaguar-only (BG removed) | BG-only (Jaguar removed)]
    for filepaths in target_files.
    Assumes batch contains: t_orig, t_bg_masked, t_fg_masked, filepath
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = np.array(mean).reshape(1,1,3)
    std  = np.array(std).reshape(1,1,3)

    def unnorm(t):
        x = t.detach().cpu().permute(1,2,0).numpy()
        x = x * std + mean
        return np.clip(x, 0, 1)

    saved = 0
    for batch in dataloader:
        for i in range(len(batch["filepath"])):
            fname = Path(batch["filepath"][i]).name
            if fname not in target_files:
                continue

            img0 = unnorm(batch["t_orig"][i])
            imgJ = unnorm(batch["t_bg_masked"][i])
            imgB = unnorm(batch["t_fg_masked"][i])

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(img0); ax[0].set_title("Orig"); ax[0].axis("off")
            ax[1].imshow(imgJ); ax[1].set_title("Jaguar-only\n(BG removed)"); ax[1].axis("off")
            ax[2].imshow(imgB); ax[2].set_title("BG-only\n(Jaguar removed)"); ax[2].axis("off")

            plt.tight_layout()
            plt.savefig(out_dir / f"mask_triptych__{Path(fname).stem}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            saved += 1
            if saved >= n_max:
                return
            

import seaborn as sns
import matplotlib.pyplot as plt

def plot_drop_bars(cls_all: pd.DataFrame, out_path: Path, title: str = "Average score drop under masking"):
    """
    Plots mean drop_bg vs drop_fg per model (two bars per model).
    Uses cls_all columns: model, drop_bg, drop_fg
    """
    df = cls_all.dropna(subset=["model", "drop_bg", "drop_fg"]).copy()

    summary = (
        df.groupby("model")
          .agg(mean_drop_bg=("drop_bg", "mean"), mean_drop_fg=("drop_fg", "mean"))
          .reset_index()
    )

    plot_df = summary.melt(
        id_vars=["model"],
        value_vars=["mean_drop_bg", "mean_drop_fg"],
        var_name="condition",
        value_name="mean_drop",
    )
    plot_df["condition"] = plot_df["condition"].map({
        "mean_drop_bg": "Background removed (Jaguar-only)",
        "mean_drop_fg": "Jaguar removed (BG-only)",
    })

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="model", y="mean_drop", hue="condition")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()




if __name__ == "__main__":
    bg_root = resolve_path("xai/background_sensitivity", EXPERIMENTS_STORE)
    save_dir = PATHS.results / "xai" / "background_sensitivity"
    ensure_dir(save_dir)

    bg_all = load_all_background_sensitivity(bg_root)
    if bg_all.empty:
        raise RuntimeError(f"No background sensitivity files found under {bg_root}")

    # ----- Embedding analysis -----
    emb_summary, emb_long = summarize_spuriousness_generic(
        bg_all,
        value_bg_removed_col="stability_jaguar_only",
        value_fg_removed_col="stability_bg_only",
        margin_col_name="spurious_margin_emb",
    )
    emb_summary.to_csv(save_dir / "embedding_spuriousness_summary.csv", index=False)

    cls_summary, cls_long = summarize_spuriousness_generic(
        cls_all,
        value_bg_removed_col="score_jaguar_only_logp",
        value_fg_removed_col="score_bg_only_logp",
        margin_col_name="spurious_margin_logp",
    )
    cls_summary.to_csv(save_dir / "cls_spuriousness_summary_logp.csv", index=False)
    cls_long.to_csv(save_dir / "cls_spuriousness_long_logp.csv", index=False)

    plot_margin_box(
        emb_long,
        save_dir / "embedding_spurious_margin_box.png",
        margin_col="spurious_margin",
        title="Embedding spurious margin (BG-only − Jaguar-only)"
    )
    plot_mean_bars(
        emb_summary,
        save_dir / "embedding_mean_stability_bars.png",
        title="Mean embedding stability under masking"
    )
    plot_spurious_ratio(
        emb_summary,
        save_dir / "embedding_spurious_ratio.png",
        title="Embedding spuriousness ratio"
    )


    # load
    cls_all = load_all_classification_sensitivity(bg_root)
    if cls_all.empty:
        raise RuntimeError("No classification_sensitivity__* files found. Did you run the cls pipeline?")

    # summarize using your generic function
    cls_summary, cls_long = summarize_spuriousness_generic(
        cls_all,
        value_bg_removed_col="score_jaguar_only_logp",
        value_fg_removed_col="score_bg_only_logp",
        margin_col_name="spurious_margin_logp",
    )

    cls_summary.to_csv(save_dir / "cls_spuriousness_summary_logp.csv", index=False)

    # plots
    plot_margin_box(
        cls_long,
        save_dir / "cls_spurious_margin_logp_box.png",
        margin_col="spurious_margin_logp",
        title="Classification spurious margin (logp): BG-only − Jaguar-only"
    )
    plot_mean_bars(
        cls_summary,
        save_dir / "cls_mean_logp_bars.png",
        title="Mean log-prob under masking (classification)"
    )
    plot_spurious_ratio(
        cls_summary,
        save_dir / "cls_spurious_ratio_logp.png",
        title="Spuriousness ratio (classification logp)"
    )

    # usage:
    plot_drop_bars(
        cls_all, save_dir / "cls_mean_drop_bars.png", 
        title="Average classification score drop under masking"
    )




    # ----- Qualitative: pick most/least spurious and save mask triptychs -----
    # Use emb_long (it has filepath + spurious_margin + is_spurious)
    most_files = select_topk_filepaths(emb_long, k=10, mode="most_spurious")
    least_files = select_topk_filepaths(emb_long, k=10, mode="least_spurious")

    # TODO: you must create your dataloader here (MaskAwareJaguarDataset + DataLoader)
    BATCH_SIZE = 16
    backbone_name = "MegaDescriptor-L"
    head_type = "arcface"
    n_samples = 10
    dataset_name = "jaguar_init"
    manifest_dir = PATHS.data_export / "init"
    checkpoint_path = ""
    
    # Load Sample List
    _, temp_ds = load_jaguar_from_FO_export(manifest_dir, dataset_name=dataset_name)
    ### load subsamples
    samples_list = [temp_ds.samples[i] for i in torch_idx]

    num_classes = int(len(np.unique(np.asarray(temp_ds.labels))))

    # Load Model (Pre-trained/Fine-tuned)
    model = JaguarIDModel(
        backbone_name=backbone_name, 
        num_classes=num_classes,
        head_type=head_type,
        device=str(DEVICE)
    )
    # model.load_state_dict(torch.load("path/to/weights.pt"))       ## TODO!
    model = model.to(DEVICE).eval()

    # Create Final Dataset
    xai_ds = MaskAwareJaguarDataset(
        jaguar_model=model,
        base_root=PATHS.data_train,
        samples_list=samples_list,
        is_test=False
    )
    loader = torch.utils.data.DataLoader(xai_ds, batch_size=BATCH_SIZE, shuffle=False)
    

    # save_mask_triptychs(loader, save_dir / "qual_masks/most_spurious", IMGNET_MEAN, IMGNET_STD, most_files, n_max=10)
    # save_mask_triptychs(loader, save_dir / "qual_masks/least_spurious", IMGNET_MEAN, IMGNET_STD, least_files, n_max=10)

    print(f"Done. Saved to {save_dir}")



    cls_summary, cls_long = summarize_spuriousness_generic(
        cls_all,
        value_bg_removed_col="score_jaguar_only_logp",
        value_fg_removed_col="score_bg_only_logp",
    )

    
    plot_mean_bars(emb_summary, save_dir / "bg_mean_sims_bars.png")
    plot_spurious_ratio(emb_summary, save_dir / "bg_spurious_ratio.png")
    plot_mean_bars(cls_summary, save_dir / "bg_mean_sims_bars.png")
    plot_spurious_ratio(cls_summary, save_dir / "bg_spurious_ratio.png")

    target_files = select_topk_filepaths(df_embedding, k=10, mode="most_spurious")
    save_mask_triptychs(loader, save_dir / "qual_masks/embedding_most_spurious", IMGNET_MEAN, IMGNET_STD, target_files)

    target_files = select_topk_filepaths(df_logits, k=10, mode="most_spurious")
    save_mask_triptychs(loader, save_dir / "qual_masks/classification_most_spurious", IMGNET_MEAN, IMGNET_STD, target_files)

    target_files = select_topk_filepaths(df_embedding, k=10, mode="least_spurious")
    save_mask_triptychs(loader, save_dir / "qual_masks/embedding_least_spurious", IMGNET_MEAN, IMGNET_STD, target_files)
    
    top_files = select_topk_filepaths(cls_long.rename(columns={"spurious_margin_logp":"spurious_margin"}), k=10, mode="most_spurious")
    save_mask_triptychs(loader, save_dir / "qual_masks/cls_most_spurious", IMGNET_MEAN, IMGNET_STD, top_files, n_max=10)