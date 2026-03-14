from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from jaguar.config import PATHS
from jaguar.utils.utils import read_json_if_exists


def plot_single_run_loss_and_map(
    run_dir: str | Path,
    save_dir: str | Path,
    model_label: str | None = None,
) -> dict[str, Path]:
    """
    Build train-loss and val-mAP plots for a single run from train_history.json.
    """
    run_dir = Path(run_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = pd.DataFrame(read_json_if_exists(run_dir / "train_history.json"))
    if "epoch" in history.columns:
        history = history.sort_values("epoch").reset_index(drop=True)

    label = model_label or run_dir.name

    outputs: dict[str, Path] = {}

    sns.set_theme(style="whitegrid")

    if "train_loss" in history.columns:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=history, x="epoch", y="train_loss", marker="o")
        plt.title(f"Train Loss — {label}")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.tight_layout()

        loss_path = save_dir / f"{label}_train_loss.png"
        plt.savefig(loss_path, dpi=200, bbox_inches="tight")
        plt.close()
        outputs["train_loss_plot"] = loss_path

    if "val_mAP" in history.columns:
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=history, x="epoch", y="val_mAP", marker="o")
        plt.title(f"Validation mAP — {label}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation mAP")
        plt.tight_layout()

        map_path = save_dir / f"{label}_val_mAP.png"
        plt.savefig(map_path, dpi=200, bbox_inches="tight")
        plt.close()
        outputs["val_map_plot"] = map_path

    return outputs


if __name__ == "__main__":
    run_dir = PATHS.runs / ""
    save_dir = PATHS.results
    model_label = "EVA-02"

    out = plot_single_run_loss_and_map(
        run_dir=run_dir,
        save_dir=save_dir,
        model_label="megadescriptor_full",
    )

