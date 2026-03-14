from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from jaguar.config import DATA_STORE, USE_FIFTYONE
from jaguar.utils.utils import ensure_dir, resolve_path
from jaguar.utils.utils_datasets import load_full_jaguar_from_FO_export
from jaguar.preprocessing.preprocessing_background import PROCESSORS


def show_jaguar_images_with_backgrounds(
    jaguar_id: str,
    dataset,
    processor_names: list[str],
    n_max: int = 4,
    processor_kwargs: dict | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Plot background variants for up to n_max images of one jaguar identity.
    Rows correspond to images, columns to background processors.
    """
    processor_kwargs = processor_kwargs or {}

    selected_samples = []
    for sample in dataset.samples:
        label = str(sample["ground_truth"]["label"])
        if label != str(jaguar_id):
            continue

        sample_copy = dict(sample)
        sample_copy.setdefault(
            "filename",
            sample_copy.get(dataset.filename_key)
            or Path(sample_copy[dataset.filepath_key]).name,
        )
        selected_samples.append(sample_copy)

        if len(selected_samples) >= n_max:
            break

    if not selected_samples:
        print(f"No images found for jaguar_id={jaguar_id}")
        return

    n_rows = len(selected_samples)
    n_cols = len(processor_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for r, sample in enumerate(selected_samples):
        img_path = dataset._resolve_path(sample[dataset.filepath_key])

        with Image.open(img_path) as img:
            img = img.convert("RGBA")

            for c, processor_name in enumerate(processor_names):
                ax = axes[r][c]

                processor_sample = dict(sample)
                processor_sample["_epoch"] = getattr(dataset, "epoch", 0)
                processor_sample["filepath"] = str(img_path)

                if processor_name == "raw_original":
                    out_img = img.copy()
                else:
                    processor = PROCESSORS[processor_name]
                    out_img = processor(
                        img.copy(),
                        sample=processor_sample,
                        base_root=dataset.base_root,
                        **processor_kwargs,
                    )

                ax.imshow(out_img)
                ax.axis("off")

                if r == 0:
                    ax.set_title(processor_name, fontsize=10)
                if c == 0:
                    ax.set_ylabel(processor_sample["filename"], fontsize=9)

    fig.suptitle(f"Jaguar: {jaguar_id} | background variants", fontsize=14)
    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.4)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close(fig)


def create_background_plots(
    config: dict,
    run_dir: Path | None = None,
    root_dir: Path | None = None,
    exemplar_run_dir: Path | None = None,
    save_dir: Path | None = None,
) -> None:
    """
    Save qualitative background-variant visualizations for selected jaguar identities.
    """
    if save_dir is None:
        raise ValueError("save_dir must be provided")

    ensure_dir(save_dir)

    analysis_cfg = config.get("analysis", {})
    jaguar_ids = analysis_cfg.get("jaguar_ids", ["Marcela"])
    processor_names = analysis_cfg.get(
        "processor_names",
        ["raw_original", "gray_bg", "blur_bg", "random_bg"],
    )
    n_max = analysis_cfg.get("n_max", 4)

    _, torch_ds = load_full_jaguar_from_FO_export(
        resolve_path("fiftyone/init", DATA_STORE),
        dataset_name="jaguar_init",
        processing_fn=None,
        overwrite_db=False,
        use_fiftyone=USE_FIFTYONE,
    )

    processor_kwargs = {
        "bg_dir": str(resolve_path("backgrounds", DATA_STORE)),
    }

    qual_dir = save_dir / "qualitative_backgrounds"
    ensure_dir(qual_dir)

    for jaguar_id in jaguar_ids:
        out_path = qual_dir / f"{jaguar_id}_background_variants.png"
        show_jaguar_images_with_backgrounds(
            jaguar_id=jaguar_id,
            dataset=torch_ds,
            processor_names=processor_names,
            n_max=n_max,
            processor_kwargs=processor_kwargs,
            save_path=out_path,
        )