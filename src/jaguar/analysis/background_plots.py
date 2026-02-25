from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from jaguar.config import PATHS
from jaguar.preprocessing.preprocessing_background import PROCESSORS
from jaguar.utils.utils import ensure_dir
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export


def show_jaguar_images_with_backgrounds(
    jaguar_id,
    dataset,  
    processors: dict,
    processor_names,
    n_max: int = 4,
    processor_kwargs: dict | None = None,
    save_path: Path = None
):
    """
    rows = images of one jaguar
    cols = different background processors
    """

    processor_kwargs = processor_kwargs or {}

    # collect filenames / samples for this jaguar
    selected_samples = []
    for s in dataset.samples:
        label = str(s["ground_truth"]["label"])
        if label == str(jaguar_id):
            s2 = dict(s)
            s2.setdefault("filename", s2.get(dataset.filename_key) or Path(s2[dataset.filepath_key]).name)
            selected_samples.append(s2)
            if len(selected_samples) >= n_max:
                break

    if len(selected_samples) == 0:
        print(f"No images found for jaguar_id={jaguar_id}")
        return

    n_rows = len(selected_samples)
    n_cols = len(processor_names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))

    # normalize axes shape
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for r, s in enumerate(selected_samples):
        img_path = dataset._resolve_path(s[dataset.filepath_key])

        with Image.open(img_path) as img:
            img = img.convert("RGBA")

            for c, pname in enumerate(processor_names):
                ax = axes[r][c]

                sample = dict(s)
                sample["_epoch"] = getattr(dataset, "epoch", 0)
                sample["filepath"] = str(img_path)

                if pname == "raw_original":
                    out_img = img.copy()
                else:
                    proc = processors[pname]
                    out_img = proc(
                        img.copy(),
                        sample=sample,
                        base_root=dataset.base_root,
                        **processor_kwargs,
                    )

                ax.imshow(out_img)
                ax.axis("off")

                if r == 0:
                    ax.set_title(pname, fontsize=10)
                if c == 0:
                    ax.set_ylabel(sample["filename"], fontsize=9)

    fig.suptitle(f"Jaguar: {jaguar_id} | background variants", fontsize=14)
    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.4)
    
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    save_dir = PATHS.results / "backgrounds"
    ensure_dir(save_dir)
    jaguar_id = "Marcela"

    _, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export / "init",
        dataset_name="jaguar_init",
        processing_fn=None,
        overwrite_db=False,
    )

    show_jaguar_images_with_backgrounds(
        jaguar_id=jaguar_id,
        dataset=torch_ds,
        processors=PROCESSORS,
        processor_names=["raw_original", "gray_bg", "blur_bg", "random_bg"],
        n_max=4,
        processor_kwargs={"bg_dir": str(PATHS.data / "backgrounds")},
        save_path=save_dir / f"{jaguar_id}_background_variants.png"
    )