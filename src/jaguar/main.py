import argparse
from pathlib import Path
import yaml
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import torch
from functools import partial

from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.config import PROJECT_ROOT, PATHS, SEED, DEVICE
from jaguar.utils.utils import set_seeds, ensure_dirs, init_wandb
from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.preprocessing import PROCESSORS


def load_experiment_template(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def save_npz(out_path: Path, **arrays):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print("Saved:", out_path)


def overlay_cam_on_image(img_pil: Image.Image, cam_2d: np.ndarray, out_path: Path):
    """
    img_pil: original RGB PIL
    cam_2d:  HxW float array
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam = cam_2d.astype(np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_pil)
    ax.imshow(cam, alpha=0.45)  # heat overlay
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def show_grid_from_torch_ds(torch_ds, n=16, cols=4, bg=(128,128,128)):
    n = min(n, len(torch_ds))
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(n):
        it = torch_ds[i]
        p = Path(it["filepath"])

        rgba = Image.open(p).convert("RGBA")
        bg_rgba = Image.new("RGBA", rgba.size, (*bg, 255))
        img = Image.alpha_composite(bg_rgba, rgba).convert("RGB")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f'{it.get("id","")}\n{p.name}', fontsize=8)

    plt.tight_layout()
    plt.show()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train"])
    parser.add_argument("template_path", nargs="?", type=Path, help="Meaning depends on action")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.action == "train": 
        set_seeds(SEED) 
        ensure_dirs()

        #exp_config = load_experiment_template(args.template_path)

        # intialize wandb if on colab
        #run = init_wandb(exp_config)

        processor_name = "cutout_gray_bg"  # e.g. from YAML
        model_name = "EfficientNet-B4"
        out_dir = PATHS.data / "embeddings"

        processing_fn = partial(PROCESSORS[processor_name], base_root=PATHS.data_export)
        wrapper = FoundationModelWrapper(model_name, device=str(DEVICE))

        # get data from fifty One for that experiment
        fo_ds, torch_ds = load_jaguar_from_FO_export(
            PATHS.data_export,
            dataset_name="jaguar_stage0",
            processing_fn=processing_fn,
            overwrite_db=False,
        )

        p = torch_ds[0]["filepath"]
        im = Image.open(p)
        n=8

        # ---- 1) take N samples ----
        items = [torch_ds[i] for i in range(min(n, len(torch_ds)))]
        pil_images = []
        ids = []
        filenames = []
        for it in items:
            # Your dataset returns transformed tensor usually; for CAM we want PIL.
            # If your torch_ds currently returns tensors, adapt it to also return PIL or reload by filepath.
            # Here we reload by filepath:
            img_path = it["filepath"]
            img = Image.open(img_path).convert("RGB")
            pil_images.append(img)
            ids.append(it.get("id", ""))
            filenames.append(it.get("filename", Path(img_path).name))

        show_grid_from_torch_ds(torch_ds, n=16, cols=4)

        '''
        
        # ---- 2) extract embeddings ----
        emb = wrapper.extract_embeddings(pil_images)  # np (N, D)
        print("Embeddings shape:", emb.shape)

        wrapper.save_embeddings(emb)

        # ---- 4) GradCAM on 1 image (if configured) ----
        try:
            target_layers, reshape_transform = wrapper.get_grad_cam_config()
        except NotImplementedError:
            print(f"[GradCAM] Not configured for {model_name}, skipping.")
            return

        cam = GradCAM(
            model=wrapper.model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        )

        # preprocess 1 image to tensor [1,3,H,W]
        x = wrapper.preprocess(pil_images[3]).unsqueeze(0).to(DEVICE)

        # For ReID/embeddings: GradCAM expects a target.
        # If your model returns logits, you can use argmax.
        # If it returns embeddings, simplest smoke target = L2 norm (scalar).
        with torch.no_grad():
            out = wrapper.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("logits", None) or next(iter(out.values()))

        if out.ndim == 2 and out.shape[1] > 1:
            target_idx = int(out.argmax(dim=1).item())
            targets = [ClassifierOutputTarget(target_idx)]
            grayscale_cam = cam(input_tensor=x, targets=targets)  # shape (B,H,W) depending on lib

        # normalize and overlay
        cam_2d = grayscale_cam[0]
        overlay_cam_on_image(pil_images[3], cam_2d, out_dir / f"gradcam_{model_name}_sample0.png")

        '''

            

if __name__ == "__main__":
    main()