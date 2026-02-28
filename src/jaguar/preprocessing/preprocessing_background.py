"""
Background compositing and background-patch generation utilities for Jaguar re-ID preprocessing.

Project role:
- converts RGBA cutouts into 3-channel RGB images using controlled background policies
- supports background ablations (solid, random habitat, blurred)
- provides deterministic random-background assignment per (epoch, sample)
- builds a reusable habitat background patch pool from raw images + cutout alpha masks

Used for preprocessing/augmentation experiments, not for model training logic itself.
"""

import random
from pathlib import Path
from PIL import Image, ImageFilter
from jaguar.utils.utils import resolve_path
import numpy as np

from jaguar.config import DATA_STORE, PATHS


def rgba_on_solid_bg(rgba: Image.Image, color=(128, 128, 128)) -> Image.Image:
    """Composite an RGBA cutout onto a solid RGB background."""
    rgba = rgba.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (*color, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")

def rgba_on_random_bg(rgba: Image.Image, bg_path: Path) -> Image.Image:
    """Composite an RGBA cutout onto a chosen background image (resized to match)."""
    rgba = rgba.convert("RGBA")
    bg = Image.open(bg_path).convert("RGB").resize(rgba.size).convert("RGBA")
    return Image.alpha_composite(bg, rgba).convert("RGB")

# Shared helper for solid-background ablations (black/white/gray) with optional edge feathering.
def _alpha_composite_on_color(rgba: Image.Image, rgb_color=(0, 0, 0), edge_softness: int = 0) -> Image.Image:
    rgba = rgba.convert("RGBA")

    # Optional: feather alpha edges to reduce hard cutout boundaries in ablation images.
    if edge_softness > 0:
        a = rgba.getchannel("A").filter(ImageFilter.GaussianBlur(radius=edge_softness))
        rgba = rgba.copy()
        rgba.putalpha(a)

    bg = Image.new("RGBA", rgba.size, (*rgb_color, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


# Collection of preprocessing variants used to test background sensitivity in re-ID models.
class ImageProcessor:
    @staticmethod
    def original(img: Image.Image, sample: dict, base_root: Path, **kwargs) -> Image.Image:
        # "Original" in the experiment sense = always return RGB.
        return img.convert("RGB")

    @staticmethod
    def black_bg_cutout_alpha(img: Image.Image, sample: dict, base_root, edge_softness: int = 0, **kwargs) -> Image.Image:
        return _alpha_composite_on_color(img, rgb_color=(0, 0, 0), edge_softness=edge_softness)

    @staticmethod
    def white_bg_cutout_alpha(img: Image.Image, sample: dict, base_root, edge_softness: int = 0, **kwargs) -> Image.Image:
        return _alpha_composite_on_color(img, rgb_color=(255, 255, 255), edge_softness=edge_softness)

    @staticmethod
    def gray_bg_cutout_alpha(img: Image.Image, sample: dict, base_root, edge_softness: int = 0, **kwargs) -> Image.Image:
        return _alpha_composite_on_color(img, rgb_color=(128, 128, 128), edge_softness=edge_softness)
    
    @staticmethod
    def random_bg_cutout_deterministic(
        img: Image.Image,
        sample: dict,
        base_root: Path,
        bg_dir: str,
        key_for_seed: str = "filename",   # or "filepath"
        **kwargs,
    ) -> Image.Image:
        """
        Composite cutout onto a random habitat background, but deterministically per (epoch, sample).

        This makes augmentation reproducible while still allowing background changes across epochs.
        """
        bg_dir = Path(bg_dir)
        bg_files = [p for p in bg_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if not bg_files:
            raise ValueError(f"No backgrounds in {bg_dir}")

        # Deterministic background assignment keeps augmentation reproducible:
        # same sample in same epoch -> same background, different epoch -> can change.
        epoch = int(sample.get("_epoch", 0))
        seed_key = str(sample.get(key_for_seed) or sample.get("filepath") or "")

        # Tie randomness to (epoch, sample key) so runs are reproducible but still epoch-varying.
        seed = hash((epoch, seed_key)) & 0xFFFFFFFF

        rng = random.Random(seed)
        bg_path = rng.choice(bg_files)

        return rgba_on_random_bg(img, bg_path)

    @staticmethod
    def blur_bg_cutout_alpha(
        img: Image.Image,
        sample: dict,
        base_root: Path,
        blur_radius: int = 10,
        edge_softness: int = 2,
        bg_color=(128, 128, 128),
        **kwargs,
    ) -> Image.Image:
        """
        Keeps foreground appearance unchanged while suppressing background detail, useful to test how much 
        the model relies on contextual cues.

        Uses alpha as mask:
        - foreground stays sharp
        - background becomes blurred (or a solid bg) and edges can be feathered
        """
        rgba = img.convert("RGBA")
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")  # L mask

        # Feather edges to soften cutout transitions.
        if edge_softness > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=edge_softness))

        # Blur background while keeping the jaguar foreground unchanged.
        blurred_bg = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        #  Composite: show original foreground where alpha is present, blurred background elsewhere.
        out = Image.composite(rgb, blurred_bg, alpha)
        return out

# Registry used by dataset loading/preprocessing code to select a background policy by name.
PROCESSORS = {
    "original": ImageProcessor.original,
    "gray_bg": ImageProcessor.gray_bg_cutout_alpha,
    "random_bg": ImageProcessor.random_bg_cutout_deterministic,
    "black_bg": ImageProcessor.black_bg_cutout_alpha,
    "white_bg": ImageProcessor.white_bg_cutout_alpha,
    "blur_bg": ImageProcessor.blur_bg_cutout_alpha,
}

def build_habitat_backgrounds(
    raw_dir: Path,
    cutout_dir: Path,
    out_dir: Path,
    n_patches: int = 100,
    patch_size: int = 224,
    max_tries_per_patch: int = 30,
    max_fg_frac: float = 0.02,   # allow <=2% foreground pixels in patch
    seed: int = 51,
):
    """
    Create a reusable pool of mostly-foreground-free habitat patches from the raw training images.

    These patches are later used as realistic random backgrounds for cutout compositing
    (background-ablation experiments).
    """
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    if not raw_files:
        raise ValueError(f"No images found in {raw_dir}")

    saved = 0
    attempts = 0

    while saved < n_patches and attempts < n_patches * max_tries_per_patch:
        attempts += 1
        raw_path = random.choice(raw_files)
        cut_path = cutout_dir / raw_path.name
        if not cut_path.exists():
            continue

        raw = Image.open(raw_path).convert("RGB")
        cut = Image.open(cut_path).convert("RGBA")  # Use cutout alpha as a foreground mask proxy to avoid sampling patches that contain the jaguar.

        if raw.size != cut.size:
            # if misaligned, skip (or resize cut alpha to raw)
            continue

        W, H = raw.size
        if W < patch_size or H < patch_size:
            continue

        alpha = np.array(cut.getchannel("A"))  # H,W
        # sample a random crop location
        x0 = random.randint(0, W - patch_size)
        y0 = random.randint(0, H - patch_size)

        a_crop = alpha[y0:y0+patch_size, x0:x0+patch_size]
        fg_frac = (a_crop > 0).mean()

        if fg_frac > max_fg_frac:       # Keep only near-background patches; small tolerance handles imperfect cutout masks/edges.
            continue

        patch = raw.crop((x0, y0, x0+patch_size, y0+patch_size))
        out_path = out_dir / f"bg_{saved:06d}.jpg"
        patch.save(out_path, quality=90)
        saved += 1

        if saved % 200 == 0:
            print(f"Saved {saved}/{n_patches} backgrounds...")

    print(f"Done. Saved {saved} patches to {out_dir}. Attempts={attempts}")


if __name__ == "__main__":

    out_dir = resolve_path("backgrounds", DATA_STORE)

    # Assumes cutout alpha masks are stored alongside/raw-equivalent filenames in PATHS.data_train.
    # If raw and cutout folders differ in your setup, pass separate directories here.
    build_habitat_backgrounds(
        raw_dir=PATHS.data_train,
        cutout_dir=PATHS.data_train,
        out_dir=out_dir,
        n_patches=2000,
        patch_size=224,
    )