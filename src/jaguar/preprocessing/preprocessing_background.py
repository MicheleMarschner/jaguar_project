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

from jaguar.config import DATA_ROOT


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
        bg_path = DATA_ROOT / Path(bg_dir)
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
    
    @staticmethod
    def mixed_original_random_bg(
        img: Image.Image,
        sample: dict,
        base_root: Path,
        bg_dir: str,
        p_original: float = 0.5,
        key_for_seed: str = "filename",
        **kwargs,
    ) -> Image.Image:
        """
        Deterministically choose between original RGB and random background
        per (epoch, sample).
        """
        epoch = int(sample.get("_epoch", 0))
        seed_key = str(sample.get(key_for_seed))
        seed = hash(("mixed_original_random_bg", epoch, seed_key)) & 0xFFFFFFFF
        rng = random.Random(seed)

        if rng.random() < p_original:
            return img.convert("RGB")

        return ImageProcessor.random_bg_cutout_deterministic(
            img=img,
            sample=sample,
            base_root=base_root,
            bg_dir=bg_dir,
            key_for_seed=key_for_seed,
            **kwargs,
        )

# Registry used by dataset loading/preprocessing code to select a background policy by name.
PROCESSORS = {
    "original": ImageProcessor.original,
    "gray_bg": ImageProcessor.gray_bg_cutout_alpha,
    "random_bg": ImageProcessor.random_bg_cutout_deterministic,
    "black_bg": ImageProcessor.black_bg_cutout_alpha,
    "white_bg": ImageProcessor.white_bg_cutout_alpha,
    "blur_bg": ImageProcessor.blur_bg_cutout_alpha,
    "mixed_original_random_bg": ImageProcessor.mixed_original_random_bg,
}