import random
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

from jaguar.config import PATHS

def _resolve(base_root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base_root / pp)

def rgba_on_solid_bg(rgba: Image.Image, color=(128, 128, 128)) -> Image.Image:
    rgba = rgba.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (*color, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")

def rgba_on_random_bg(rgba: Image.Image, bg_path: Path) -> Image.Image:
    rgba = rgba.convert("RGBA")
    bg = Image.open(bg_path).convert("RGB").resize(rgba.size).convert("RGBA")
    return Image.alpha_composite(bg, rgba).convert("RGB")

def mask_composite(image_rgb: Image.Image, mask_l: Image.Image, bg: Image.Image) -> Image.Image:
    if mask_l.size != image_rgb.size:
        mask_l = mask_l.resize(image_rgb.size, resample=Image.NEAREST)
    return Image.composite(image_rgb, bg, mask_l)

class ImageProcessor:
    @staticmethod
    def original(img: Image.Image, sample: dict, base_root: Path, **kwargs) -> Image.Image:
        # for raw RGB: return RGB
        # for cutout RGBA: return RGB with neutral bg (so model has 3 channels)
        if img.mode == "RGBA":
            return rgba_on_solid_bg(img, color=(128, 128, 128))
        return img.convert("RGB")

    @staticmethod
    def gray_bg_cutout(img: Image.Image, sample: dict, base_root: Path, **kwargs) -> Image.Image:
        print("[DEBUG] gray_bg_cutout called | mode:", img.mode)  
        # expects img is RGBA cutout
        return rgba_on_solid_bg(img, color=(128, 128, 128))

    @staticmethod
    def random_bg_cutout_deterministic(
        img: Image.Image,
        sample: dict,
        base_root: Path,
        bg_dir: str,
        key_for_seed: str = "filename",   # or "filepath"
        **kwargs,
    ) -> Image.Image:
        bg_dir = Path(bg_dir)
        bg_files = [p for p in bg_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if not bg_files:
            raise ValueError(f"No backgrounds in {bg_dir}")

        # deterministic seed per (epoch, sample)
        epoch = int(sample.get("_epoch", 0))
        seed_key = str(sample.get(key_for_seed) or sample.get("filepath") or "")
        seed = hash((epoch, seed_key)) & 0xFFFFFFFF

        rng = random.Random(seed)
        bg_path = rng.choice(bg_files)

        return rgba_on_random_bg(img, bg_path)

    @staticmethod
    def black_bg_mask(img: Image.Image, sample: dict, base_root: Path, mask_key="mask_path", **kwargs) -> Image.Image:
        # expects raw RGB + mask_path in sample
        mask_path = _resolve(base_root, sample[mask_key])
        mask = Image.open(mask_path).convert("L")
        black = Image.new("RGB", img.size, (0, 0, 0))
        return mask_composite(img.convert("RGB"), mask, black)

    @staticmethod
    def white_bg_mask(img: Image.Image, sample: dict, base_root: Path, mask_key="mask_path", **kwargs) -> Image.Image:
        mask_path = _resolve(base_root, sample[mask_key])
        mask = Image.open(mask_path).convert("L")
        white = Image.new("RGB", img.size, (255, 255, 255))
        return mask_composite(img.convert("RGB"), mask, white)

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
        Uses alpha as mask:
        - foreground stays sharp
        - background becomes blurred (or a solid bg) and edges can be feathered
        """
        rgba = img.convert("RGBA")
        rgb = rgba.convert("RGB")
        alpha = rgba.getchannel("A")  # L mask

        # feather edges
        if edge_softness > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=edge_softness))

        # background: either blur original RGB or use solid color
        blurred_bg = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # OR solid bg:
        # blurred_bg = Image.new("RGB", rgb.size, bg_color)

        # composite: show rgb where alpha is white, else show blurred_bg
        out = Image.composite(rgb, blurred_bg, alpha)
        return out

PROCESSORS = {
    "original": ImageProcessor.original,
    "cutout_gray_bg": ImageProcessor.gray_bg_cutout,
    "cutout_random_bg": ImageProcessor.random_bg_cutout_deterministic,
    "mask_black_bg": ImageProcessor.black_bg_mask,
    "mask_white_bg": ImageProcessor.white_bg_mask,
    "mask_blur_bg": ImageProcessor.blur_bg_cutout_alpha,
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
    raw_dir: folder with raw RGB images (same filenames as cutouts)
    cutout_dir: folder with RGBA cutouts where alpha=foreground mask (same filenames)
    out_dir: will be created and filled with background crops
    """
    random.seed(seed)
    raw_dir = Path(raw_dir)
    cutout_dir = Path(cutout_dir)
    out_dir = Path(out_dir)
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
        cut = Image.open(cut_path).convert("RGBA")

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

        if fg_frac > max_fg_frac:
            continue

        patch = raw.crop((x0, y0, x0+patch_size, y0+patch_size))
        out_path = out_dir / f"bg_{saved:06d}.jpg"
        patch.save(out_path, quality=90)
        saved += 1

        if saved % 200 == 0:
            print(f"Saved {saved}/{n_patches} backgrounds...")

    print(f"Done. Saved {saved} patches to {out_dir}. Attempts={attempts}")


if __name__ == "__main__":

    out_dir = PATHS.data / "backgrounds"

    build_habitat_backgrounds(
        raw_dir=PATHS.data_train,
        cutout_dir=PATHS.data_train,
        out_dir=out_dir,
        n_patches=2000,
        patch_size=224,
    )