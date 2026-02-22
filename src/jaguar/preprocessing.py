import random
from pathlib import Path
from PIL import Image, ImageFilter

### add a method that applies the same background to all -> random_bg_cutout but the same for all

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
    def blur_bg_mask(
        img: Image.Image,
        sample: dict,
        base_root: Path,
        mask_key="mask_path",
        blur_radius: int = 10,
        edge_softness: int = 2,
        **kwargs,
    ) -> Image.Image:
        mask_path = _resolve(base_root, sample[mask_key])
        mask = Image.open(mask_path).convert("L")
        if mask.size != img.size:
            mask = mask.resize(img.size, resample=Image.NEAREST)
        if edge_softness > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_softness))
        blurred_bg = img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return Image.composite(img.convert("RGB"), blurred_bg, mask)

PROCESSORS = {
    "original": ImageProcessor.original,
    "cutout_gray_bg": ImageProcessor.gray_bg_cutout,
    "cutout_random_bg": ImageProcessor.random_bg_cutout,
    "mask_black_bg": ImageProcessor.black_bg_mask,
    "mask_white_bg": ImageProcessor.white_bg_mask,
    "mask_blur_bg": ImageProcessor.blur_bg_mask,
}