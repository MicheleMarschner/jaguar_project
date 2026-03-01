from typing import Tuple
from PIL import Image, ImageStat
import cv2

def calculate_sharpness(img_path):
    """
    Fast sharpness proxy (variance of Laplacian), good enough for ranking candidates in EDA/XAI tooling.
    """
    image = cv2.imread(str(img_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def brightness_score(img_pil: Image.Image) -> float:
    """Mean luminance in ~[0..255]."""
    return float(ImageStat.Stat(img_pil.convert("L")).mean[0])


def quality_score(img_path, sharpness_fn, low_bright=25, high_bright=230) -> Tuple[float, float, float]:
    """
    Lightweight image-quality heuristic for *example selection* (not dataset filtering):
    combine sharpness with a brightness penalty so chosen pairs are easier to inspect visually.
    
    Simple quality score:
      - higher sharpness is better
      - penalize too-dark / too-bright images
    """
    img = Image.open(str(img_path)).convert("RGB")
    sharp = float(sharpness_fn(img_path))
    bright = brightness_score(img)

    penalty = 0.0
    if bright < low_bright:
        penalty = (low_bright - bright) / low_bright
    elif bright > high_bright:
        penalty = (bright - high_bright) / (255 - high_bright)

    score = sharp - 50.0 * penalty
    return score, sharp, bright