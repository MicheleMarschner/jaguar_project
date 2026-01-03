from PIL import Image, ImageFilter
import numpy as np
import torch
from utils import to_numpy


class ImageProcessor:
    @staticmethod
    def original(image, mask=None):
        """Returns the raw image."""
        return image

    @staticmethod
    def black_background(image, mask):
        """Sets everything outside the mask to black."""
        black_bg = Image.new("RGB", image.size, (0, 0, 0))
        return Image.composite(image, black_bg, mask)

    @staticmethod
    def blur_background(image, mask, radius=10):
        """Blurs everything outside the mask."""
        blurred_bg = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return Image.composite(image, blurred_bg, mask)

    @staticmethod
    def white_background(image, mask):
        """Sets everything outside the mask to white."""
        white_bg = Image.new("RGB", image.size, (255, 255, 255))
        return Image.composite(image, white_bg, mask)


def pick_best_mask_box(masks, boxes, scores):
    """
    masks: Tensor [N,1,H,W] or [N,H,W] or []
    boxes: Tensor [N,4]
    scores: Tensor [N]
    Returns (best_mask_hw_bool, best_box_4_float, best_score_float) or (None,None,None)
    """
    if masks is None:
        return None, None, None

    # sometimes SAM returns Tensor with shape (0, ...)
    if torch.is_tensor(masks) and masks.shape[0] == 0:
        return None, None, None
    if isinstance(masks, (list, tuple)) and len(masks) == 0:
        return None, None, None

    # normalize shapes
    masks_np = to_numpy(masks)
    boxes_np = to_numpy(boxes) if boxes is not None else None
    scores_np = to_numpy(scores) if scores is not None else None

    # choose best by score if available, else first
    if scores_np is not None and len(scores_np) > 0:
        idx = int(np.argmax(scores_np))
        best_score = float(scores_np[idx])
    else:
        idx = 0
        best_score = None

    m = masks_np[idx]
    m = np.squeeze(m)  # -> [H,W]
    best_mask = m > 0.0  # boolean mask

    best_box = boxes_np[idx] if boxes_np is not None and len(boxes_np) > 0 else None
    if best_box is not None:
        best_box = best_box.astype(float)

    return best_mask, best_box, best_score


def clamp_and_pad_box(box, W, H, pad_frac=0.10):
    """
    box: [x1,y1,x2,y2] floats
    pad_frac: 0.10 means add 10% of box size as margin on each side
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    pad_x = pad_frac * bw
    pad_y = pad_frac * bh

    x1 = max(0, int(np.floor(x1 - pad_x)))
    y1 = max(0, int(np.floor(y1 - pad_y)))
    x2 = min(W, int(np.ceil(x2 + pad_x)))
    y2 = min(H, int(np.ceil(y2 + pad_y)))

    # guard
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)

    return x1, y1, x2, y2


def mask_to_rgba_foreground(image_rgb_pil: Image.Image, mask_hw_bool: np.ndarray) -> Image.Image:
    """
    Returns RGBA image with alpha = mask.
    """
    img = np.array(image_rgb_pil)  # H,W,3
    H, W = img.shape[:2]
    mask = mask_hw_bool.astype(np.uint8) * 255
    rgba = np.dstack([img, mask])  # H,W,4
    return Image.fromarray(rgba, mode="RGBA")


def get_gray_background_from_rgba(image_path):
    """
    Loads an image, composites it over a neutral gray background, and converts it to RGB.
    Args:
        image_path (str): The file path to the input image (rgba).
    Returns:
        PIL.Image.Image: The resulting composite image as rgb.
    """
    img_rgba = Image.open(image_path).convert("RGBA")
    
    # Create a solid gray background
    # Using (128, 128, 128) minimizes strong contrast edges
    background = Image.new("RGBA", img_rgba.size, (128, 128, 128, 255))
    
    # Paste the jaguar on top of the background using the Alpha channel as a mask
    combined = Image.alpha_composite(background, img_rgba)
    
    # Convert to RGB
    return combined.convert("RGB")


def get_blurred_background_image(image_path, mask_path, bg_blur_radius=10, edge_softness=2):
    """
    Creates an image where the foreground is sharp, but the background is heavily blurred
    using pure PIL
    
    Args:
        image_path (str): Path to the original RGB image.
        mask_path (str): Path to the binary mask (white=foreground, black=bg).
        bg_blur_radius (int): PIL Gaussian blur radius (approx 10-50 for heavy blur).
        edge_softness (int): Radius to blur the mask edges (feathering).
        
    Returns:
        PIL.Image: The ready-to-use composited image.
    """
    # 1. Load Images
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L") # Load as grayscale (L)
    
    # Safety: Resize mask to match image if dimensions differ
    if img.size != mask.size:
        mask = mask.resize(img.size, resample=Image.NEAREST)

    # 2. Create the "Context" (Blurred Background)
    # We blur the original image to act as the background layer
    blurred_bg = img.filter(ImageFilter.GaussianBlur(radius=bg_blur_radius))

    # 3. Process the Mask (Feathering)
    # We blur the mask slightly to create soft edges (alpha gradients)
    if edge_softness > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_softness))

    # 4. Composite
    # Logic: "Show 'img' where mask is white, show 'blurred_bg' where mask is black"
    combined = Image.composite(img, blurred_bg, mask)
    
    return combined