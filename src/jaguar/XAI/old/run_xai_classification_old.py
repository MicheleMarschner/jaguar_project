import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from jaguar.config import DEVICE, PATHS, IMGNET_MEAN, IMGNET_STD
from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.utils.utils_xai import MaskAwareJaguarDataset


# ---------------------------------------------------------
# 2. EVALUATION LOGIC
# ---------------------------------------------------------
def compute_sensitivity_scores(model, dataloader) -> pd.DataFrame:
    """
    Runs the masking experiment:
    1. Pass Original -> Record Score
    2. Pass BG_Masked -> Record Score
    3. Pass FG_Masked -> Record Score
    """
    model.eval()
    model.to(DEVICE)
    results = []

    print("--- Computing Sensitivity Drops ---")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move to device
            orig = batch["t_orig"].to(DEVICE)
            bg_masked = batch["t_bg_masked"].to(DEVICE)
            fg_masked = batch["t_fg_masked"].to(DEVICE)
            targets = batch["label_idx"].to(DEVICE)
            
            # Forward Passes
            # Assumption: Model returns logits [B, NumClasses]
            logits_orig = model(orig)
            logits_bg_masked = model(bg_masked)
            logits_fg_masked = model(fg_masked)
            
            # Convert to Probabilities
            probs_orig = F.softmax(logits_orig, dim=1)
            probs_bg = F.softmax(logits_bg_masked, dim=1)
            probs_fg = F.softmax(logits_fg_masked, dim=1)
            
            # Extract Metrics per Sample
            for i in range(len(targets)):
                target_cls = targets[i].item()
                
                s0 = probs_orig[i, target_cls].item()
                s_bg = probs_bg[i, target_cls].item() # Score with ONLY Jaguar
                s_fg = probs_fg[i, target_cls].item() # Score with ONLY Background
                
                results.append({
                    "id": batch["id"][i],
                    "filepath": Path(batch["filepath"][i]).name,
                    "score_orig": s0,
                    "score_jaguar_only": s_bg,
                    "score_bg_only": s_fg,
                    "drop_bg": s0 - s_bg, # High drop = Bad (needs context)
                    "drop_fg": s0 - s_fg, # High drop = Good (needs jaguar)
                    # Spurious if model is more confident in BG than Jaguar
                    "is_spurious": s_fg > s_bg 
                })
                
    return pd.DataFrame(results)



def generate_gradcam_overlay(model, target_layer, input_tensor, target_category_idx=None):
    """
    Wraps pytorch-grad-cam to generate a heatmap for a single image.
    input_tensor: [1, C, H, W]
    """
    # Initialize Cam
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Define Target (if None, uses highest confidence class)
    targets = [ClassifierOutputTarget(target_category_idx)] if target_category_idx is not None else None

    # Generate Grayscale CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # Take first in batch
    
    # Create RGB overlay
    # We need the un-normalized image for visualization
    # Quick un-norm (approximate is fine for vis)
    img_disp = input_tensor.squeeze().cpu().permute(1,2,0).numpy()
    img_disp = img_disp * IMGNET_STD + IMGNET_MEAN
    img_disp = np.clip(img_disp, 0, 1)
    
    visualization = show_cam_on_image(img_disp, grayscale_cam, use_rgb=True)
    return visualization



def save_visual_report(model, target_layer, dataloader, df_results, num_samples=5):
    """
    Saves a panel: [Original + CAM] | [BG Masked + CAM] | [FG Masked + CAM]
    Only for the top 'spurious' cases (where BG was more important).
    """
    print("--- Generating Visual Explanations (GradCAM) ---")
    vis_path = PATHS.results / "visuals"
    vis_path.mkdir(parents=True, exist_ok=True)
    
    # Filter for interesting cases (e.g., spurious ones)
    spurious_cases = df_results[df_results["is_spurious"] == True]
    
    # If no spurious cases, just take random ones
    candidates = spurious_cases if not spurious_cases.empty else df_results
    candidates = candidates.head(num_samples)
    
    # Map filenames back to indices in dataloader (simple search)
    # Note: In production, better to lookup by index directly
    target_filenames = set(candidates["filepath"].values)
    
    count = 0
    model.eval()
    
    for batch in dataloader:
        for i in range(len(batch["filepath"])):
            fname = Path(batch["filepath"][i]).name
            if fname in target_filenames:
                
                # Inputs [1, C, H, W]
                t_orig = batch["t_orig"][i].unsqueeze(0).to(DEVICE)
                t_bg = batch["t_bg_masked"][i].unsqueeze(0).to(DEVICE)
                t_fg = batch["t_fg_masked"][i].unsqueeze(0).to(DEVICE)
                target_cls = batch["label_idx"][i].item()
                
                # Generate CAMs
                cam_orig = generate_gradcam_overlay(model, target_layer, t_orig, target_cls)
                cam_bg   = generate_gradcam_overlay(model, target_layer, t_bg, target_cls)
                cam_fg   = generate_gradcam_overlay(model, target_layer, t_fg, target_cls)
                
                # Stitch side-by-side
                combined = np.hstack((cam_orig, cam_bg, cam_fg))
                
                # Save
                cv2.imwrite(str(vis_path / f"gradcam_{fname}"), combined)
                count += 1
                if count >= num_samples: return


# ---------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------
def run_xai_pipeline(model_wrapper, samples_list, mode="similarity"):
    """
    model_wrapper: The initialized FoundationModelWrapper instance
    mode: 'logit' (if model has classification head) or 'similarity' (if embedding only)
    """
    
    config = model_wrapper.get_config()
    model = model_wrapper.model
    model.eval()

    # 1. Setup Dataset (Passes wrapper to handle transforms)
    ds = MaskAwareJaguarDataset(
        model_wrapper=model_wrapper, 
        base_root=PATHS.train_data,
        samples_list=samples_list,
        is_test=False 
    )
    
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    
    compute_logit_drops(model, loader, device=model_wrapper.device)
    
    # 3. Save Results
    save_dir = PATHS.results / "xai" / model_wrapper.model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "sensitivity.csv", index=False)
    
    # 4. GradCAM
    # Retrieve layer getter from registry
    layer_getter = config["grad_cam"].get("layer_getter")
    
    if layer_getter:
        target_layer = layer_getter(model)
        save_visual_report(model, target_layer, loader, df, save_dir)

# --- MOCK USAGE ---
if __name__ == "__main__":
    BATCH_SIZE = 32
    model_name = "MiewID"
    
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    
    # Load your samples list
    # samples = load_my_json(...)
    # load the whole dataset and pick randomly 150 samples from val keep curated=True?
    
    # run_xai_classification(model_wrapper, my_samples, BATCH_SIZE)
    pass






import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List
from PIL import Image
import torchvision.transforms.v2 as transforms

# Import your project modules
from jaguar.models.jaguar_id_model import JaguarIDModel
from jaguar.models.foundation_models import FoundationModelWrapper
from jaguar.utils.utils_datasets import JaguarDataset # Your base dataset
from jaguar.config import PATHS, DEVICE

# Import GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

