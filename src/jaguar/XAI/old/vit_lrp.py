import torch
import numpy as np
import torch.nn.functional as F

#import dino.vision_transformer as dino_vit
from jaguar.config import PATHS, DEVICE
from jaguar.utils.utils_datasets import load_jaguar_from_FO_export
from jaguar.models.foundation_models import FoundationModelWrapper



if __name__ == "__main__":
    model_name = "MegaDescriptor-L"         # DINOv2-Base, MiewID, ConvNeXt-V2, MegaDescriptor-L
    dataset_name = "jaguar_init"
    base_root = PATHS.data_export / "init"

    fo_ds, torch_ds = load_jaguar_from_FO_export(
        PATHS.data_export / "init",
        dataset_name=dataset_name,
        processing_fn=None,
        overwrite_db=False,
    )

    print("[Info] Dataset size:", len(torch_ds))

    filepaths = [str(torch_ds._resolve_path(s[torch_ds.filepath_key])) for s in torch_ds.samples]
    labels    = np.asarray(torch_ds.labels)

    # Load model wrapper
    model_wrapper = FoundationModelWrapper(model_name, device=DEVICE)
    split = "training"
    print("[Info] Loaded model:", model_wrapper.name)

    print(model_wrapper.model)