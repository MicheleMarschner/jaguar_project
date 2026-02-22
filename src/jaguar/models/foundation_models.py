import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode

from jaguar.config import PATHS
from jaguar.utils.utils_models import (
    load_megadescriptor_model,
    load_dino_model,  # For DINO/DINOv2
    load_megadescriptor_dino_model,
    load_convnext_v2,
    load_efficientnet_b4,
    load_swin_transformer,
    load_eva_02
)
from jaguar.utils.utils_explainer import (
    vit_reshape_transform, 
    swin_reshape_transform, 
    lrp_zennit_input_relevance
)

class FoundationModelWrapper:
    MODEL_REGISTRY = {
        "MegaDescriptor-L": {
            "loader": lambda: load_megadescriptor_model("L"),
            "transform": transforms.Compose([
                transforms.Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                # This automatically creates the exact transform the model was trained with
                # config = timm.data.resolve_data_config({}, model=model)
                # transform = timm.data.create_transform(**config, is_training=False)
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.layer4[-1], 
                "reshape_transform": None
            }
        },
        "DINO-Small": {
            "loader": lambda: load_dino_model("dino", "small", patch_size=8, pretrained=True),
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.blocks[-1].norm1,
                "reshape_transform": vit_reshape_transform
            }
        },
        "DINOv2-Base": {
            "loader": lambda: load_dino_model("dinov2", "base", patch_size=14, pretrained=True, with_register_tokens=False),
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                # DINOv2 works well with 518 resize/crop; 224 also works but can be weaker for fine detail.
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.blocks[-1].norm1,
                "reshape_transform": vit_reshape_transform
            }
        },
        "DINOv2_for_wildlife": {
            "loader": lambda: load_megadescriptor_dino_model(),
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Resize(518, 518)
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.blocks[-1].norm1,
                "reshape_transform": vit_reshape_transform
            }
        },
        "ConvNeXt-V2": {
            "loader": lambda: load_convnext_v2("large"),
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.stages[-1].blocks[-1],
                "reshape_transform": None 
            },
            "lrp": { 
                "backend": "zennit",
                "composite": "epsilon_plus",
                "canonizer": None, 
            },
        },
        "EfficientNet-B4": {
            "loader": lambda: load_efficientnet_b4(),
            "transform": transforms.Compose([
                transforms.Resize((380, 380), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.features[-1],
                "reshape_transform": None
            },
            "lrp": { 
                "backend": "zennit",
                "composite": "epsilon_plus",
                "canonizer": None,
            },
        },
        "Swin-Transformer": {
            "loader": lambda: load_swin_transformer("base"),
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.features[-1][-1].norm1,
                "reshape_transform": swin_reshape_transform
            }
        },
        "EVA-02": {
            "loader": lambda: load_eva_02(),
            "transform": transforms.Compose([
                transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.blocks[-1].norm1,
                "reshape_transform": vit_reshape_transform
            }
        }
    }

    def __init__(self, model_name: str, device=None):
        assert model_name in self.MODEL_REGISTRY, f"Unknown model {model_name}"
        self.name = model_name
        self.device = device

        print(f"[Info] Loading model {model_name} on {device}...")
        self.registry_entry = self.MODEL_REGISTRY[model_name]
        self.model = self.MODEL_REGISTRY[model_name]["loader"]()
        self.model.to(device)
        self.model.eval()

        self.transform = self.MODEL_REGISTRY[model_name]["transform"]

    def preprocess(self, image: Image.Image):
        return self.transform(image)
    
    def eval(self):
        """Delegate eval() to the underlying PyTorch model."""
        self.model.eval()
        return self

    def train(self, mode=True):
        """Delegate train() to the underlying PyTorch model."""
        self.model.train(mode)
        return self

    def to(self, device):
        """Delegate to() to the underlying PyTorch model."""
        self.device = device
        self.model.to(device)
        return self

    def __call__(self, x):
        """
        Allows the wrapper to be called like a function: output = wrapper(x)
        This is often what evaluation scripts expect.
        """
        return self.model(x)

    def extract_embeddings(self, images):
        """
        images: list of PIL.Image or torch.Tensor
        Returns np.ndarray of shape (N, D)
        """
        tensors = []
        for img in images:
            if isinstance(img, Image.Image):
                tensors.append(self.preprocess(img))
            else:
                tensors.append(img)
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            # assume the model has `get_embeddings` method
            if hasattr(self.model, "get_embeddings"):
                emb = self.model.get_embeddings(batch)
            else:
                emb = self.model(batch)  # fallback: use raw output
        emb_np = emb.cpu().numpy()
        print(f"[Info] Extracted embeddings shape: {emb_np.shape}")
        return emb_np

    def save_embeddings(self, embeddings: np.ndarray, split="training", folder=PATHS.data / "embeddings"):
        os.makedirs(folder, exist_ok=True)
        filename = f"embeddings_{self.name}_{split}.npy"
        path = os.path.join(folder, filename)
        np.save(path, embeddings)
        print(f"[Info] Saved embeddings to {path}")
        return path

    def load_embeddings(self, split="training", folder=PATHS.data / "embeddings"):
        filename = f"embeddings_{self.name}_{split}.npy"
        path = os.path.join(folder, filename)
        emb = np.load(path)
        print(f"[Info] Loaded embeddings from {path}, shape={emb.shape}")
        return emb
    
    def get_grad_cam_config(self):
        """
        Returns the resolved target layers and reshape function for GradCAM.
        """
        if "grad_cam" not in self.registry_entry:
            raise NotImplementedError(f"GradCAM not configured for {self.name}")

        config = self.registry_entry["grad_cam"]
        target_layers = [config["layer_getter"](self.model)]
        
        return target_layers, config["reshape_transform"]
    
    def get_lrp_explainer(self):
        cfg = self.registry_entry.get("lrp")
        if cfg is None:
            raise NotImplementedError(f"LRP not configured for {self.name}")

        if cfg.get("backend") != "zennit":
            raise NotImplementedError("Only zennit backend implemented")

        composite = cfg.get("composite", "epsilon_plus")
        canonizers = cfg.get("canonizer", None)

        def explain(x: torch.Tensor, target_idx: int) -> np.ndarray:
            return lrp_zennit_input_relevance(
                model=self.model,
                x=x,
                target_idx=target_idx,
                composite=composite,
                canonizers=canonizers,
            )

        return explain
    
if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    # Create a dummy image
    dummy_img = Image.fromarray((np.random.rand(224,224,3)*255).astype(np.uint8))
    # Initialize model wrapper
    model_wrapper = FoundationModelWrapper("DINO-Small", device="cpu")
    # Extract embeddings
    embeddings = model_wrapper.extract_embeddings([dummy_img])
    print("Embeddings:", embeddings.shape)
    # Save and load embeddings
    model_wrapper.save_embeddings(embeddings, split="training")
    loaded_emb = model_wrapper.load_embeddings(split="training")