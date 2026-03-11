import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaguar.utils.utils import resolve_path, save_npy
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode

from jaguar.config import DATA_STORE, IMGNET_MEAN, IMGNET_STD, PATHS
from jaguar.utils.utils_models import (
    load_megadescriptor_model,
    load_dino_model,  # For DINO/DINOv2
    load_megadescriptor_dino_model,
    load_convnext_v2,
    load_efficientnet_b4,
    load_swin_transformer,
    load_eva_02,
    load_miewid
)
from jaguar.utils.utils_models import (
    vit_reshape_transform, 
    swin_reshape_transform
)

class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        new_img = Image.new("RGB", (max_side, max_side), self.fill)
        new_img.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        return new_img

class FoundationModelWrapper:
    MODEL_REGISTRY = {
        "MegaDescriptor-L": {
            "loader": lambda: load_megadescriptor_model("L"),
            "input_size": 384,
            "transform": transforms.Compose([
                transforms.Resize((384, 384), interpolation=InterpolationMode.BILINEAR),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
                # This automatically creates the exact transform the model was trained with
                # config = timm.data.resolve_data_config({}, model=model)
                # transform = timm.data.create_transform(**config, is_training=False)
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.layer4[-1], 
                "reshape_transform": None
            },
            "supports_progressive_resizing": False,
        },
        "DINO-Small": {
            "loader": lambda: load_dino_model("dino", "small", patch_size=8, pretrained=True),
            "input_size": 224,
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.blocks[-1].norm1,
                "reshape_transform": vit_reshape_transform
            },
            "supports_progressive_resizing": False,
        },
        "DINOv2-Base": {
            "loader": lambda: load_dino_model("dinov2", "base", patch_size=14, pretrained=True, with_register_tokens=False),
            "input_size": 224,
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
                # DINOv2 works well with 518 resize/crop; 224 also works but can be weaker for fine detail.
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.blocks[-1].norm1,
                "reshape_transform": vit_reshape_transform
            },
            "supports_progressive_resizing": False,
        },
        "DINOv3-Base": {
            "loader": lambda: load_dino_model("dinov3", "base", patch_size=16, pretrained=True, with_register_tokens=False),
            "input_size": 224,
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.blocks[-1].norm1,
                "reshape_transform": vit_reshape_transform
            },
            "supports_progressive_resizing": False,
        },
        "DINOv3-Large": {
            "loader": lambda: load_dino_model("dinov3", "large", patch_size=16, pretrained=True, with_register_tokens=False),
            "input_size": 224,
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD),
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.blocks[-1].norm1,
                "reshape_transform": vit_reshape_transform
            },
            "supports_progressive_resizing": False,
        },
        "DINOv2_for_wildlife": {
            "loader": lambda: load_megadescriptor_dino_model(),
            "input_size": 518,
            "transform": transforms.Compose([
                transforms.Resize((518, 518), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
                # transforms.Resize(518, 518)
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.blocks[-1].norm1,
                "reshape_transform": vit_reshape_transform
            },
            "supports_progressive_resizing": False,
        },
        "ConvNeXt-V2": {
            "loader": lambda: load_convnext_v2("large"),
            "input_size": 224,
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.features[-1][-1],
                "reshape_transform": None 
            },
            "supports_progressive_resizing": True,
        },
        "EfficientNet-B4": {
            "loader": lambda: load_efficientnet_b4(),
            "input_size": 380,
            "transform": transforms.Compose([
                transforms.Resize((380, 380), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.features[-1],
                "reshape_transform": None
            },
            "supports_progressive_resizing": True,
        },
        "Swin-Transformer": {
            "loader": lambda: load_swin_transformer("base"),
            "input_size": 224,
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.features[-1][-1].norm1,
                "reshape_transform": swin_reshape_transform
            },
            "supports_progressive_resizing": False,
        },
        "EVA-02": {
            "loader": lambda: load_eva_02(),
            "input_size": 448, #224,
            "transform": transforms.Compose([
                transforms.Resize((448,448), interpolation=InterpolationMode.BICUBIC), #224,224
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.blocks[-1].norm1,
                "reshape_transform": vit_reshape_transform
            },
            "supports_progressive_resizing": False,
        },
        "MiewID": {
            "loader": lambda: load_miewid(),
            "input_size": 440,
            "transform": transforms.Compose([
                transforms.Resize((440,440), interpolation=InterpolationMode.BICUBIC),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(IMGNET_MEAN, IMGNET_STD)
            ]),
            "grad_cam": {
                "layer_getter": lambda m: m.backbone.conv_head,
                "reshape_transform": None
            },
            "supports_progressive_resizing": False,
        }
    }

    def __init__(self, model_name: str, device=None):
        assert model_name in self.MODEL_REGISTRY, f"Unknown model {model_name}"
        self.name = model_name
        self.device = device

        print(f"[Info] Loading model {model_name} on {device}...")
        self.registry_entry = self.MODEL_REGISTRY[model_name]
        self.input_size = self.registry_entry.get("input_size") 
        self.model = self.MODEL_REGISTRY[model_name]["loader"]()
        self.supports_progressive_resizing = self.registry_entry.get(
            "supports_progressive_resizing", False
        )
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
        
        #TO-DO: re-check whether we need the part commented out 
        with torch.no_grad():
            # Process the batch in one go
            emb = self.model(batch)  # Direct call for embedding extraction
        # with torch.no_grad():
        #     # assume the model has `get_embeddings` method
        #     if hasattr(self.model, "get_embeddings"):
        #         emb = self.model.get_embeddings(batch)
        #     else:
        #         emb = self.model(batch)  # fallback: use raw output
        emb_np = emb.cpu().numpy()
        print(f"[Info] Extracted embeddings shape: {emb_np.shape}")
        return emb_np

    def save_embeddings(self, embeddings: np.ndarray, split="training", folder=None):
        if folder is None:
            folder = resolve_path("embeddings", DATA_STORE)
        os.makedirs(folder, exist_ok=True)
        filename = f"embeddings_{self.name}_{split}.npy"
        path = os.path.join(folder, filename)
        save_npy(path, embeddings)
        print(f"[Info] Saved embeddings to {path}")
        return path

    def load_embeddings(self, split="training", folder=None):
        if folder is None:
            folder = resolve_path("embeddings", DATA_STORE)
        filename = f"embeddings_{self.name}_{split}.npy"
        path = os.path.join(folder, filename)
        emb = np.load(path)
        print(f"[Info] Loaded embeddings from {path}, shape={emb.shape}")
        return emb
    
    def get_embeddings_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] already preprocessed tensor on self.device
        returns: [B,D] torch tensor (keeps graph if x.requires_grad=True)
        """
        out = self.model.get_embeddings(x) if hasattr(self.model, "get_embeddings") else self.model(x)

        if isinstance(out, dict):
            out = out.get("embeddings") or next(iter(out.values()))
        elif isinstance(out, (tuple, list)):
            out = out[0]

        if out.ndim == 1:
            out = out.unsqueeze(0)
        return out
    
    def get_grad_cam_config(self):
        """
        Returns the resolved target layers and reshape function for GradCAM.
        """
        if "grad_cam" not in self.registry_entry:
            raise NotImplementedError(f"GradCAM not configured for {self.name}")

        config = self.registry_entry["grad_cam"]
        target_layers = [config["layer_getter"](self.model)]
        
        return target_layers, config["reshape_transform"]

    
if __name__ == "__main__":
    
    # Create a dummy image
    dummy_img = Image.fromarray((np.random.rand(448,448,3)*255).astype(np.uint8))
    # Initialize model wrapper
    model_wrapper = FoundationModelWrapper("EVA-02", device="cpu")
    # Extract embeddings
    embeddings = model_wrapper.extract_embeddings([dummy_img])
    print("Embeddings:", embeddings.shape)
    # Save and load embeddings
    model_wrapper.save_embeddings(embeddings, split="training")
    loaded_emb = model_wrapper.load_embeddings(split="training")