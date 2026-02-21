import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from jaguar.utils.utils_models import (
    load_megadescriptor_model,
    load_dino_model,  # For DINO/DINOv2
    load_convnext_v2,
    load_efficientnet_b4,
    load_swin_transformer,
    load_eva_02
)

class FoundationModelWrapper:
    MODEL_REGISTRY = {
        "MegaDescriptor-L": {
            "loader": lambda: load_megadescriptor_model("L"),
            "transform": transforms.Compose([
                transforms.Resize((256, 128), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        },
        "DINO-Small": {
            "loader": lambda: load_dino_model("dino", "small", patch_size=8, pretrained=True),
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        },
        "DINOv2-Base": {
            "loader": lambda: load_dino_model("dinov2", "base", patch_size=14, pretrained=True, with_register_tokens=False),
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        },
        "ConvNeXt-V2": {
            "loader": lambda: load_convnext_v2("large"),
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        },
        "EfficientNet-B4": {
            "loader": lambda: load_efficientnet_b4(),
            "transform": transforms.Compose([
                transforms.Resize((380, 380), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        },
        "Swin-Transformer": {
            "loader": lambda: load_swin_transformer("base"),
            "transform": transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        },
        "EVA-02": {
            "loader": lambda: load_eva_02(),
            "transform": transforms.Compose([
                transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
        }
    }

    def __init__(self, model_name: str, device="cuda"):
        assert model_name in self.MODEL_REGISTRY, f"Unknown model {model_name}"
        self.name = model_name
        self.device = device

        print(f"[Info] Loading model {model_name} on {device}...")
        self.model = self.MODEL_REGISTRY[model_name]["loader"]()
        self.model.to(device)
        self.model.eval()

        self.transform = self.MODEL_REGISTRY[model_name]["transform"]

    def preprocess(self, image: Image.Image):
        return self.transform(image)

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

    def save_embeddings(self, embeddings: np.ndarray, split="training", folder="data/embeddings"):
        os.makedirs(folder, exist_ok=True)
        filename = f"embeddings_{self.name}_{split}.npy"
        path = os.path.join(folder, filename)
        np.save(path, embeddings)
        print(f"[Info] Saved embeddings to {path}")
        return path

    def load_embeddings(self, split="training", folder="data/embeddings"):
        filename = f"embeddings_{self.name}_{split}.npy"
        path = os.path.join(folder, filename)
        emb = np.load(path)
        print(f"[Info] Loaded embeddings from {path}, shape={emb.shape}")
        return emb

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