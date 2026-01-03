import torch
import torchvision.transforms.v2 as transforms
import timm

from config import DEVICE

def load_dinov2(dinoname="dinov2_vitb14", frozen=True, device=DEVICE):
  # ---- Load DINOv2 from torch.hub ----
  # pick one:
  #   dinov2_vits14  (fastest, 384-d)
  #   dinov2_vitb14  (768-d)
  #   dinov2_vitl14  (1024-d)
  #   dinov2_vitg14  (1536-d)

  model = torch.hub.load("facebookresearch/dinov2", dinoname)
  model.eval().to(device)

  if frozen == True:
    for p in model.parameters():
        p.requires_grad_(False)

  # ---- Preprocess (good default) ----
  # DINOv2 works well with 518 resize/crop; 224 also works but can be weaker for fine detail.
  transform = transforms.Compose([
      transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
      transforms.CenterCrop(518),
      transforms.ToImage(),
      transforms.ToDtype(torch.float32, scale=True),
      transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
  ])

  return model, transform


def load_mega_descriptor(frozen=True, device=DEVICE):
   # num_classes=0 ensures we get the embedding, not class logits
   model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)
   model.eval().to(device)

   # Get the correct config from the model itself
   config = timm.data.resolve_data_config({}, model=model)
   # This automatically creates the exact transform the model was trained with
   transform = timm.data.create_transform(**config, is_training=False)

   if frozen == True:
    for p in model.parameters():
        p.requires_grad_(False)

    return model, transform
   

def load_dinov2_for_wildlife(frozen=True, device=DEVICE):
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-DINOv2-518", pretrained=True)
    model = model.eval().to(device)

    if frozen == True:
        for p in model.parameters():
            p.requires_grad_(False)
    
    transform = transforms.Compose([
        transforms.Resize(518, 518),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return model, transform
