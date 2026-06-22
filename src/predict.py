from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def get_inference_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_image(
    image_path: str | Path,
    model: torch.nn.Module,
    class_names: list[str],
    device: str = "cpu",
    image_size: int = 224,
) -> tuple[str, float, dict]:
    tf  = get_inference_transform(image_size)
    img = Image.open(image_path).convert("RGB")
    x   = tf(img).unsqueeze(0).to(device)

    model.eval()
    logits = model(x)
    probs  = torch.softmax(logits, dim=1).squeeze()

    pred_idx   = probs.argmax().item()
    pred_label = class_names[pred_idx]
    confidence = probs[pred_idx].item()
    all_probs  = {cls: probs[i].item() for i, cls in enumerate(class_names)}

    return pred_label, confidence, all_probs
