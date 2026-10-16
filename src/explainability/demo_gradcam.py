"""Grad-CAM demo: see a heat map for yourself.

Uses a ready-made ResNet-50 (trained on everyday photos, NOT on CT scans) and any
photo you give it. The point is only to understand what Grad-CAM does, before the
real PDAC classifier exists.

Run from the repo root (the folder that contains README.md):
    python src/explainability/demo_gradcam.py path/to/photo.jpg

Output: outputs/gradcam_demo.png (the "outputs" folder is ignored by Git)
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM demo")
    parser.add_argument("image", help="path to a .jpg or .png photo")
    parser.add_argument("--out", default="outputs/gradcam_demo.png")
    args = parser.parse_args()

    # 1. Load a ready-made ResNet-50. The first run downloads about 100 MB.
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).eval()
    categories = weights.meta["categories"]

    # 2. Load the photo and resize it to the 224x224 size the model expects.
    img = Image.open(args.image).convert("RGB").resize((224, 224))
    rgb = np.asarray(img, dtype=np.float32) / 255.0  # 0-1 values, used for the overlay
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    x = to_tensor(img).unsqueeze(0)  # shape (1, 3, 224, 224): a batch of one image

    # 3. Ask the model what it sees.
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = int(probs.argmax())
    label = f"{categories[top]} ({probs[top].item():.0%})"
    print("Model's top guess:", label)

    # 4. Grad-CAM: which parts of the image pushed the model toward that guess?
    #    layer4[-1] is the last convolutional block of ResNet-50.
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    heat = cam(input_tensor=x, targets=None)[0]  # targets=None means "the top guess"
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)

    # 5. Save the original, the raw heat map, and the overlay side by side.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[1].imshow(heat, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heat map")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Top guess: {label}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
