"""Crop the pancreas from one real CT slice and run Grad-CAM on it.

Uses the STAND-IN model (see standin_model.py), so the heat map is NOT meaningful
medically. The point is to prove the whole chain works on real PanVision data:
CT slice -> pancreas crop -> model input -> Grad-CAM heat map -> overlay.

Run from the repo root:
    python src/explainability/crop_gradcam_scan.py "C:\\panvision_samples\\100000_00001"

Output: outputs/crop_gradcam.png (the "outputs" folder is ignored by Git)
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# These two live in the same folder as this script.
from standin_model import StandInClassifier
from view_scan import PATTERNS, find_file, load_volume

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # standard photo-model values
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def square_crop_box(mask2d, margin=16):
    """Return (row0, row1, col0, col1): a square box around the mask plus a margin."""
    rows, cols = np.where(mask2d)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    side = max(r1 - r0, c1 - c0) + 2 * margin
    side = min(side, *mask2d.shape)  # never bigger than the image
    r_mid, c_mid = (r0 + r1) // 2, (c0 + c1) // 2
    top = int(np.clip(r_mid - side // 2, 0, mask2d.shape[0] - side))
    left = int(np.clip(c_mid - side // 2, 0, mask2d.shape[1] - side))
    return top, top + side, left, left + side


def main():
    parser = argparse.ArgumentParser(description="Pancreas crop + Grad-CAM on one slice")
    parser.add_argument("folder", help="study folder containing the .nii.gz files")
    parser.add_argument("--slice", type=int, default=None, help="slice number to use")
    parser.add_argument("--margin", type=int, default=16, help="extra pixels around pancreas")
    parser.add_argument("--out", default="outputs/crop_gradcam.png")
    args = parser.parse_args()

    # 1. Load the CT and pancreas mask (same helpers as view_scan.py).
    image_path = find_file(args.folder, PATTERNS["image"])
    pancreas_path = find_file(args.folder, PATTERNS["pancreas"])
    if image_path is None or pancreas_path is None:
        raise SystemExit("Need both an image file and a pancreas file in that folder.")
    ct, _ = load_volume(image_path)
    pancreas, _ = load_volume(pancreas_path)
    pancreas = pancreas > 0
    if pancreas.shape != ct.shape:
        raise SystemExit(f"Shape mismatch: CT {ct.shape} vs pancreas {pancreas.shape}")

    # 2. Pick the slice with the most pancreas, unless one was requested.
    z = args.slice if args.slice is not None else int(np.argmax(pancreas.sum(axis=(1, 2))))
    if not pancreas[z].any():
        raise SystemExit(f"Slice {z} has no pancreas in the mask. Try another --slice.")
    print(f"Using slice {z} of {ct.shape[0]}")

    # 3. Crop a square around the pancreas and resize to the 224 x 224 the model expects.
    r0, r1, c0, c1 = square_crop_box(pancreas[z], args.margin)
    crop = ct[z, r0:r1, c0:c1]
    print(f"Crop box rows {r0}-{r1}, cols {c0}-{c1}  ({r1 - r0} x {c1 - c0} pixels)")
    small = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8)).resize((224, 224))
    gray = np.asarray(small, dtype=np.float32) / 255.0  # 224 x 224, values 0-1
    rgb = np.stack([gray, gray, gray], axis=-1)  # models expect 3 colour channels
    x = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).float()

    # 4. Run the stand-in model.
    model = StandInClassifier()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        emb = model.embedding(x)[0]
    print(f"Stand-in scores  non-PDAC {probs[0]:.2f}  PDAC {probs[1]:.2f}  (random head: meaningless)")
    print(f"Embedding (the fingerprint) has {emb.numel()} numbers")

    # 5. Grad-CAM for the PDAC output (class 1).
    cam = GradCAM(model=model, target_layers=[model.gradcam_layer])
    heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)

    # 6. Draw four panels: whole slice with crop box, the crop, heat map, overlay.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(ct[z], cmap="gray", vmin=0, vmax=1)
    axes[0].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor="yellow", lw=2))
    axes[0].set_title(f"Slice {z} with crop box")
    axes[1].imshow(rgb)
    axes[1].set_title("Pancreas crop (model input)")
    axes[2].imshow(heat, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM heat map")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("STAND-IN model: heat map is not medically meaningful", color="red")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
