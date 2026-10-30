"""Grad-CAM for one study, with the real classifier: where does the model look?

Scores every saved slice of the study, takes the most suspicious ones, and draws each
slice next to its Grad-CAM overlay. For a PDAC patient the title also says whether the
slice is a reference tumour slice (from the manual lesion mask), so you can see whether
the model's most suspicious slices are the ones that really show the tumour.

Run from the repo root:
    python src/explainability/gradcam_patient.py 100000_00001 --model models/classifier.pt --crops outputs/crops

Output: outputs/gradcam_<study>.png (the "outputs" folder is ignored by Git)
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image

from gradcam import generate_heatmap
from model_loader import default_device, load_model, to_model_input
from explain import score_slices


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for the most suspicious slices of one study")
    parser.add_argument("study", help="study id, for example 100000_00001")
    parser.add_argument("--model", default="models/classifier.pt")
    parser.add_argument("--crops", default="outputs/crops")
    parser.add_argument("--top", type=int, default=3, help="how many slices to draw")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = default_device()
    model, notes = load_model(args.model, device)
    with np.load(os.path.join(args.crops, f"{args.study}.npz")) as f:
        crops, slice_idx = f["crops"], f["slice_idx"]
        has_lesion, label, split = f["has_lesion"].astype(bool), int(f["label"]), str(f["split"])

    probs, _ = score_slices(model, crops, device)
    score, threshold = float(probs.mean()), float(notes.get("threshold", 0.5))
    print(f"Study {args.study} ({split}, true label {'PDAC' if label else 'non-PDAC'})")
    print(f"Patient score {score:.3f}  threshold {threshold:.2f}  ->  "
          f"{'PDAC' if score >= threshold else 'non-PDAC'}")

    order = np.argsort(-probs)[:args.top]
    fig, axes = plt.subplots(len(order), 3, figsize=(10, 3.4 * len(order)), squeeze=False)
    for row, i in enumerate(order):
        x = to_model_input(crops[i:i + 1]).to(device)
        heat = generate_heatmap(model, x, model.gradcam_layer, target_class=1)
        gray = crops[i].astype(np.float32) / 255.0
        rgb = np.stack([gray] * 3, axis=-1)
        overlay = show_cam_on_image(rgb, heat, use_rgb=True)
        tag = " | reference tumour slice" if has_lesion[i] else ""
        axes[row, 0].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Slice {int(slice_idx[i])}: chance of PDAC {probs[i]:.2f}{tag}", fontsize=8)
        axes[row, 1].imshow(heat, cmap="jet", vmin=0, vmax=1)
        axes[row, 1].set_title("Grad-CAM heat map", fontsize=8)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
        print(f"   slice {int(slice_idx[i])}: {probs[i]:.2f}{tag}")
    fig.suptitle(f"{args.study}: patient score {score:.2f} (threshold {threshold:.2f})", fontsize=10)
    fig.tight_layout()
    out = args.out or f"outputs/gradcam_{args.study}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
