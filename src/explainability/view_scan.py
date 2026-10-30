"""View one processed CT scan with its pancreas and lesion masks.

Purpose: look at a real PanVision study before building anything on it. This only
READS the files; it never changes them.

Run from the repo root (the folder that contains README.md):
    python src/explainability/view_scan.py "C:\\panvision_data\\samples\\100000_00001"

Give it the study folder (or the folder holding the .nii.gz files). It searches
inside for image / pancreas / lesion files.

Output: outputs/scan_view.png (the "outputs" folder is ignored by Git)
"""
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap

# File-name patterns to look for. The first pattern that matches wins.
PATTERNS = {
    "image": ["image*.nii*", "images*.nii*"],
    "pancreas": ["pancreas*.nii*"],
    "lesion": ["lesion*.nii*", "lesions*.nii*"],
}


def find_file(folder, patterns):
    """Return the first file under `folder` that matches one of the patterns."""
    for pattern in patterns:
        hits = sorted(Path(folder).rglob(pattern))
        if hits:
            return hits[0]
    return None


def load_volume(path):
    """Read a NIfTI file. Returns (array, spacing). The array order is (Z, Y, X)."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img.GetSpacing()


def main():
    parser = argparse.ArgumentParser(description="View one processed CT study")
    parser.add_argument("folder", help="study folder containing the .nii.gz files")
    parser.add_argument("--slice", type=int, default=None, help="slice number to show")
    parser.add_argument("--out", default="outputs/scan_view.png")
    args = parser.parse_args()

    # 1. Find the files.
    found = {name: find_file(args.folder, pats) for name, pats in PATTERNS.items()}
    print("Files found:")
    for name, path in found.items():
        print(f"  {name:9s} -> {path}")
    if found["image"] is None:
        print("\nNo CT image file found. Files in that folder:")
        for p in sorted(Path(args.folder).rglob("*")):
            if p.is_file():
                print("  ", p)
        raise SystemExit(1)

    # 2. Load the CT. Each slice is one 2D picture; the first number is the slice index.
    ct, spacing = load_volume(found["image"])
    print(f"\nCT shape (slices, rows, cols): {ct.shape}")
    print(f"Voxel spacing (x, y, z) in mm: {tuple(round(s, 2) for s in spacing)}")
    print(f"CT value range: {ct.min():.3f} to {ct.max():.3f}")

    # 3. Load the masks, if they exist and line up with the CT.
    masks = {}
    for name in ("pancreas", "lesion"):
        if found[name] is None:
            print(f"No {name} mask file found, skipping it.")
            continue
        arr, _ = load_volume(found[name])
        if arr.shape != ct.shape:
            print(f"WARNING: {name} mask shape {arr.shape} != CT shape {ct.shape}; skipping.")
            continue
        masks[name] = arr > 0
        print(f"{name} mask: {int(masks[name].sum())} voxels")

    # 4. Choose which slice to show: the one with the most lesion (or pancreas).
    if args.slice is not None:
        z = int(np.clip(args.slice, 0, ct.shape[0] - 1))
    else:
        z = ct.shape[0] // 2  # fall back to the middle slice
        for name in ("lesion", "pancreas"):
            if name in masks and masks[name].any():
                z = int(np.argmax(masks[name].sum(axis=(1, 2))))
                break
    print(f"Showing slice {z} of {ct.shape[0]}")

    # 5. Draw: plain CT on the left, CT with mask colours on the right.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ct[z], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"CT slice {z}")
    axes[1].imshow(ct[z], cmap="gray", vmin=0, vmax=1)
    legend = []
    for name, colour in (("pancreas", "limegreen"), ("lesion", "red")):
        if name in masks:
            layer = np.ma.masked_where(~masks[name][z], masks[name][z])
            axes[1].imshow(layer, cmap=ListedColormap([colour]), alpha=0.5)
            legend.append(f"{name} (green)" if name == "pancreas" else f"{name} (red)")
    axes[1].set_title("With masks: " + (", ".join(legend) if legend else "none"))
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
