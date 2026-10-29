"""Look at the saved crops of one study, to check the cropping is sensible.

Run from the repo root:
    python src/classification/view_crops.py outputs/crops_test/100002_00001.npz

Output: outputs/crops_view.png. Slices that contain tumour have a RED title.
The pancreas should sit in the middle of each picture.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="View the crops saved for one study")
    parser.add_argument("file", help="a .npz file made by make_crops.py")
    parser.add_argument("--out", default="outputs/crops_view.png")
    parser.add_argument("--max-show", type=int, default=24)
    args = parser.parse_args()

    with np.load(args.file) as f:
        crops, idx, has_lesion = f["crops"], f["slice_idx"], f["has_lesion"]
        label, study = int(f["label"]), str(f["study_id"])
    # Show evenly spread pictures, always including tumour slices when there are any.
    order = np.linspace(0, len(crops) - 1, min(len(crops), args.max_show)).round().astype(int)
    cols = 6
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), order):
        ax.imshow(crops[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"slice {idx[i]}", fontsize=8, color="red" if has_lesion[i] else "black")
    fig.suptitle(f"{study}  (PDAC={label})  red = contains tumour", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
