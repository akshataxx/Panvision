"""Does the model find the tumour slices? A simple, honest check of the explanations.

For PDAC patients in a split, take the 3 slices the model finds most suspicious and count
how many are reference tumour slices (slices that contain tumour in the manual lesion
mask). Compare with chance: the share of a patient's saved slices that are tumour slices.
If the model's top slices are tumour slices much more often than chance, its
suspicion is pointing at the right place, not at something else in the scan.

Run from the repo root (validation while developing; test only once, at the end):
    python src/explainability/slice_localisation.py --model models/classifier.pt --crops outputs/crops --split val
"""
import argparse
from pathlib import Path

import numpy as np

from model_loader import default_device, load_model
from explain import score_slices


def main():
    parser = argparse.ArgumentParser(description="Are the most suspicious slices the tumour slices?")
    parser.add_argument("--model", default="models/classifier.pt")
    parser.add_argument("--crops", default="outputs/crops")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    device = default_device()
    model, _ = load_model(args.model, device)
    share, chance, any_hit, n = [], [], [], 0
    for path in sorted(Path(args.crops).glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as f:
            if str(f["split"]) != args.split or int(f["label"]) != 1:
                continue
            has_lesion = f["has_lesion"].astype(bool)
            if not has_lesion.any() or has_lesion.all():
                continue  # no reference tumour slices, or no healthy-looking slices to confuse with
            probs, _ = score_slices(model, f["crops"], device)
        top = np.argsort(-probs)[:args.top]
        share.append(has_lesion[top].mean())
        any_hit.append(float(has_lesion[top].any()))
        chance.append(has_lesion.mean())
        n += 1

    print(f"{n} PDAC studies in {args.split} with reference tumour slices")
    print(f"Share of the top-{args.top} slices that are tumour slices: {np.mean(share):.2f}  "
          f"(chance {np.mean(chance):.2f})")
    print(f"At least one of the top-{args.top} is a tumour slice: {np.mean(any_hit):.0%} of studies")
    print("The first number should be clearly higher than chance.")


if __name__ == "__main__":
    main()
