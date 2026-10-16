"""Compute one fingerprint (512 numbers) per study and save them to a file.

For every study folder inside the samples folder:
  1. load the CT and the pancreas mask
  2. take up to --max-slices slices that clearly contain pancreas
  3. crop each slice around the pancreas, resize to 224 x 224 (same as crop_gradcam_scan.py)
  4. run the model and average the slices' fingerprints -> one fingerprint per study

Uses the STAND-IN model for now, so the fingerprints are NOT medically meaningful.
The aim is to prove the plumbing works; swap in the real classifier later.

Run from the repo root:
    python src/explainability/build_embeddings.py "C:\path\to\panvision_samples"

Output: outputs/embeddings.npz (the "outputs" folder is ignored by Git)
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# These live in the same folder as this script.
from crop_gradcam_scan import MEAN, STD, square_crop_box
from standin_model import StandInClassifier
from view_scan import PATTERNS, find_file, load_volume


def slice_to_tensor(ct2d, mask2d, margin):
    """Crop one slice around the pancreas -> a (3, 224, 224) model input."""
    r0, r1, c0, c1 = square_crop_box(mask2d, margin)
    crop = ct2d[r0:r1, c0:c1]
    small = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8)).resize((224, 224))
    gray = np.asarray(small, dtype=np.float32) / 255.0
    rgb = np.stack([gray, gray, gray], axis=-1)
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)).float()


def choose_slices(pancreas, max_slices):
    """Pick slices where the pancreas is clearly visible, spread evenly through the scan."""
    area = pancreas.sum(axis=(1, 2))  # pancreas pixels on each slice
    good = np.where(area >= max(50, 0.25 * area.max()))[0]  # skip slices with only a sliver
    if len(good) > max_slices:
        good = good[np.linspace(0, len(good) - 1, max_slices).round().astype(int)]
    return good


def main():
    parser = argparse.ArgumentParser(description="Build one embedding per study")
    parser.add_argument("samples", help="folder holding one sub-folder per study")
    parser.add_argument("--splits", default="data/splits.csv")
    parser.add_argument("--out", default="outputs/embeddings.npz")
    parser.add_argument("--max-slices", type=int, default=16)
    parser.add_argument("--margin", type=int, default=16)
    args = parser.parse_args()

    # patient_id -> (split, label) from the team's shared split file
    info = {}
    with open(args.splits, newline="") as f:
        for row in csv.DictReader(f):
            info[row["patient_id"]] = (row["split"], int(row["label"]))

    model = StandInClassifier()
    study_ids, patient_ids, splits, labels, vectors = [], [], [], [], []

    for folder in sorted(p for p in Path(args.samples).iterdir() if p.is_dir()):
        patient_id = folder.name.split("_")[0]
        image_path = find_file(folder, PATTERNS["image"])
        pancreas_path = find_file(folder, PATTERNS["pancreas"])
        if patient_id not in info or image_path is None or pancreas_path is None:
            print(f"SKIP {folder.name}: missing from splits.csv or no image/pancreas file")
            continue
        ct, _ = load_volume(image_path)
        pancreas = load_volume(pancreas_path)[0] > 0
        if pancreas.shape != ct.shape or not pancreas.any():
            print(f"SKIP {folder.name}: pancreas mask missing, empty or wrong shape")
            continue

        slices = choose_slices(pancreas, args.max_slices)
        batch = torch.stack([slice_to_tensor(ct[z], pancreas[z], args.margin) for z in slices])
        with torch.no_grad():
            study_vector = model.embedding(batch).mean(dim=0).numpy()  # average of the slices

        split, label = info[patient_id]
        study_ids.append(folder.name)
        patient_ids.append(patient_id)
        splits.append(split)
        labels.append(label)
        vectors.append(study_vector)
        print(f"{folder.name}  split={split}  PDAC={label}  slices used={len(slices)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        study_ids=np.array(study_ids),
        patient_ids=np.array(patient_ids),
        splits=np.array(splits),
        labels=np.array(labels),
        embeddings=np.stack(vectors),
    )
    print(f"\nSaved {len(study_ids)} study fingerprints to {args.out}")


if __name__ == "__main__":
    main()
