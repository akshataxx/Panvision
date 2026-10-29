"""Compute one fingerprint (512 numbers) per study from the shared crops, using the real classifier.

For every study in every split:
  1. read its saved crops (made by src/classification/make_crops.py; same pictures the
     classifier was trained and scored on)
  2. run the classifier's 512-number layer on every picture
  3. combine the pictures' fingerprints -> one fingerprint per study (each scaled to
     length 1, averaged, scaled to length 1 again: same rule as src/classification/export_embeddings.py)

Output: outputs/embeddings.npz with study_ids, patient_ids, splits, labels, embeddings.
retrieval_eval.py later keeps TRAINING patients only as the search database (for evaluation).

Run from the repo root (a GPU, e.g. Colab, makes it much faster; it runs every picture once):
    python src/explainability/build_embeddings.py --model models/classifier.pt --crops outputs/crops
"""
import argparse
import os

import numpy as np
import torch

from model_loader import default_device, load_model, to_model_input
from retrieval_eval import study_fingerprint
from src.classification.dataset import load_cache


def embed_images(model, images, device, batch_size=64):
    """Fingerprints for an array of saved crops, in order. Returns (N, 512)."""
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = to_model_input(images[start:start + batch_size]).to(device)
            out.append(model.embedding(x).cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description="Build one embedding per study")
    parser.add_argument("--model", default="models/classifier.pt")
    parser.add_argument("--crops", default="outputs/crops")
    parser.add_argument("--out", default="outputs/embeddings.npz")
    args = parser.parse_args()

    device = default_device()
    model, notes = load_model(args.model, device)
    print(f"Model from epoch {notes.get('epoch')} (validation AUC {notes.get('val_auc')}), running on {device}")

    study_ids, patient_ids, splits, labels, vectors = [], [], [], [], []
    for split in ["train", "val", "test"]:
        data = load_cache(args.crops, split, mode="eval")  # every saved slice of every patient
        print(f"{split}: {len(data['images'])} pictures from {len(data['patient_ids'])} patients")
        emb = embed_images(model, data["images"], device)
        picture_study = data["study_ids"]
        for study in dict.fromkeys(picture_study.tolist()):  # keeps the original order
            rows = np.where(picture_study == study)[0]
            study_ids.append(study)
            patient_ids.append(str(data["patient_ids"][data["patient_idx"][rows[0]]]))
            splits.append(split)
            labels.append(int(data["labels"][rows[0]]))
            vectors.append(study_fingerprint(emb[rows]))  # same rule as the database in export_embeddings.py

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