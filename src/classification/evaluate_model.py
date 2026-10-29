"""Score a trained classifier on the VALIDATION or TEST patients.

The cut-off is chosen on the VALIDATION patients (for the chosen scoring rule) and then
applied to the other split, so the test patients never influence any choice.
LOOK AT THE TEST PATIENTS ONCE, at the very end. If you keep re-checking test results and
changing things, the test score stops being honest.

Run (for example on Colab, after training):
    python src/classification/evaluate_model.py --model models/classifier.pt --crops outputs/crops --split test --aggregation topk:5

Also writes outputs/<split>_scores.csv: one row per patient (id, true label, model score).
"""
import argparse
import csv
import os

import torch

from dataset import CropDataset
from evaluation import (bootstrap_auc_ci, evaluate_patients, format_metrics,
                        patient_scores, pick_threshold, predict_probs)
from resnet_model import load_classifier


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier")
    parser.add_argument("--model", default="models/classifier.pt")
    parser.add_argument("--crops", default="outputs/crops")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--aggregation", default="mean",
                        help="how slice scores become one patient score (mean, max, topk:5, frac:0.2 ...)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, notes = load_classifier(args.model, device)
    print(f"Model from epoch {notes.get('epoch')}, validation AUC {notes.get('val_auc')}; "
          f"scoring rule: {args.aggregation}")

    # Step 1: choose the cut-off on the VALIDATION patients.
    val = CropDataset(args.crops, "val", mode="eval")
    val_probs, val_patient = predict_probs(model, val, device)
    val_scores, val_labels = patient_scores(val_probs, val_patient, val.patient_labels, args.aggregation)
    threshold = pick_threshold(val_scores, val_labels)
    val_metrics = evaluate_patients(val_probs, val_patient, val.patient_labels, threshold, args.aggregation)
    print(f"\nVALIDATION (cut-off {threshold:.3f} chosen here): {format_metrics(val_metrics)}")
    if args.split == "val":
        dataset, probs, patient = val, val_probs, val_patient
    else:
        # Step 2: apply that cut-off, unchanged, to the other patients.
        dataset = CropDataset(args.crops, args.split, mode="eval")
        probs, patient = predict_probs(model, dataset, device)
        metrics = evaluate_patients(probs, patient, dataset.patient_labels, threshold, args.aggregation)
        print(f"\n{args.split.upper()}: {format_metrics(metrics)}")

    scores, labels = patient_scores(probs, patient, dataset.patient_labels, args.aggregation)
    low, high = bootstrap_auc_ci(scores, labels)
    print(f"{args.split.upper()} AUC 95% range from resampling the patients: {low:.3f} to {high:.3f}")
    print("(a wide range means the set is too small to separate close models)")

    out = f"outputs/{args.split}_scores.csv"
    os.makedirs("outputs", exist_ok=True)
    assert len(scores) == len(dataset.patient_ids)  # every patient has pictures in eval mode
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "label", "score"])
        for pid, label, score in zip(dataset.patient_ids, labels, scores):
            writer.writerow([pid, int(label), f"{score:.4f}"])
    print("Saved", out)


if __name__ == "__main__":
    main()
