"""Try different ways of combining a patient's slice scores, on the VALIDATION patients.

The trained model gives every slice a score. This script tests several rules for turning a
patient's slice scores into ONE score (see evaluation.py) and prints the validation AUC and
the sensitivity/specificity at the best cut-off for each rule. No retraining is needed.

Only the validation patients are used here, so the test patients stay untouched.
(The cut-off is picked on the same patients it is measured on, so sensitivity and
specificity look slightly better than they really are. AUC does not depend on the cut-off.)

Run (Colab is easiest; it needs the GPU once, then saves the slice scores for reuse):
    python src/classification/compare_aggregations.py --model models/classifier.pt --crops outputs/crops
"""
import argparse
import os

import numpy as np

from evaluation import patient_scores, pick_threshold
from sklearn.metrics import roc_auc_score

METHODS = ["mean", "max", "topk:3", "topk:5", "topk:10", "frac:0.1", "frac:0.2", "frac:0.5"]


def main():
    parser = argparse.ArgumentParser(description="Compare ways of combining slice scores")
    parser.add_argument("--model", default="models/classifier.pt")
    parser.add_argument("--crops", default="outputs/crops")
    parser.add_argument("--probs", default="outputs/val_probs.npz",
                        help="slice scores are saved here, and reused if the file exists")
    args = parser.parse_args()

    if os.path.exists(args.probs):
        saved = np.load(args.probs)
        probs, patient, patient_labels = saved["probs"], saved["patient"], saved["patient_labels"]
        print(f"Reusing saved slice scores from {args.probs}")
    else:
        import torch  # only needed when the model has to be run
        from dataset import CropDataset
        from evaluation import predict_probs
        from resnet_model import load_classifier

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, notes = load_classifier(args.model, device)
        dataset = CropDataset(args.crops, "val", mode="eval")
        probs, patient = predict_probs(model, dataset, device)
        patient_labels = dataset.patient_labels
        os.makedirs(os.path.dirname(args.probs) or ".", exist_ok=True)
        np.savez(args.probs, probs=probs, patient=patient, patient_labels=patient_labels)
        print(f"Saved slice scores to {args.probs}")

    print(f"\nValidation patients: {len(patient_labels)} ({int(np.sum(patient_labels))} PDAC)")
    print(f"{'rule':10s} {'AUC':>6s} {'cut-off':>8s} {'sensitivity':>12s} {'specificity':>12s}")
    for method in METHODS:
        scores, labels = patient_scores(probs, patient, patient_labels, method)
        auc = roc_auc_score(labels, scores)
        cut = pick_threshold(scores, labels)
        predicted = scores >= cut
        sens = predicted[labels == 1].mean()
        spec = (~predicted)[labels == 0].mean()
        print(f"{method:10s} {auc:6.3f} {cut:8.3f} {sens:12.2f} {spec:12.2f}")
    print("\nPick ONE rule and keep it. Differences of about 0.01 in AUC are just noise.")


if __name__ == "__main__":
    main()
