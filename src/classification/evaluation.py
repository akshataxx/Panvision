"""Shared scoring, so everyone who trains a classifier reports the SAME numbers.

The label belongs to a patient, not to a slice. So the model's scores for all of one
patient's slices are averaged into ONE score per patient, and everything below is
measured per patient.

Rules for a fair comparison:
  * choose settings (epochs, threshold ...) on the VALIDATION patients only;
  * look at the TEST patients once, at the very end.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def patient_scores(probs, patient_idx, patient_labels):
    """Average slice scores into one score per patient.

    probs: PDAC probability of every picture;  patient_idx: patient number of every picture;
    patient_labels: label (0/1) of every patient. Returns (score per patient, label per patient).
    """
    probs = np.asarray(probs, dtype=float)
    patient_idx = np.asarray(patient_idx, dtype=int)
    patient_labels = np.asarray(patient_labels, dtype=int)
    n = len(patient_labels)
    sums = np.bincount(patient_idx, weights=probs, minlength=n)
    counts = np.bincount(patient_idx, minlength=n)
    seen = counts > 0  # ignore any patient who has no pictures
    return sums[seen] / counts[seen], patient_labels[seen]


def pick_threshold(scores, labels):
    """The cut-off that best separates the classes (largest sensitivity + specificity - 1).

    Choose it on the validation patients, then keep it fixed for the test patients.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best = int(np.argmax(tpr - fpr))
    return float(thresholds[best])


def evaluate_patients(probs, patient_idx, patient_labels, threshold=0.5):
    """Per-patient results: AUC, sensitivity, specificity, accuracy.

    AUC: 0.5 = coin flip, 1.0 = perfect. Sensitivity = share of PDAC patients caught.
    Specificity = share of non-PDAC patients correctly cleared.
    """
    scores, y = patient_scores(probs, patient_idx, patient_labels)
    predicted = scores >= threshold
    positives, negatives = y == 1, y == 0
    return {
        "n_patients": int(len(y)),
        "n_pdac": int(positives.sum()),
        "auc": float(roc_auc_score(y, scores)) if positives.any() and negatives.any() else float("nan"),
        "sensitivity": float(predicted[positives].mean()) if positives.any() else float("nan"),
        "specificity": float((~predicted)[negatives].mean()) if negatives.any() else float("nan"),
        "accuracy": float((predicted == (y == 1)).mean()),
        "threshold": float(threshold),
    }


def format_metrics(m):
    return (f"AUC {m['auc']:.3f} | sensitivity {m['sensitivity']:.2f} | "
            f"specificity {m['specificity']:.2f} | accuracy {m['accuracy']:.2f} "
            f"(threshold {m['threshold']:.2f}, {m['n_patients']} patients, {m['n_pdac']} PDAC)")


def predict_probs(model, dataset, device, batch_size=64, workers=0):
    """Run the model over a CropDataset. Returns (PDAC probability, patient number) per picture."""
    import torch  # imported here so the scoring functions above work without torch

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model.eval()
    probs, patients = [], []
    with torch.no_grad():
        for x, _, patient in loader:
            scores = torch.softmax(model(x.to(device)), dim=1)[:, 1]  # chance of PDAC
            probs.append(scores.cpu().numpy())
            patients.append(patient.numpy())
    return np.concatenate(probs), np.concatenate(patients)
