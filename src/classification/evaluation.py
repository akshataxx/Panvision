"""Shared scoring, so everyone who trains a classifier reports the SAME numbers.

The label belongs to a patient, not to a slice. So the model's scores for all of one
patient's slices are averaged into ONE score per patient, and everything below is
measured per patient.

How a patient's slice scores are combined ("aggregation"):
  mean       average of all slices (the default; a tumour on a few slices gets diluted)
  max        the single most suspicious slice
  topk:K     average of the K most suspicious slices, for example topk:5
  frac:F     average of the most suspicious fraction F of slices, for example frac:0.2
Choose the rule on the VALIDATION patients, then keep it fixed for the test patients.

Rules for a fair comparison:
  * choose settings (epochs, threshold ...) on the VALIDATION patients only;
  * look at the TEST patients once, at the very end.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def patient_scores(probs, patient_idx, patient_labels, method="mean"):
    """Combine slice scores into one score per patient.

    probs: PDAC probability of every picture;  patient_idx: patient number of every picture;
    patient_labels: label (0/1) of every patient;  method: see the list at the top of this file.
    Returns (score per patient, label per patient).
    """
    probs = np.asarray(probs, dtype=float)
    patient_idx = np.asarray(patient_idx, dtype=int)
    patient_labels = np.asarray(patient_labels, dtype=int)
    scores, labels = [], []
    for p in range(len(patient_labels)):
        s = probs[patient_idx == p]
        if len(s) == 0:
            continue  # ignore any patient who has no pictures
        if method == "mean":
            value = s.mean()
        elif method == "max":
            value = s.max()
        elif method.startswith("topk:"):
            k = max(1, int(method.split(":")[1]))
            value = np.sort(s)[-k:].mean()
        elif method.startswith("frac:"):
            k = max(1, int(np.ceil(float(method.split(":")[1]) * len(s))))
            value = np.sort(s)[-k:].mean()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        scores.append(value)
        labels.append(patient_labels[p])
    return np.array(scores), np.array(labels)


def pick_threshold(scores, labels):
    """The cut-off that best separates the classes (largest sensitivity + specificity - 1).

    Choose it on the validation patients, then keep it fixed for the test patients.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best = int(np.argmax(tpr - fpr))
    return float(thresholds[best])


def evaluate_patients(probs, patient_idx, patient_labels, threshold=0.5, aggregation="mean"):
    """Per-patient results: AUC, sensitivity, specificity, accuracy.

    AUC: 0.5 = coin flip, 1.0 = perfect. Sensitivity = share of PDAC patients caught.
    Specificity = share of non-PDAC patients correctly cleared.
    """
    scores, y = patient_scores(probs, patient_idx, patient_labels, aggregation)
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


def bootstrap_auc_ci(scores, labels, n_resamples=1000, seed=0):
    """95% range for the AUC, by re-drawing the patients with replacement many times.

    With few patients the AUC wobbles depending on which patients happen to be in the set.
    If two models' ranges overlap a lot, we cannot say one is really better.
    Returns (low, high).
    """
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(labels), len(labels))
        if labels[pick].min() == labels[pick].max():
            continue  # this draw happened to contain only one class; AUC is undefined
        aucs.append(roc_auc_score(labels[pick], scores[pick]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


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
