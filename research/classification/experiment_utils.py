"""Validation and scoring for crop-based experiments."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)

METHODS = ['mean', 'max', 'topk:3', 'topk:5', 'topk:10', 'frac:0.1', 'frac:0.2', 'frac:0.5']


def file_hash(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def audit_crops(folder, splits):
    table = pd.read_csv(splits, dtype=str, keep_default_na=False)
    if not {'patient_id', 'split', 'label'}.issubset(table.columns) or table.patient_id.duplicated().any():
        raise ValueError('Invalid shared split table')
    table = table.set_index('patient_id')
    records, studies, windows = [], set(), set()
    for path in sorted(Path(folder).glob('*.npz')):
        if path.name.endswith('.tmp.npz'):
            raise ValueError(f'Incomplete crop file: {path.name}')
        with np.load(path, allow_pickle=False) as item:
            patient, study, split = (str(item[k].item()) for k in ('patient_id', 'study_id', 'split'))
            label = int(item['label'].item())
            if patient not in table.index or split not in ('train', 'val', 'test'):
                raise ValueError(f'Unknown patient/split: {study}')
            if table.loc[patient, 'split'] != split or table.loc[patient, 'label'] != str(label):
                raise ValueError(f'Split/label mismatch: {study}')
            if study in studies or study != path.stem or study.split('_')[0] != patient:
                raise ValueError(f'Duplicate or inconsistent study identity: {study}')
            studies.add(study)
            crops, lesion, slices = item['crops'], item['has_lesion'], item['slice_idx']
            if crops.dtype != np.uint8 or crops.ndim != 3 or crops.shape[1:] != (224, 224) or not len(crops):
                raise ValueError(f'Invalid crops: {study}')
            if lesion.dtype != np.bool_ or lesion.shape != (len(crops),) or slices.shape != (len(crops),):
                raise ValueError(f'Invalid slice metadata: {study}')
            if len(set(slices.tolist())) != len(slices):
                raise ValueError(f'Duplicate slices: {study}')
            window = float(item['window_mm'].item())
            if not np.isfinite(window) or window <= 0:
                raise ValueError(f'Invalid crop window: {study}')
            windows.add(window)
            records.append(dict(patient_id=patient, study_id=study, split=split, label=label, slices=len(crops)))
    if not records or len(windows) != 1:
        raise ValueError('Empty crop directory or mixed window settings')
    return pd.DataFrame(records), windows.pop()


def threshold_on_validation(scores, labels):
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Both classes are required')
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = np.flatnonzero(np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1))
    return float(thresholds[valid[np.argmax((tpr - fpr)[valid])]])


def score_metrics(scores, labels, threshold):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid patient predictions')
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Both classes are required')
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return dict(n_patients=len(labels), n_pdac=int(labels.sum()),
                auc=float(roc_auc_score(labels, scores)),
                average_precision=float(average_precision_score(labels, scores)),
                sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
                accuracy=float((tp+tn)/len(labels)), f1=float(f1_score(labels, predicted, zero_division=0)),
                threshold=float(threshold), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
