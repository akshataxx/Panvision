"""Tiny synthetic PanVision data root in the processed/full layout written by src/preprocessing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

SPACING = (1.5, 1.5, 3.0)
SHAPE = (20, 64, 64)  # zyx


def volumes(lesion, seed):
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(SHAPE)
    pancreas = ((z - 10) / 5) ** 2 + ((y - 32) / 6) ** 2 + ((x - 32) / 18) ** 2 <= 1
    labels = np.where(pancreas, 4, 0).astype(np.uint8)
    image = rng.normal(0.2, 0.02, SHAPE).astype(np.float32)
    image[pancreas] = 0.5
    if lesion:
        blob = ((z - 10) / 2) ** 2 + ((y - 32) / 3) ** 2 + ((x - 22) / 3) ** 2 <= 1
        labels[blob] = 1
        image[blob] = 0.35
    return image, labels


def write_study(root, study_id, patient_id, split, label, tier):
    folder = root / 'processed/full' / study_id
    volumes_dir = folder / 'volumes'
    destination = volumes_dir / split / study_id
    destination.mkdir(parents=True)
    image, labels = volumes(label == '1', int(patient_id))
    for name, array in (('image', image), ('labels', labels)):
        item = sitk.GetImageFromArray(array)
        item.SetSpacing(SPACING)
        sitk.WriteImage(item, str(destination / f'{name}.nii.gz'))
    pd.DataFrame([{'study_id': study_id, 'patient_id': patient_id, 'label': label,
                   'annotation_tier': tier, 'split': split,
                   'processed_image': f'{split}/{study_id}/image.nii.gz',
                   'processed_labels': f'{split}/{study_id}/labels.nii.gz'}]).to_csv(
        volumes_dir / 'manifest.csv', index=False)
    (folder / 'complete.json').write_text(json.dumps({'study_id': study_id}))


def make_root(tmp):
    """Two train PDAC (manual), one train PDAC (automatic), one train non-PDAC, one val of each label."""
    root = Path(tmp) / 'PanVisionData'
    studies = [('100_1', '100', 'train', '1', 'manual'), ('101_1', '101', 'train', '1', 'manual'),
               ('102_1', '102', 'train', '1', 'automatic'), ('103_1', '103', 'train', '0', 'automatic'),
               ('104_1', '104', 'val', '1', 'manual'), ('105_1', '105', 'val', '0', 'automatic'),
               ('106_1', '106', 'test', '1', 'manual')]
    for study in studies:
        write_study(root, *study)
    splits = Path(tmp) / 'splits.csv'
    pd.DataFrame([{'patient_id': p, 'split': s, 'label': l} for _, p, s, l, _ in studies]).to_csv(splits, index=False)
    return root, splits
