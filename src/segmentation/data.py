"""Study discovery, volume loading and in-memory 2.5D slice datasets for the segmentation cascade."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from src.preprocessing.storage import data_root
from src.segmentation.roi import ROI, crop, make_roi

# Lesion (1) and pancreatic duct (5) lie inside the organ, so the pancreas target is the whole gland.
# The common bile duct (6) is excluded.
PANCREAS_VALUES = (1, 4, 5)
LESION_VALUE = 1


def find_studies(root=None, splits_path='data/splits.csv'):
    """All completed studies under <root>/processed/full, with absolute paths, checked against the shared split."""
    root = Path(root) if root else data_root()
    frames = []
    for manifest in sorted((root / 'processed/full').glob('*/volumes/manifest.csv')):
        if not (manifest.parents[1] / 'complete.json').exists():
            continue
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        for column in ('processed_image', 'processed_labels'):
            frame[column] = frame[column].map(lambda value: str(manifest.parent / value) if value else '')
        frames.append(frame)
    if not frames:
        raise ValueError(f'No completed studies under {root / "processed/full"}')
    studies = pd.concat(frames, ignore_index=True)
    if studies.study_id.duplicated().any():
        raise ValueError('Duplicate study_id across manifests; do not mix processing runs')
    splits = pd.read_csv(splits_path, dtype=str, keep_default_na=False)
    merged = studies.merge(splits[['patient_id', 'split']].rename(columns={'split': 'shared_split'}),
                           on='patient_id', how='left')
    if merged.shared_split.isna().any():
        raise ValueError(f'Patients missing from {splits_path}: {merged[merged.shared_split.isna()].patient_id.tolist()}')
    if (merged.split != merged.shared_split).any():
        raise ValueError('Manifest split disagrees with the shared split file')
    if merged.groupby('patient_id').split.nunique().gt(1).any():
        raise ValueError('A patient appears in more than one split')
    return merged.drop(columns='shared_split')


def pancreas_rows(studies, split):
    return studies[(studies.split == split) & (studies.processed_labels != '')]


def lesion_rows(studies, split):
    """PDAC studies with manual lesion masks (positives) and non-PDAC studies with labels (negatives)."""
    rows = pancreas_rows(studies, split)
    return (rows[(rows.label == '1') & (rows.annotation_tier == 'manual')],
            rows[rows.label == '0'])


def subsample(rows, limit, seed=42):
    return rows if not limit or len(rows) <= limit else rows.sample(n=limit, random_state=seed)


def load_case(row):
    image = sitk.ReadImage(row.processed_image, sitk.sitkFloat32)
    case = {'study_id': row.study_id, 'patient_id': row.patient_id, 'label': row.label,
            'spacing': image.GetSpacing(), 'image': sitk.GetArrayFromImage(image)}
    if row.processed_labels:
        case['labels'] = sitk.GetArrayFromImage(sitk.ReadImage(row.processed_labels)).astype(np.uint8)
    return case


def load_cases(rows, workers=8, transform=None):
    """Load and transform all rows in parallel; use the transform to shrink each case before the next arrives."""
    transform = transform or (lambda case: case)
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda row: transform(load_case(row)), rows.itertuples(index=False)))


def iter_cases(rows):
    """Load full cases one at a time (with one-ahead prefetch) so evaluation memory stays flat."""
    rows = list(rows.itertuples(index=False))
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(load_case, rows[0]) if rows else None
        for index in range(len(rows)):
            case = pending.result()
            pending = pool.submit(load_case, rows[index + 1]) if index + 1 < len(rows) else None
            yield case


def pancreas_target(labels):
    return np.isin(labels, PANCREAS_VALUES)


def context_stack(volume, z, context):
    """(context, Y, X) stack of neighbouring slices around z, clamped at the volume edges."""
    half = context // 2
    index = np.clip(np.arange(z - half, z + half + 1), 0, volume.shape[0] - 1)
    return volume[index]


def _patch(array, cy, cx, size):
    """Fixed-size in-plane patch of a (..., Y, X) array centred on (cy, cx), zero padded."""
    out = np.zeros(array.shape[:-2] + (size, size), dtype=array.dtype)
    y0, x0 = cy - size // 2, cx - size // 2
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(array.shape[-2], y0 + size), min(array.shape[-1], x0 + size)
    out[..., sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = array[..., sy0:sy1, sx0:sx1]
    return out


def augment(image, target, rng):
    """Light intensity jitter and small in-plane rotation/scale (no flips: abdominal anatomy is not symmetric)."""
    image = image * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)
    if rng.random() < 0.3:
        from scipy import ndimage
        angle, zoom = rng.uniform(-15, 15), rng.uniform(0.9, 1.1)
        matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                           [np.sin(np.radians(angle)), np.cos(np.radians(angle))]]) / zoom
        center = np.array(target.shape) / 2
        offset = center - matrix @ center
        image = np.stack([ndimage.affine_transform(c, matrix, offset, order=1) for c in image])
        target = ndimage.affine_transform(target, matrix, offset, order=0)
    return image.astype(np.float32), target


class _RandomSlices(Dataset):
    """Map-style dataset whose items are random draws; length fixes the number of draws per epoch."""

    def __init__(self, length, train=True):
        self.length, self.train = length, train

    def __len__(self):
        return self.length

    def rng(self, index):
        return np.random.default_rng([torch.initial_seed() % 2 ** 32, index])

    def finish(self, image, target, rng):
        if self.train:
            image, target = augment(image, target, rng)
        return torch.from_numpy(image.astype(np.float32)), torch.from_numpy(target[None].astype(np.float32))


def slice_band(case, target, z_margin=16):
    """Training copy of a case reduced to a z-band around the reference gland (still includes slices
    without pancreas), with `target` as the training mask."""
    zs = np.flatnonzero(pancreas_target(case['labels']).any(axis=(1, 2)))
    if len(zs) == 0:
        return None
    z0, z1 = max(0, zs[0] - z_margin), min(target.shape[0], zs[-1] + 1 + z_margin)
    target = target[z0:z1].astype(np.uint8)
    return {'study_id': case['study_id'], 'image': case['image'][z0:z1].astype(np.float16),
            'target': target, 'positive': np.flatnonzero(target.any(axis=(1, 2)))}


def pancreas_band(case, z_margin=16):
    return slice_band(case, pancreas_target(case['labels']), z_margin)


def lesion_band(case, z_margin=16):
    """Full-slice lesion target for the no-ROI lesion baseline (L0)."""
    return slice_band(case, case['labels'] == LESION_VALUE, z_margin)


class FullSlices(_RandomSlices):
    """Full-slice 2.5D patches (pancreas stage, and the no-ROI lesion baseline); most centred on foreground."""

    def __init__(self, bands, length, patch=192, context=1, positive_fraction=0.7, train=True):
        super().__init__(length, train)
        self.patch, self.context, self.positive_fraction = patch, context, positive_fraction
        self.cases = [band for band in bands if band is not None]
        if not any(len(case['positive']) for case in self.cases):
            raise ValueError('No training case contains a foreground annotation')

    def __getitem__(self, index):
        rng = self.rng(index)
        case = self.cases[rng.integers(len(self.cases))]
        image, target = case['image'], case['target']
        if len(case['positive']) and rng.random() < self.positive_fraction:
            z = int(rng.choice(case['positive']))
            ys, xs = np.nonzero(target[z])
            pick = rng.integers(len(ys))
            cy, cx = int(ys[pick]), int(xs[pick])
        else:
            z = int(rng.integers(image.shape[0]))
            cy, cx = int(rng.integers(image.shape[1])), int(rng.integers(image.shape[2]))
        stack = context_stack(image, z, self.context).astype(np.float32)
        return self.finish(_patch(stack, cy, cx, self.patch), _patch(target[z], cy, cx, self.patch), rng)


def lesion_crop(case, size=160, jitter=16, margin_mm=10.0):
    """ROI crop around the reference pancreas, enlarged by `jitter` on each side for random shifts."""
    roi = make_roi(pancreas_target(case['labels']), case['spacing'], size + 2 * jitter, margin_mm)
    if roi is None:
        return None
    return {'study_id': case['study_id'], 'roi': roi,
            'image': crop(case['image'], roi).astype(np.float16),
            'target': crop((case['labels'] == LESION_VALUE).astype(np.uint8), roi)}


class LesionSlices(_RandomSlices):
    """Stage 2 training samples: 2.5D slices from pancreas ROIs, mixing lesion and lesion-free slices."""

    def __init__(self, positives, negatives, length, size=160, jitter=16, context=3,
                 negative_fraction=0.25, lesion_fraction=0.7, train=True):
        super().__init__(length, train)
        self.size, self.jitter, self.context = size, jitter, context
        self.negative_fraction, self.lesion_fraction = negative_fraction, lesion_fraction
        self.positives = [c for c in positives if c is not None and c['target'].any()]
        self.negatives = [c for c in negatives if c is not None]
        if not self.positives:
            raise ValueError('No positive lesion case available for training')
        for case in self.positives:
            case['lesion_slices'] = np.flatnonzero(case['target'].any(axis=(1, 2)))

    def __getitem__(self, index):
        rng = self.rng(index)
        use_negative = self.negatives and rng.random() < self.negative_fraction
        pool = self.negatives if use_negative else self.positives
        case = pool[rng.integers(len(pool))]
        if not use_negative and rng.random() < self.lesion_fraction:
            z = int(rng.choice(case['lesion_slices']))
        else:
            z = int(rng.integers(case['image'].shape[0]))
        full = case['image'].shape[1]
        shift = rng.integers(-self.jitter, self.jitter + 1, size=2) if self.train else (0, 0)
        cy, cx = full // 2 + int(shift[0]), full // 2 + int(shift[1])
        stack = context_stack(case['image'], z, self.context).astype(np.float32)
        return self.finish(_patch(stack, cy, cx, self.size), _patch(case['target'][z], cy, cx, self.size), rng)


PancreasSlices = FullSlices

__all__ = ['ROI', 'find_studies', 'pancreas_rows', 'lesion_rows', 'subsample', 'load_case', 'load_cases',
           'iter_cases', 'pancreas_target', 'pancreas_band', 'lesion_band', 'context_stack', 'FullSlices',
           'PancreasSlices', 'LesionSlices', 'lesion_crop']
