"""Build classifier crops using segmentation-predicted pancreas masks only."""
import argparse
import csv
import hashlib
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.classification.make_crops import (PATTERNS, crop_slice, evenly_spaced,
                                           find_file)


def prediction_path(root, study_id):
    root = Path(root)
    candidates = [root / study_id / 'pancreas.nii.gz', root / f'{study_id}.nii.gz']
    return next((path for path in candidates if path.exists()), None)


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def existing_matches(path, prediction_sha256, prediction_id, max_pancreas, window_mm,
                     patient_id, study_id, split, label):
    with np.load(path, allow_pickle=False) as item:
        return (str(item.get('crop_source', '')) == 'predicted_pancreas'
                and str(item.get('prediction_sha256', '')) == prediction_sha256
                and str(item.get('prediction_id', '')) == prediction_id
                and int(item.get('max_pancreas', -1)) == max_pancreas
                and float(item.get('window_mm', np.nan)) == window_mm
                and str(item.get('patient_id', '')) == patient_id
                and str(item.get('study_id', '')) == study_id
                and str(item.get('split', '')) == split
                and int(item.get('label', -1)) == label)


def same_grid(image, mask, tolerance=1e-5):
    return (image.GetSize() == mask.GetSize()
            and np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=tolerance, rtol=0)
            and np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=tolerance, rtol=0)
            and np.allclose(image.GetDirection(), mask.GetDirection(), atol=tolerance, rtol=0))


def selected_pancreas_slices(mask, maximum):
    area = mask.sum(axis=(1, 2))
    if not area.any():
        return np.array([], dtype=int)
    clear = np.where(area >= max(50, 0.25 * area.max()))[0]
    return evenly_spaced(clear, maximum)


def read_splits(path):
    result = {}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            result[row['patient_id']] = (row['split'], int(row['label']))
    return result


def build_predicted_crops(studies, predictions, prediction_id, splits, out, only_split='all', limit=None,
                          max_pancreas=24, window_mm=144.0):
    if not prediction_id.strip():
        raise ValueError('prediction_id must identify the segmentation checkpoint or run')
    info = read_splits(splits)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {'saved': 0, 'existing': 0, 'missing_prediction': 0,
              'empty_prediction': 0, 'grid_mismatch': 0, 'missing_image': 0}

    for folder in sorted(path for path in Path(studies).iterdir()
                         if path.is_dir() and not path.name.startswith('.')):
        patient_id = folder.name.split('_')[0]
        if patient_id not in info or not (folder / 'complete.json').exists():
            continue
        split, label = info[patient_id]
        if only_split != 'all' and split != only_split:
            continue
        image_path = find_file(folder, PATTERNS['image'])
        mask_path = prediction_path(predictions, folder.name)
        if image_path is None:
            counts['missing_image'] += 1
            print(f'SKIP {folder.name}: processed image is missing', flush=True)
            continue
        if mask_path is None:
            counts['missing_prediction'] += 1
            print(f'SKIP {folder.name}: predicted pancreas mask is missing', flush=True)
            continue
        target = out / f'{folder.name}.npz'
        prediction_sha256 = file_hash(mask_path)
        if target.exists():
            if not existing_matches(target, prediction_sha256, prediction_id, max_pancreas,
                                    window_mm, patient_id, folder.name, split, label):
                raise ValueError(f'{target} was made from a different prediction or crop configuration; '
                                 'choose a new --out directory')
            counts['existing'] += 1
            continue
        if limit is not None and counts['saved'] >= limit:
            break

        image = sitk.ReadImage(str(image_path))
        mask_image = sitk.ReadImage(str(mask_path))
        if not same_grid(image, mask_image):
            counts['grid_mismatch'] += 1
            print(f'SKIP {folder.name}: predicted mask is not on the CT grid', flush=True)
            continue
        ct = sitk.GetArrayFromImage(image)
        pancreas = sitk.GetArrayFromImage(mask_image) > 0
        slices = selected_pancreas_slices(pancreas, max_pancreas)
        if not len(slices):
            counts['empty_prediction'] += 1
            print(f'SKIP {folder.name}: predicted pancreas mask is empty', flush=True)
            continue

        window_px = int(round(window_mm / float(image.GetSpacing()[0])))
        crops, kept = [], []
        for z in slices:
            crop, _ = crop_slice(ct[z], pancreas[z], window_px)
            crops.append(crop)
            kept.append(z)

        temporary = out / f'{folder.name}.tmp.npz'
        np.savez_compressed(
            temporary,
            crops=np.stack(crops),
            slice_idx=np.asarray(kept, dtype=int),
            has_lesion=np.zeros(len(crops), dtype=bool),
            lesion_available=False,
            label=label,
            patient_id=patient_id,
            split=split,
            study_id=folder.name,
            window_mm=window_mm,
            max_pancreas=max_pancreas,
            crop_source='predicted_pancreas',
            prediction_sha256=prediction_sha256,
            prediction_id=prediction_id,
        )
        os.replace(temporary, target)
        counts['saved'] += 1
        print(f'{folder.name}  {split:5s} PDAC={label}  slices={len(crops)}', flush=True)

    print('Predicted-mask crop summary: ' + ', '.join(f'{key}={value}' for key, value in counts.items()))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('studies', help='processed/full folder containing one directory per study')
    parser.add_argument('--predictions', required=True,
                        help='segmentation output containing <study>/pancreas.nii.gz masks')
    parser.add_argument('--prediction-id', required=True,
                        help='segmentation checkpoint/run identifier recorded in every crop')
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--out', required=True, help='new output directory; do not reuse oracle crops')
    parser.add_argument('--only-split', default='all', choices=['all', 'train', 'val', 'test'])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-pancreas', type=int, default=24)
    parser.add_argument('--window-mm', type=float, default=144.0)
    args = parser.parse_args()
    build_predicted_crops(args.studies, args.predictions, args.prediction_id, args.splits, args.out,
                          args.only_split, args.limit, args.max_pancreas, args.window_mm)


if __name__ == '__main__':
    main()
