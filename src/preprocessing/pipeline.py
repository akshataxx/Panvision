"""Patient-safe PANORAMA preparation. Run from the repository root."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

from src.preprocessing.storage import data_root, manifest_path


REQUIRED = {'study_id', 'patient_id', 'label', 'image_path', 'mask_path',
            'annotation_tier', 'external_source'}


def load_manifest(path):
    """Paths are relative to the manifest, not the process working directory."""
    path = Path(path).resolve()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('Manifest is empty. Add real studies before running.')
    for column in ('study_id', 'patient_id', 'label', 'image_path'):
        if frame[column].str.strip().ne(frame[column]).any() or frame[column].eq('').any():
            raise ValueError(f'{column} contains blank values or surrounding spaces')
    if frame.study_id.duplicated().any():
        raise ValueError('study_id must be unique')
    if not frame.study_id.str.fullmatch(r'[A-Za-z0-9_-]+').all():
        raise ValueError('study_id must contain only letters, numbers, underscores or hyphens')
    if not frame.label.isin(['0', '1']).all():
        raise ValueError('label must be 0 (non-PDAC) or 1 (PDAC), from clinical metadata')
    if (frame.groupby('patient_id').label.nunique() != 1).any():
        raise ValueError('Conflicting labels for the same patient')
    if not frame.annotation_tier.isin(['manual', 'automatic', 'none']).all():
        raise ValueError('annotation_tier must be manual, automatic or none')
    if (frame.mask_path.eq('') != frame.annotation_tier.eq('none')).any():
        raise ValueError('Missing masks require annotation_tier=none, and vice versa')
    for column in ('image_path', 'mask_path'):
        frame[column] = frame[column].map(
            lambda value: str((path.parent / value).resolve()) if value else '')
        for value in frame[column]:
            if value and not Path(value).is_file():
                raise ValueError(f'File not found: {value}')
    if frame.image_path.duplicated().any():
        raise ValueError('One CT file is assigned to multiple studies')
    return frame


def patient_splits(frame, seed=42):
    """70/15/15 stratified patient split; small cohorts fail explicitly."""
    patients = frame[['patient_id', 'label']].drop_duplicates().sort_values('patient_id')
    try:
        train, held = train_test_split(patients, test_size=0.30, random_state=seed,
                                      stratify=patients.label)
        val, test = train_test_split(held, test_size=0.50, random_state=seed,
                                    stratify=held.label)
    except ValueError as exc:
        raise ValueError('Not enough patients per class for stratified 70/15/15 splits. '
                         'Use the full cohort or a team-approved existing split.') from exc
    return pd.concat([part.assign(split=name) for part, name in
                      ((train, 'train'), (val, 'val'), (test, 'test'))]
                     )[['patient_id', 'split', 'label']].sort_values('patient_id').reset_index(drop=True)


def validate_splits(frame, splits, allow_subset=False):
    if set(splits.columns) != {'patient_id', 'split', 'label'}:
        raise ValueError('Split columns must be patient_id, split, label')
    if splits.patient_id.duplicated().any():
        raise ValueError('Each patient must occur exactly once in splits')
    if splits.patient_id.eq('').any() or not splits.label.isin(['0', '1']).all():
        raise ValueError('Invalid patient identifiers or labels in shared split')
    if not splits.split.isin(['train', 'val', 'test']).all():
        raise ValueError('Unknown split name')
    expected = frame[['patient_id', 'label']].drop_duplicates().set_index('patient_id').label
    actual = splits.set_index('patient_id').label
    if (not set(expected.index).issubset(actual.index) or
            (not allow_subset and set(expected.index) != set(actual.index))):
        raise ValueError('Split patients differ from manifest; existing splits were not overwritten')
    if not actual.reindex(expected.index).eq(expected).all():
        raise ValueError('Split labels differ from manifest labels')


def read_pair(row):
    image = sitk.ReadImage(row.image_path, sitk.sitkFloat32)
    if image.GetDimension() != 3 or min(image.GetSize()) < 2:
        raise ValueError('Expected a nonempty 3D CT volume')
    if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
        raise ValueError('CT contains NaN or infinity')
    if not np.isfinite(image.GetSpacing()).all() or min(image.GetSpacing()) <= 0:
        raise ValueError('Invalid voxel spacing')
    mask = None
    if row.mask_path:
        mask = sitk.ReadImage(row.mask_path)
        if mask.GetDimension() != 3:
            raise ValueError('Expected a 3D annotation')
        if mask.GetSize() != image.GetSize() or any(
            not np.allclose(getattr(mask, method)(), getattr(image, method)(), atol=1e-4, rtol=0)
            for method in ('GetSpacing', 'GetOrigin', 'GetDirection')
        ):
            raise ValueError('CT/mask geometry differs. Review registration before processing.')
        values = np.unique(sitk.GetArrayViewFromImage(mask))
        if not np.isin(values, np.arange(7)).all():
            raise ValueError(f'Unexpected annotation values: {values.tolist()}')
        if row.label == '0' and 1 in values:
            raise ValueError('PDAC lesion annotation conflicts with non-PDAC clinical label')
        mask = sitk.Cast(mask, sitk.sitkUInt8)
    return image, mask


def preprocess(image, mask, spacing, window):
    """Canonical LPS geometry, linear CT and nearest-neighbour label resampling."""
    image = sitk.DICOMOrient(image, 'LPS')
    if mask is not None:
        mask = sitk.DICOMOrient(mask, 'LPS')
    old_size, old_spacing = image.GetSize(), image.GetSpacing()
    size = [max(2, int(np.ceil((n - 1) * old / new)) + 1)
            for n, old, new in zip(old_size, old_spacing, spacing)]
    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetSpacing(spacing)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    transform = sitk.Transform(3, sitk.sitkIdentity)
    image = sitk.Resample(image, reference, transform, sitk.sitkLinear, -1024,
                         sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(image)
    array = ((np.clip(array, *window) - window[0]) / (window[1] - window[0])).astype(np.float32)
    normalized = sitk.GetImageFromArray(array)
    normalized.CopyInformation(reference)
    labels = None if mask is None else sitk.Resample(
        mask, reference, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return normalized, labels


def run(args):
    if min(args.spacing) <= 0 or not np.isfinite(args.spacing).all():
        raise ValueError('Spacing must be finite and positive')
    if not np.isfinite(args.window).all() or args.window[1] <= args.window[0]:
        raise ValueError('Window requires a finite lower bound smaller than the upper bound')
    frame = load_manifest(args.manifest)
    # Validate every study before creating a split or writing processed images.
    audit = []
    for row in frame.itertuples(index=False):
        try:
            image, mask = read_pair(row)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f'{row.study_id}: {exc}') from exc
        audit.append({'study_id': row.study_id, 'size_xyz': image.GetSize(),
                      'spacing_xyz_mm': image.GetSpacing(), 'has_mask': mask is not None,
                      'annotation_tier': row.annotation_tier})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if args.command == 'validate':
        print(f'Validated {len(frame)} studies; audit: {output / "audit.json"}')
        return
    split_path = Path(args.splits)
    if not split_path.exists():
        raise ValueError('Shared split is missing. Run python -m src.preprocessing.create_splits '
                         'from the full clinical spreadsheet first.')
    splits = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    validate_splits(frame, splits, allow_subset=True)
    frame = frame.merge(splits[['patient_id', 'split']], on='patient_id', validate='many_to_one')
    # Fail before processing if any target already exists. Never mix configurations silently.
    for row in frame.itertuples(index=False):
        if (output / row.split / row.study_id).exists():
            raise ValueError(f'Output exists for {row.study_id}; choose a fresh --output directory')
    records = []
    for row in frame.itertuples(index=False):
        image, mask = read_pair(row)
        image, mask = preprocess(image, mask, args.spacing, args.window)
        destination = output / row.split / row.study_id
        destination.mkdir(parents=True)
        sitk.WriteImage(image, str(destination / 'image.nii.gz'))
        if mask is not None:
            sitk.WriteImage(mask, str(destination / 'labels.nii.gz'))
            values = sitk.GetArrayFromImage(mask)
            for name, value in (('pancreas', 4), ('lesion', 1)):
                binary = sitk.GetImageFromArray((values == value).astype(np.uint8))
                binary.CopyInformation(image)
                sitk.WriteImage(binary, str(destination / f'{name}.nii.gz'))
        record = row._asdict()
        record.update(processed_image=str((destination / 'image.nii.gz').resolve()),
                      processed_labels=str((destination / 'labels.nii.gz').resolve()) if mask is not None else '',
                      num_slices=image.GetSize()[2])
        records.append(record)
        print(f'Prepared {row.study_id} ({row.split})')
    pd.DataFrame(records).to_csv(output / 'manifest.csv', index=False)
    (output / 'config.json').write_text(json.dumps({
        'spacing_xyz_mm': args.spacing, 'window_hu': args.window, 'orientation': 'LPS',
        'array_order': 'zyx', 'seed': args.seed, 'splits': str(split_path.resolve()),
        'input_manifest': str(Path(args.manifest).resolve()),
        'pancreas_value': 4, 'lesion_value': 1,
    }, indent=2) + '\n')
    print(f'Prepared {len(records)} studies. Outputs: {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'prepare'])
    parser.add_argument('--manifest', default=str(manifest_path()))
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--output', default=str(data_root() / 'processed'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.5, 1.5, 3.0],
                        metavar=('X', 'Y', 'Z'))
    parser.add_argument('--window', nargs=2, type=float, default=[-100, 300],
                        metavar=('LOW', 'HIGH'))
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
