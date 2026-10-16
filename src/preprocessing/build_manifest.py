"""Join downloaded PANORAMA CTs to the official clinical spreadsheet."""

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.storage import data_root, manifest_path


def build_manifest(images, annotations, clinical, studies=None):
    images, annotations = Path(images), Path(annotations)
    if not images.is_dir() or not annotations.is_dir():
        raise ValueError('Images and annotations must be existing directories')
    clinical = Path(clinical)
    metadata = pd.read_excel(clinical, dtype=str, keep_default_na=False)
    required = {'PANORAMA_patient_id', 'PANORAMA_study_id', 'label', 'level'}
    if not required.issubset(metadata.columns):
        raise ValueError(f'Clinical spreadsheet missing {sorted(required - set(metadata.columns))}')
    if metadata.PANORAMA_study_id.duplicated().any():
        raise ValueError('Clinical spreadsheet has duplicate study IDs')
    if not metadata.label.isin(['PDAC', 'non-PDAC']).all():
        raise ValueError('Unrecognized clinical diagnosis; review spreadsheet')
    indexed = metadata.set_index('PANORAMA_study_id')
    records = []
    for image in sorted(images.rglob('*.nii.gz')):
        # Official CT convention: patient_study_0000.nii.gz; masks omit _0000.
        if not image.name.endswith('_0000.nii.gz'):
            raise ValueError(f'Unexpected CT filename: {image.name}')
        study = image.name.removesuffix('_0000.nii.gz')
        if studies and study not in studies:
            continue
        if study not in indexed.index:
            raise ValueError(f'No clinical metadata for {study}')
        info = indexed.loc[study]
        tier, mask = 'none', ''
        for candidate in ('manual', 'automatic'):
            path = annotations / f'{candidate}_labels' / f'{study}.nii.gz'
            if path.is_file():
                tier, mask = candidate, str(path.resolve())
                break
        records.append(dict(
            study_id=study, patient_id=info.PANORAMA_patient_id,
            label='1' if info.label == 'PDAC' else '0', image_path=str(image.resolve()),
            mask_path=mask, annotation_tier=tier,
            external_source=info.level if info.level in ('MSD_dataset', 'NIH_dataset') else 'PANORAMA',
            level=info.level,
        ))
    if not records:
        raise ValueError('No CT files found under images directory')
    frame = pd.DataFrame(records)
    if frame.study_id.duplicated().any():
        raise ValueError('Duplicate CT studies across image folders')
    if studies and set(studies) != set(frame.study_id):
        raise ValueError(f'Requested studies not downloaded: {sorted(set(studies) - set(frame.study_id))}')
    return frame, len(metadata) - len(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    parser.add_argument('--images', default=str(root / 'raw/images'))
    parser.add_argument('--annotations', default=str(root / 'raw/annotations'))
    parser.add_argument('--clinical', default=str(root / 'raw/annotations/clinical_information.xlsx'))
    parser.add_argument('--output', default=str(manifest_path()))
    parser.add_argument('--studies', nargs='+', help='Only include these study IDs for a batch')
    args = parser.parse_args()
    try:
        output = Path(args.output)
        if output.exists():
            raise ValueError('Manifest already exists; use another --output to avoid overwriting')
        frame, missing = build_manifest(args.images, args.annotations, args.clinical, args.studies)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f'Linked {len(frame)} studies / {frame.patient_id.nunique()} patients to {output}')
        print(f'{missing} spreadsheet studies not downloaded. Manifest covers LOCAL CTs only.')
        print('Diagnosis counts:', frame.label.value_counts().to_dict())
        print('Lesion annotation tiers:', frame.annotation_tier.value_counts().to_dict())
        print('Run pipeline validate and inspect annotations before preparing a shared split.')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
