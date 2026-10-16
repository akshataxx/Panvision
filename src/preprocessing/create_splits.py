"""Create the shared patient split from full metadata, independent of local CTs."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn

from src.preprocessing.pipeline import patient_splits, validate_splits
from src.preprocessing.storage import data_root


def clinical_cohort(path):
    frame = pd.read_excel(path, dtype=str, keep_default_na=False)
    columns = {'PANORAMA_patient_id', 'PANORAMA_study_id', 'label'}
    if not columns.issubset(frame.columns):
        raise ValueError(f'Missing clinical columns: {sorted(columns - set(frame.columns))}')
    if frame.empty or frame.PANORAMA_study_id.duplicated().any():
        raise ValueError('Empty clinical table or duplicate study IDs')
    for column in ('PANORAMA_patient_id', 'PANORAMA_study_id'):
        if frame[column].str.strip().ne(frame[column]).any() or frame[column].eq('').any():
            raise ValueError(f'Invalid identifiers in {column}')
    if not frame.label.isin(['PDAC', 'non-PDAC']).all():
        raise ValueError('Unknown clinical diagnosis')
    frame = frame.rename(columns={'PANORAMA_patient_id': 'patient_id',
                                  'PANORAMA_study_id': 'study_id'})
    frame['label'] = frame.label.map({'PDAC': '1', 'non-PDAC': '0'})
    if (frame.groupby('patient_id').label.nunique() != 1).any():
        raise ValueError('Conflicting patient diagnoses need review before splitting')
    return frame[['patient_id', 'study_id', 'label']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clinical', default=str(data_root() / 'raw/annotations/clinical_information.xlsx'))
    parser.add_argument('--output', default='data/splits.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        clinical = Path(args.clinical)
        frame = clinical_cohort(clinical)
        output = Path(args.output)
        if output.exists():
            splits = pd.read_csv(output, dtype=str, keep_default_na=False)
            validate_splits(frame, splits)
            print('Existing matching split reused; assignments unchanged.')
        else:
            splits = patient_splits(frame, args.seed)
            validate_splits(frame, splits)
            output.parent.mkdir(parents=True, exist_ok=True)
            splits.to_csv(output, index=False)
            metadata = {'clinical_sha256': hashlib.sha256(clinical.read_bytes()).hexdigest(),
                        'seed': args.seed, 'fractions': [0.70, 0.15, 0.15],
                        'studies': len(frame), 'patients': frame.patient_id.nunique(),
                        'sklearn_version': sklearn.__version__,
                        'split_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
            output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'{len(frame)} studies, {frame.patient_id.nunique()} unique patients; split: {output}')
        print(pd.crosstab(splits.split, splits.label).rename(columns={'0': 'non-PDAC', '1': 'PDAC'}))
        print('Counts above are PATIENTS. Every study from each patient inherits their split.')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
