"""Validation-only lesion ensemble: average two saved models' probabilities, then score masks.

Both checkpoints are read for inference; this module never trains or modifies them.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.storage import data_root
from src.segmentation.data import LESION_VALUE, find_studies, iter_cases, lesion_rows
from src.segmentation.infer import LesionSegmentor, pancreas_source
from src.segmentation.measure import max_axial_diameter_mm
from src.segmentation.roi import crop, remove_small
from src.segmentation.tune import score_mask, summarize


def average_probabilities(first, second):
    """Average aligned lesion probabilities; fail if model crops do not match."""
    probabilities_a, roi_a = first
    probabilities_b, roi_b = second
    if roi_a != roi_b:
        raise ValueError('Ensemble models produced different pancreas ROIs')
    if probabilities_a is None or probabilities_b is None:
        if probabilities_a is not None or probabilities_b is not None:
            raise ValueError('Only one ensemble model returned probabilities')
        return None, roi_a
    if probabilities_a.shape != probabilities_b.shape:
        raise ValueError('Ensemble probability volumes have different shapes')
    return (probabilities_a.astype(np.float32) + probabilities_b.astype(np.float32)) / 2, roi_a


def run(args):
    if not all(0 < value < 1 for value in args.thresholds):
        raise ValueError('Thresholds must lie strictly between 0 and 1')
    if not all(value >= 1 for value in args.min_voxels):
        raise ValueError('Minimum component sizes must be positive')
    root = Path(args.data_root) if args.data_root else data_root()
    studies = find_studies(root, args.splits)
    positives, negatives = lesion_rows(studies, 'val')
    rows = pd.concat([positives, negatives], ignore_index=True)
    pancreas = pancreas_source(args.pancreas_checkpoint, args.pancreas_predictions)
    first, second = LesionSegmentor(args.first), LesionSegmentor(args.second)
    grid = [(threshold, size) for threshold in sorted(set(args.thresholds))
            for size in sorted(set(args.min_voxels))]
    records = []
    for index, case in enumerate(iter_cases(rows), 1):
        pancreas_mask = pancreas.predict_case(case)
        first_result = first.predict_probabilities(case['image'], pancreas_mask, case['spacing'])
        second_result = second.predict_probabilities(case['image'], pancreas_mask, case['spacing'])
        probabilities, roi = average_probabilities(first_result, second_result)
        reference = case['labels'] == LESION_VALUE
        reference_voxels = int(reference.sum())
        if probabilities is None:
            reference_crop = valid_crop = None
        elif roi is None:
            reference_crop = reference
            valid_crop = np.ones(probabilities.shape, dtype=bool)
        else:
            reference_crop = crop(reference, roi)
            valid_crop = crop(np.ones(case['image'].shape, dtype=bool), roi)
        voxel_cm3 = float(np.prod(case['spacing'])) / 1000
        reference_volume = reference_voxels * voxel_cm3
        reference_diameter = (max_axial_diameter_mm(reference, case['spacing'])
                              if case['label'] == '1' else float('nan'))
        for model_name, model_probabilities in (
                ('L2', first_result[0]), ('L5', second_result[0]), ('L2_L5_ensemble', probabilities)):
            for threshold, min_voxels in grid:
                mask = (remove_small(model_probabilities >= threshold, min_voxels) & valid_crop
                        if model_probabilities is not None else np.zeros((0,), dtype=bool))
                metrics = (score_mask(mask, reference_crop, valid_crop, reference_voxels)
                           if model_probabilities is not None else score_mask(mask, mask, mask, reference_voxels))
                record = {'study_id': case['study_id'], 'label': case['label'], 'model': model_name,
                          'threshold': threshold, 'min_voxels': min_voxels, **metrics}
                if case['label'] == '1':
                    record['volume_abs_error_cm3'] = abs(int(mask.sum()) * voxel_cm3 - reference_volume)
                    record['diameter_abs_error_mm'] = abs(
                        max_axial_diameter_mm(mask, case['spacing']) - reference_diameter)
                records.append(record)
        if index % 10 == 0 or index == len(rows):
            print(f'Scored {index}/{len(rows)} validation studies', flush=True)
    frame = pd.DataFrame(records)
    summary = pd.concat([summarize(group).assign(model=model_name)
                         for model_name, group in frame.groupby('model', sort=False)], ignore_index=True)
    positive = frame[frame.label == '1'].groupby(['model', 'threshold', 'min_voxels'])
    errors = positive[['volume_abs_error_cm3', 'diameter_abs_error_mm']].mean().reset_index()
    summary = summary.merge(errors, on=['model', 'threshold', 'min_voxels'], how='left')
    out = Path(args.out) if args.out else root / 'outputs/segmentation/results/L2_L5_ensemble_val.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    frame.to_csv(out.with_name(out.stem + '_cases.csv'), index=False)
    print(summary.to_string(index=False))
    print(f'Validation-only ensemble results: {out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True, help='first saved lesion checkpoint')
    parser.add_argument('--second', required=True, help='second saved lesion checkpoint')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pancreas-checkpoint')
    source.add_argument('--pancreas-predictions')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.55, 0.6])
    parser.add_argument('--min-voxels', type=int, nargs='+', default=[10, 30, 60])
    parser.add_argument('--data-root')
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--out')
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
