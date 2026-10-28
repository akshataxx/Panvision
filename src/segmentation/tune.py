"""Sweep lesion postprocessing on validation studies without retraining or touching the test split.

Example:
    python -m src.segmentation.tune --checkpoint models/segmentation/L2_attention_unet_roi/best.pt \
        --pancreas-predictions outputs/segmentation/nnunet/predictions
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.storage import data_root
from src.segmentation.data import LESION_VALUE, find_studies, iter_cases, lesion_rows
from src.segmentation.infer import LesionSegmentor, pancreas_source
from src.segmentation.roi import crop, remove_small


def score_mask(mask, reference_crop, valid_crop, reference_voxels):
    """Score an ROI mask against the full-volume reference, including lesions outside the ROI as misses."""
    mask = mask & valid_crop
    predicted = int(mask.sum())
    true_positive = int(np.logical_and(mask, reference_crop).sum())
    false_negative = reference_voxels - true_positive
    false_positive = predicted - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        'dice': 2 * true_positive / denominator if denominator else 1.0,
        'iou': true_positive / (true_positive + false_positive + false_negative)
        if (true_positive + false_positive + false_negative) else 1.0,
        'recall': true_positive / reference_voxels if reference_voxels else float('nan'),
        'detected': float(true_positive > 0),
        'false_positive': float(predicted > 0),
    }


def summarize(frame):
    rows = []
    for (threshold, min_voxels), group in frame.groupby(['threshold', 'min_voxels'], sort=True):
        positive = group[group.label == '1']
        negative = group[group.label == '0']
        rows.append({
            'threshold': threshold, 'min_voxels': min_voxels,
            'pdac_cases': len(positive), 'non_pdac_cases': len(negative),
            'dice': positive.dice.mean(), 'iou': positive.iou.mean(),
            'recall': positive.recall.mean(), 'detection_rate': positive.detected.mean(),
            'false_positive_rate': negative.false_positive.mean(),
            'false_positive_cases': int(negative.false_positive.sum()),
        })
    return pd.DataFrame(rows)


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
    lesion = LesionSegmentor(args.checkpoint)
    grid = [(threshold, size) for threshold in sorted(set(args.thresholds))
            for size in sorted(set(args.min_voxels))]
    records = []
    for index, case in enumerate(iter_cases(rows), 1):
        pancreas_mask = pancreas.predict_case(case)
        probabilities, roi = lesion.predict_probabilities(case['image'], pancreas_mask, case['spacing'])
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
        for threshold, min_voxels in grid:
            mask = (remove_small(probabilities >= threshold, min_voxels)
                    if probabilities is not None else np.zeros((0,), dtype=bool))
            metrics = (score_mask(mask, reference_crop, valid_crop, reference_voxels)
                       if probabilities is not None else
                       score_mask(mask, mask, mask, reference_voxels))
            records.append({'study_id': case['study_id'], 'label': case['label'],
                            'threshold': threshold, 'min_voxels': min_voxels, **metrics})
        if index % 10 == 0 or index == len(rows):
            print(f'Scored {index}/{len(rows)} validation studies', flush=True)
    frame = pd.DataFrame(records)
    summary = summarize(frame)
    out = Path(args.out) if args.out else root / 'outputs/segmentation/results/lesion_val_tuning.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    frame.to_csv(out.with_name(out.stem + '_cases.csv'), index=False)
    print(summary.to_string(index=False))
    print(f'Validation-only tuning results: {out}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pancreas-checkpoint')
    source.add_argument('--pancreas-predictions')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument('--min-voxels', type=int, nargs='+', default=[10, 30, 60])
    parser.add_argument('--data-root')
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--out')
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
