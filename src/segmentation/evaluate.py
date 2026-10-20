"""Per-case evaluation of segmentation checkpoints and paired comparison between candidates.

    python -m src.segmentation.evaluate pancreas --checkpoint P.pt --split val
    python -m src.segmentation.evaluate lesion --checkpoint L.pt --pancreas-checkpoint P.pt --split val
    python -m src.segmentation.evaluate compare A_val_cases.csv B_val_cases.csv --metric dice

Selection uses val; run test once per selected model. Lesion metrics are reported for two ROI sources:
'oracle' (reference pancreas) and 'end_to_end' (predicted pancreas), which shows cascade error propagation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.storage import data_root
from src.segmentation.data import (LESION_VALUE, find_studies, iter_cases, lesion_rows, pancreas_rows,
                                   pancreas_target, subsample)
from src.segmentation.metrics import overlap, paired_wilcoxon, summarize

VOXEL_CM3 = lambda spacing: float(np.prod(spacing)) / 1000.0


def evaluation_rows(studies, stage, split, max_studies=0, max_negatives=-1, seed=42):
    """The exact studies an evaluation uses; shared with the nnU-Net export so every candidate sees the same cases."""
    if stage == 'pancreas':
        return subsample(pancreas_rows(studies, split), max_studies, seed)
    positives, negatives = lesion_rows(studies, split)
    positives = subsample(positives, max_studies, seed)
    negatives = subsample(negatives, len(positives) if max_negatives < 0 else max_negatives, seed)
    return pd.concat([positives, negatives])


def evaluate_pancreas(segmentor, cases):
    """`segmentor` is anything with predict_case(case): a PancreasSegmentor or a MaskFolder of nnU-Net outputs."""
    rows = []
    for case in cases:
        pred = segmentor.predict_case(case)
        row = {'study_id': case['study_id'], 'patient_id': case['patient_id'], 'label': case['label'],
               'pred_empty': not pred.any()}
        row.update(overlap(pred, pancreas_target(case['labels'])))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_lesion(segmentor, cases, pancreas_segmentor=None):
    rows = []
    for case in cases:
        reference = case['labels'] == LESION_VALUE
        sources = {'oracle': pancreas_target(case['labels'])}
        if pancreas_segmentor is not None:
            sources['end_to_end'] = pancreas_segmentor.predict_case(case)
        for mode, pancreas in sources.items():
            pred, roi = segmentor.predict(case['image'], pancreas, case['spacing'])
            row = {'study_id': case['study_id'], 'patient_id': case['patient_id'], 'label': case['label'],
                   'mode': mode, 'roi_found': roi is not None,
                   'pred_volume_cm3': pred.sum() * VOXEL_CM3(case['spacing']),
                   'ref_volume_cm3': reference.sum() * VOXEL_CM3(case['spacing'])}
            if case['label'] == '1':
                row.update(overlap(pred, reference))
                row['detected'] = float(np.logical_and(pred, reference).any())
            else:
                row['false_positive'] = float(pred.any())
            rows.append(row)
    return pd.DataFrame(rows)


def lesion_summary(frame):
    positives = frame[frame.label == '1']
    negatives = frame[frame.label == '0']
    parts = []
    if len(positives):
        parts.append(summarize(positives, ['dice', 'iou', 'recall', 'detected'], by='mode'))
    if len(negatives):
        parts.append(summarize(negatives, ['false_positive'], by='mode'))
    return pd.concat(parts, ignore_index=True)




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('pancreas', 'lesion'):
        p = sub.add_parser(name)
        p.add_argument('--checkpoint', default=None)
        p.add_argument('--name', default=None, help='results file prefix; default: the checkpoint run folder')
        p.add_argument('--split', default='val', choices=['val', 'test'])
        p.add_argument('--max-studies', type=int, default=0, help='0 = all (lesion: positives)')
        if name == 'lesion':
            p.add_argument('--lesion-threshold', type=float, default=0.5)
            p.add_argument('--lesion-min-voxels', type=int, default=10)
        p.add_argument('--data-root', default=None)
        p.add_argument('--splits', default='data/splits.csv')
        p.add_argument('--out', default=None)
        if name == 'pancreas':
            p.add_argument('--predictions', default=None, help='folder of nnU-Net masks instead of --checkpoint')
        if name == 'lesion':
            p.add_argument('--pancreas-checkpoint', default=None, help='adds end_to_end mode')
            p.add_argument('--pancreas-predictions', default=None, help='nnU-Net masks for end_to_end mode')
            p.add_argument('--max-negatives', type=int, default=-1, help='-1 = as many as positives, 0 = all')
    p = sub.add_parser('compare')
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--metric', default='dice')
    p.add_argument('--mode', default=None, help='lesion only: oracle or end_to_end')
    args = parser.parse_args(argv)

    if args.command == 'compare':
        a, b = pd.read_csv(args.a, dtype={'study_id': str}), pd.read_csv(args.b, dtype={'study_id': str})
        if args.mode:
            a, b = a[a['mode'] == args.mode], b[b['mode'] == args.mode]
        print(paired_wilcoxon(a, b, args.metric))
        return

    from src.segmentation.infer import LesionSegmentor, pancreas_source
    root = Path(args.data_root) if args.data_root else data_root()
    out = Path(args.out) if args.out else root / 'outputs/segmentation/results'
    out.mkdir(parents=True, exist_ok=True)
    studies = find_studies(root, args.splits)
    if args.command == 'pancreas':
        rows = evaluation_rows(studies, 'pancreas', args.split, args.max_studies)
        frame = evaluate_pancreas(pancreas_source(args.checkpoint, args.predictions), iter_cases(rows))
        summary = summarize(frame, ['dice', 'iou'])
    else:
        rows = evaluation_rows(studies, 'lesion', args.split, args.max_studies, args.max_negatives)
        pancreas = (pancreas_source(args.pancreas_checkpoint, args.pancreas_predictions)
                    if args.pancreas_checkpoint or args.pancreas_predictions else None)
        frame = evaluate_lesion(LesionSegmentor(args.checkpoint, threshold=args.lesion_threshold,
                                               min_voxels=args.lesion_min_voxels), iter_cases(rows), pancreas)
        summary = lesion_summary(frame)
    name = args.name or Path(args.checkpoint or args.predictions).parent.name
    stem = out / f'{name}_{args.split}'
    frame.to_csv(f'{stem}_cases.csv', index=False)
    summary.to_csv(f'{stem}_summary.csv', index=False)
    print(summary.to_string(index=False))
    print(f'Per-case results: {stem}_cases.csv')


if __name__ == '__main__':
    main()
