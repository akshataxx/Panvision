"""Predict P3 nnU-Net pancreas masks for every completed study, split across Colab sessions and resumable.

    python -m src.segmentation.nnunet_export install-trainer
    python -m src.segmentation.nnunet_predict --model <.../nnUNetTrainer_PanVision__nnUNetPlans__2d> \
        --shard 0 --num-shards 3

Each session takes every num_shards-th study (sorted by study_id) and skips studies already in the output folder,
so sessions never overlap and a re-run continues where it stopped. The model is loaded once; studies are predicted
in chunks to fast local disk, then each mask is copied to <data>/outputs/segmentation/nnunet/predictions under a
temporary name and renamed, so a disconnect never leaves a half-written mask for MaskFolder to read.
Settings match the earlier P3 predictions (checkpoint_best, fold 0, no mirroring).
"""

import argparse
import shutil
import time
from pathlib import Path

from src.preprocessing.storage import data_root
from src.segmentation.data import find_studies


def shard_rows(studies, shard, num_shards, done=()):
    rows = studies.sort_values('study_id').iloc[shard::num_shards]
    return rows[~rows.study_id.isin(set(done))]


def predicted(folder):
    return {p.name.removesuffix('.nii.gz') for p in Path(folder).glob('*.nii.gz')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='nnU-Net trainer folder containing plans.json and fold_0/')
    parser.add_argument('--shard', type=int, default=0, help="this session's shard, 0-based")
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--chunk', type=int, default=50, help='studies per nnU-Net call')
    parser.add_argument('--workers', type=int, default=2, help='nnU-Net preprocessing/export processes')
    parser.add_argument('--work', default='/content/nnunet/full_predictions', help='fast local scratch')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--out', default=None, help='default <data>/outputs/segmentation/nnunet/predictions')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        parser.error('--shard must be in [0, --num-shards)')

    root = Path(args.data_root) if args.data_root else data_root()
    out = Path(args.out) if args.out else root / 'outputs/segmentation/nnunet/predictions'
    out.mkdir(parents=True, exist_ok=True)
    print('Finding studies (a few minutes on Drive)...', flush=True)
    studies = find_studies(root, args.splits)
    mine = shard_rows(studies, args.shard, args.num_shards)
    rows = shard_rows(studies, args.shard, args.num_shards, predicted(out))
    print(f'Shard {args.shard + 1}/{args.num_shards}: {len(mine)} studies; already predicted: '
          f'{len(mine) - len(rows)}; pending: {len(rows)}', flush=True)
    if args.dry_run or rows.empty:
        return

    import torch
    if not torch.cuda.is_available():
        raise SystemExit('No GPU: Runtime -> Change runtime type -> T4 GPU')
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                perform_everything_on_device=True, device=torch.device('cuda'),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(args.model, use_folds=(0,), checkpoint_name='checkpoint_best.pth')
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    start, finished = time.time(), 0
    for begin in range(0, len(rows), args.chunk):
        chunk = list(rows.iloc[begin:begin + args.chunk].itertuples(index=False))
        predictor.predict_from_files([[row.processed_image] for row in chunk],
                                     [str(work / row.study_id) for row in chunk],
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=args.workers,
                                     num_processes_segmentation_export=args.workers)
        for row in chunk:
            local, temporary = work / f'{row.study_id}.nii.gz', out / f'.{row.study_id}.nii.gz.tmp'
            shutil.copyfile(local, temporary)
            temporary.replace(out / f'{row.study_id}.nii.gz')
            local.unlink()
        finished += len(chunk)
        left = (time.time() - start) / finished * (len(rows) - finished) / 60
        print(f'[{finished}/{len(rows)}] predicted  ~{left:.0f} min left', flush=True)
    print(f'Shard {args.shard + 1}/{args.num_shards} done: {finished} masks in {out}')


if __name__ == '__main__':
    main()
