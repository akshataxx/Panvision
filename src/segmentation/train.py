"""Train one candidate segmentation model (run on Colab GPU; tiny local runs are only smoke tests).

    python -m src.segmentation.train --stage pancreas --model unet --context 1 --run P1_unet_2d
    python -m src.segmentation.train --stage pancreas --model unet --context 3 --run P2_unet_25d
    python -m src.segmentation.train --stage lesion --model unet --run L1_unet_roi
    python -m src.segmentation.train --stage lesion --model attention_unet --run L2_attention_unet_roi
    python -m src.segmentation.train --stage lesion --model unet --roi none --run L0_unet_fullslice

Each run writes <models>/segmentation/<run>/{config.json, log.csv, best.pt, last.pt}. best.pt is chosen by
validation Dice (pancreas: per-volume Dice; lesion: per-volume Dice on PDAC cases with the reference ROI).
Re-running the same command resumes from last.pt.
"""

import argparse
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss, DiceFocalLoss
from torch.utils.data import DataLoader

from src.preprocessing.storage import data_root
from src.segmentation.data import (FullSlices, LesionSlices, find_studies, lesion_band, lesion_crop, lesion_rows,
                                   load_case, pancreas_band, pancreas_rows, subsample)
from src.segmentation.evaluate import evaluate_lesion, evaluate_pancreas
from src.segmentation.infer import LesionSegmentor, PancreasSegmentor, get_device
from src.segmentation.models import MODELS, build_model


def load_prepared(rows, transform, cache_dir, tag, workers=8):
    """Load rows and apply `transform`, caching results on fast local disk across runs in one session."""
    def one(row):
        path = Path(cache_dir) / f'{tag}_{row.study_id}.pkl' if cache_dir else None
        if path is not None and path.exists():
            return pickle.loads(path.read_bytes())
        item = transform(load_case(row))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL))
        return item
    start = time.time()
    with ThreadPoolExecutor(workers) as pool:
        items = list(pool.map(one, rows.itertuples(index=False)))
    print(f'Loaded {len(items)} {tag} cases in {time.time() - start:.0f}s', flush=True)
    return items


def build_data(args, studies):
    length = args.iters * args.batch
    if args.stage == 'pancreas':
        train_rows = subsample(pancreas_rows(studies, 'train'), args.max_train, args.seed)
        bands = load_prepared(train_rows, pancreas_band, args.cache_dir, 'band')
        dataset = FullSlices(bands, length, args.patch, args.context)
        val_rows = subsample(pancreas_rows(studies, 'val'), args.max_val, args.seed)
    else:
        positives, negatives = lesion_rows(studies, 'train')
        positives = subsample(positives, args.max_train, args.seed)
        negatives = subsample(negatives, len(positives), args.seed)
        train_rows = pd.concat([positives, negatives])
        if args.roi == 'none':  # L0: full slices, no pancreas ROI
            bands = load_prepared(train_rows, lesion_band, args.cache_dir, 'lesionband')
            dataset = FullSlices(bands, length, args.patch, args.context)
        else:
            crop = partial(lesion_crop, size=args.patch, jitter=args.jitter)
            tag = f'lesion{args.patch}'
            dataset = LesionSlices(load_prepared(positives, crop, args.cache_dir, tag),
                                   load_prepared(negatives, crop, args.cache_dir, tag),
                                   length, args.patch, args.jitter, args.context)
        val_pos, val_neg = lesion_rows(studies, 'val')
        val_rows = pd.concat([subsample(val_pos, args.max_val, args.seed),
                              subsample(val_neg, max(1, args.max_val // 3), args.seed)])
    val_cases = load_prepared(val_rows, lambda case: case, args.cache_dir, 'full')
    print(f'Train: {len(train_rows)} studies; val: {len(val_cases)} studies', flush=True)
    return dataset, val_cases


def validate(args, model, device, val_cases):
    meta = {'context': args.context, 'patch': args.patch, 'roi': args.roi}
    if args.stage == 'pancreas':
        frame = evaluate_pancreas(PancreasSegmentor((model, meta), device), val_cases)
        return float(frame.dice.mean()), {'val_dice': frame.dice.mean(), 'val_iou': frame.iou.mean()}
    frame = evaluate_lesion(LesionSegmentor((model, meta), device), val_cases)
    positives, negatives = frame[frame.label == '1'], frame[frame.label == '0']
    metrics = {'val_dice': positives.dice.mean(), 'val_iou': positives.iou.mean(),
               'val_recall': positives.recall.mean(), 'val_detected': positives.detected.mean(),
               'val_false_positive': negatives.false_positive.mean() if len(negatives) else float('nan')}
    return float(metrics['val_dice']), metrics


def checkpoint(args, model, step, metrics):
    return {'stage': args.stage, 'model': args.model, 'context': args.context, 'patch': args.patch, 'roi': args.roi,
            'state_dict': model.state_dict(), 'step': step, 'metrics': metrics, 'config': vars(args)}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    run_dir = Path(args.out) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    studies = find_studies(args.data_root, args.splits)
    dataset, val_cases = build_data(args, studies)

    model = build_model(args.model, args.context).to(device)
    # batch=True pools Dice over the batch: many 2D patches have no foreground, and per-slice Dice on an empty
    # target stays near 1 whatever the prediction, which drowns the useful gradient.
    loss_fn = (DiceCELoss if args.loss == 'dicece' else DiceFocalLoss)(sigmoid=True, batch=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.iters,
                                                    pct_start=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    step, best = 0, -1.0
    if (run_dir / 'last.pt').exists():
        state = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        step, best = state['step'], state['best']
        print(f'Resumed {args.run} at step {step} (best val Dice {best:.4f})', flush=True)
    if step >= args.iters:
        print('Run already complete')
        return

    remaining = torch.utils.data.Subset(dataset, range(step * args.batch, args.iters * args.batch))
    loader = DataLoader(remaining, batch_size=args.batch, num_workers=args.workers, drop_last=True,
                        pin_memory=device.type == 'cuda')
    log_path = run_dir / 'log.csv'
    running, start = [], time.time()
    for image, target in loader:
        model.train()
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            loss = loss_fn(model(image), target)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        running.append(loss.item())
        if step % 50 == 0:
            print(f'step {step}/{args.iters} loss {np.mean(running[-50:]):.4f} '
                  f'({(time.time() - start) / len(running):.2f}s/step)', flush=True)
        if step % args.val_every == 0 or step == args.iters:
            score, metrics = validate(args, model, device, val_cases)
            row = {'step': step, 'train_loss': float(np.mean(running)), 'lr': scheduler.get_last_lr()[0], **metrics}
            pd.DataFrame([row]).to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
            running = []
            if score > best:
                best = score
                torch.save(checkpoint(args, model, step, metrics), run_dir / 'best.pt')
            state = checkpoint(args, model, step, metrics)
            state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), best=best)
            torch.save(state, run_dir / 'last.pt')
            print('  '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items())
                  + f'  best={best:.4f}', flush=True)
        if args.stop_after and step >= args.stop_after:
            print(f'Stopped at step {step}; re-run the same command to resume')
            return
    print(f'Finished {args.run}: best val Dice {best:.4f} -> {run_dir / "best.pt"}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', required=True, choices=['pancreas', 'lesion'])
    parser.add_argument('--model', default='unet', choices=MODELS)
    parser.add_argument('--context', type=int, default=None, choices=[1, 3, 5],
                        help='slices per input; default 1 (pancreas) / 3 (lesion)')
    parser.add_argument('--loss', default=None, choices=['dicece', 'dicefocal'],
                        help='default dicece (pancreas) / dicefocal (lesion)')
    parser.add_argument('--run', required=True, help='run name, e.g. P1_unet_2d')
    parser.add_argument('--iters', type=int, default=4000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--roi', default='pancreas', choices=['pancreas', 'none'],
                        help="lesion only: 'none' trains the full-slice baseline without the pancreas ROI")
    parser.add_argument('--patch', type=int, default=None, help='default 192 (full slices) / 160 (lesion ROI)')
    parser.add_argument('--jitter', type=int, default=16, help='lesion ROI random shift in pixels')
    parser.add_argument('--max-train', type=int, default=250, help='0 = all train studies')
    parser.add_argument('--max-val', type=int, default=30)
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--stop-after', type=int, default=0, help='pause after this step (at a validation step)')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache-dir', default=None, help='local disk cache, e.g. /content/cache on Colab')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--out', default=None, help='default <data_root>/models/segmentation')
    args = parser.parse_args(argv)
    lesion = args.stage == 'lesion'
    args.context = args.context or (3 if lesion else 1)
    args.loss = args.loss or ('dicefocal' if lesion else 'dicece')
    args.patch = args.patch or (160 if lesion and args.roi == 'pancreas' else 192)
    args.out = args.out or str((Path(args.data_root) if args.data_root else data_root()) / 'models/segmentation')
    return args


if __name__ == '__main__':
    train(parse_args())
