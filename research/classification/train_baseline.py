"""Train CNN or DenseNet-121 on shared crop splits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import math
import random

import numpy as np

try:
    from .experiment_utils import audit_crops, file_hash, score_metrics, threshold_on_validation
except ImportError:
    from experiment_utils import audit_crops, file_hash, score_metrics, threshold_on_validation


def main(default_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['cnn', 'densenet'], default=default_model, required=default_model is None)
    parser.add_argument('--crops', required=True)
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--out', required=True, help='New experiment directory')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0 or not math.isfinite(args.lr) or args.lr <= 0 or (args.max_batches is not None and args.max_batches < 1):
        parser.error('Invalid training settings')
    import torch
    from torch.utils.data import DataLoader
    try:
        from src.classification.dataset import CropDataset
        from src.classification.evaluation import patient_scores, predict_probs
        from . import cnn_model, densenet_model
    except ImportError:
        from src.classification.dataset import CropDataset
        from src.classification.evaluation import patient_scores, predict_probs
        import cnn_model, densenet_model
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    audit, window = audit_crops(args.crops, args.splits)
    for split in ('train', 'val'):
        if set(audit.loc[audit.split == split, 'label']) != {0, 1}:
            raise ValueError(f'{split} requires both classes')
    train = CropDataset(args.crops, 'train', mode='train', augment=True)
    val = CropDataset(args.crops, 'val', mode='eval')
    if set(train.patient_ids) & set(val.patient_ids):
        raise ValueError('Patients cross training/validation splits')
    module = cnn_model if args.model == 'cnn' else densenet_model
    model = (module.SimpleCNNClassifier() if args.model == 'cnn' else module.DenseNet121Classifier()).to(device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'splits_sha256': file_hash(args.splits), 'window_mm': window,
              'scope': 'annotation-assisted crops; not deployment evaluation', 'aggregation': 'mean'}
    (output/'config.json').write_text(json.dumps(config, indent=2))
    audit.to_csv(output/'crop_audit.csv', index=False)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., train.pos_weight()], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    best, history = -1., []
    print(f'Device: {device}; train patients: {len(train.patient_ids)}; val patients: {len(val.patient_ids)}', flush=True)
    for epoch in range(1, args.epochs+1):
        model.train(); total, seen = 0., 0
        for step, (x, y, _) in enumerate(loader):
            if args.max_batches is not None and step >= args.max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward(); optimizer.step()
            total += float(loss.detach())*len(y); seen += len(y)
        if not seen:
            raise ValueError('No training batches')
        probs, patient = predict_probs(model, val, device, batch_size=args.batch_size, workers=0)
        scores, labels = patient_scores(probs, patient, val.patient_labels)
        metrics = score_metrics(scores, labels, 0.5)
        metrics.update(epoch=epoch, train_loss=total/seen)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics['auc'] > best:
            best = metrics['auc']
            module.save_checkpoint(model, str(output/'best.pt'), epoch=epoch, val_auc=best,
                                   threshold=threshold_on_validation(scores, labels), config=config,
                                   smoke_test=args.max_batches is not None)
        module.save_checkpoint(model, str(output/'latest.pt'), epoch=epoch, val_auc=metrics['auc'],
                               optimizer_state=optimizer.state_dict(), config=config)
        (output/'history.json').write_text(json.dumps(history, indent=2))
    print(f'Best validation AUC: {best:.4f}; checkpoint: {output / "best.pt"}')


if __name__ == '__main__':
    main()
