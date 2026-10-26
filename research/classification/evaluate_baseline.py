"""Compare patient aggregation methods on validation crops."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import numpy as np
import pandas as pd

try:
    from .experiment_utils import METHODS, audit_crops, file_hash, score_metrics, threshold_on_validation
except ImportError:
    from experiment_utils import METHODS, audit_crops, file_hash, score_metrics, threshold_on_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['cnn', 'densenet'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--crops', required=True)
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--out', required=True, help='New evaluation directory')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap < 1:
        parser.error('batch-size and bootstrap must be positive')
    import torch
    try:
        from src.classification.dataset import CropDataset
        from src.classification.evaluation import patient_scores, predict_probs, bootstrap_auc_ci
        from . import cnn_model, densenet_model
    except ImportError:
        from src.classification.dataset import CropDataset
        from src.classification.evaluation import patient_scores, predict_probs, bootstrap_auc_ci
        import cnn_model, densenet_model
    audit_crops(args.crops, args.splits)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = cnn_model if args.model == 'cnn' else densenet_model
    model, notes = module.load_classifier(args.checkpoint, device)
    recorded = notes.get('config', {}).get('splits_sha256')
    if recorded and recorded != file_hash(args.splits):
        raise ValueError('Checkpoint shared split differs')
    val = CropDataset(args.crops, 'val', mode='eval')
    probs, patients = predict_probs(model, val, device, batch_size=args.batch_size, workers=0)
    if not np.isfinite(probs).all() or np.any(np.bincount(patients, minlength=len(val.patient_ids)) == 0):
        raise ValueError('Invalid or missing patient predictions')
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    predictions = pd.DataFrame({'patient_id': val.patient_ids, 'label': val.patient_labels})
    rows = []
    for method in METHODS:
        scores, labels = patient_scores(probs, patients, val.patient_labels, method)
        metrics = score_metrics(scores, labels, threshold_on_validation(scores, labels))
        low, high = bootstrap_auc_ci(scores, labels, n_resamples=args.bootstrap)
        rows.append(dict(aggregation=method, **metrics, auc_ci_low=low, auc_ci_high=high))
        predictions[method] = scores
    result = pd.DataFrame(rows).sort_values('auc', ascending=False)
    result.to_csv(output/'validation_aggregation_comparison.csv', index=False)
    predictions.to_csv(output/'validation_patient_scores.csv', index=False)
    np.savez_compressed(output/'validation_slice_scores.npz', probs=probs, patient_idx=patients,
                        patient_ids=val.patient_ids, labels=val.patient_labels)
    (output/'provenance.json').write_text(json.dumps({
        'model': args.model, 'checkpoint_sha256': file_hash(args.checkpoint),
        'splits_sha256': file_hash(args.splits), 'checkpoint_epoch': notes.get('epoch'),
        'scope': 'annotation-assisted validation; thresholds selected on validation'}, indent=2))
    print(result.to_string(index=False))
    print(f'Saved: {output}. Freeze aggregation and threshold before held-out testing.')


if __name__ == '__main__':
    main()
