"""Export DenseNet slice, study, and patient embeddings for KNN retrieval."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from src.classification.experiment_utils import audit_crops, file_hash


def _l2_normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError('Embeddings must be finite, non-zero row vectors')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('Embeddings must be finite, non-zero row vectors')
    return vectors / norms


def mean_l2_embeddings(slice_embeddings, group_index, n_groups):
    """Mean L2-normalized slice vectors per group, then L2-normalize each mean."""
    vectors = _l2_normalize(slice_embeddings)
    group_index = np.asarray(group_index, dtype=int)
    if group_index.shape != (len(vectors),) or n_groups < 1:
        raise ValueError('Invalid embedding group indices')
    sums = np.zeros((n_groups, vectors.shape[1]), dtype=np.float32)
    counts = np.zeros(n_groups, dtype=np.int64)
    np.add.at(sums, group_index, vectors)
    np.add.at(counts, group_index, 1)
    if np.any(group_index < 0) or np.any(group_index >= n_groups) or np.any(counts == 0):
        raise ValueError('Every embedding group must contain at least one slice')
    return _l2_normalize(sums / counts[:, None])


def save_embedding_bank(path, embeddings, ids, labels, splits, entity, reference=False):
    """Write a validated, self-describing embedding bank."""
    embeddings = _l2_normalize(embeddings)
    ids = np.asarray(ids, dtype=str)
    labels = np.asarray(labels, dtype=np.int64)
    splits = np.asarray(splits, dtype=str)
    if not (len(embeddings) == len(ids) == len(labels) == len(splits)):
        raise ValueError('Embedding bank columns have different lengths')
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f'Duplicate {entity} identifiers in embedding bank')
    if reference and set(splits.tolist()) != {'train'}:
        raise ValueError('KNN reference banks may contain training entities only')
    path = Path(path)
    temporary = path.with_name(f'.{path.stem}.tmp.npz')
    np.savez_compressed(
        temporary,
        embeddings=embeddings,
        ids=ids,
        labels=labels,
        split=splits,
        entity=np.asarray(entity),
        knn_reference=np.asarray(bool(reference)),
        dimensions=np.asarray(embeddings.shape[1]),
        aggregation=np.asarray('mean of L2-normalized slice embeddings; L2-normalized'),
    )
    os.replace(temporary, path)


def _ordered_groups(values):
    ids, lookup, index = [], {}, []
    for value in map(str, values):
        if value not in lookup:
            lookup[value] = len(ids)
            ids.append(value)
        index.append(lookup[value])
    return np.asarray(ids), np.asarray(index, dtype=int)


def validate_checkpoint_provenance(config, splits_sha256, window_mm, prediction_id):
    """Reject a checkpoint trained with incompatible crops or patient splits."""
    if config.get('crop_source') != 'predicted_pancreas':
        raise ValueError('Checkpoint was not trained on predicted-pancreas crops')
    recorded_window = config.get('window_mm')
    if recorded_window is not None and not np.isclose(float(recorded_window), window_mm):
        raise ValueError('Checkpoint and embedding crops use different crop windows')
    recorded_prediction = config.get('prediction_id')
    if recorded_prediction and recorded_prediction != prediction_id:
        raise ValueError('Checkpoint and embedding crops use different segmentation predictions')
    recorded_hash = config.get('splits_sha256')
    if recorded_hash and recorded_hash != splits_sha256:
        raise ValueError('Checkpoint and embedding export use different patient splits')


def _extract_split(model, dataset, device, batch_size, workers):
    import torch
    import torch.nn.functional as functional

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    vectors, patient_index = [], []
    model.eval()
    with torch.no_grad():
        for images, _, patients in loader:
            batch = functional.normalize(model.embedding(images.to(device)), dim=1)
            vectors.append(batch.cpu().numpy().astype(np.float32, copy=False))
            patient_index.append(patients.numpy())
    if not vectors:
        raise ValueError('No slices were available for embedding export')
    vectors = np.concatenate(vectors)
    patient_index = np.concatenate(patient_index).astype(int, copy=False)
    patient_vectors = mean_l2_embeddings(vectors, patient_index, len(dataset.patient_ids))

    study_ids, study_index = _ordered_groups(dataset.study_ids)
    study_vectors = mean_l2_embeddings(vectors, study_index, len(study_ids))
    study_labels = np.empty(len(study_ids), dtype=np.int64)
    study_patients = np.empty(len(study_ids), dtype=object)
    for group in range(len(study_ids)):
        rows = np.flatnonzero(study_index == group)
        labels = np.unique(dataset.labels[rows])
        patients = np.unique(patient_index[rows])
        if len(labels) != 1 or len(patients) != 1:
            raise ValueError(f'Inconsistent study metadata: {study_ids[group]}')
        study_labels[group] = labels[0]
        study_patients[group] = dataset.patient_ids[patients[0]]
    return {
        'patient_vectors': patient_vectors,
        'patient_ids': np.asarray(dataset.patient_ids, dtype=str),
        'patient_labels': np.asarray(dataset.patient_labels, dtype=np.int64),
        'study_vectors': study_vectors,
        'study_ids': study_ids,
        'study_labels': study_labels,
        'study_patients': np.asarray(study_patients, dtype=str),
        'n_slices': len(vectors),
    }


def _write_manifest(path, rows, fieldnames):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def export_dense_embeddings(checkpoint, crops, output, splits='data/splits.csv',
                            batch_size=32, workers=0, device='auto', overwrite=False):
    """Export train-only KNN references and separate validation/test query banks."""
    import torch

    from src.classification.dataset import CropDataset
    from src.classification.densenet_model import load_classifier

    audit, window_mm = audit_crops(crops, splits)
    if audit.crop_source.iloc[0] != 'predicted_pancreas':
        raise ValueError('Embedding export requires predicted-pancreas crops')
    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu') if device == 'auto' else device
    model, checkpoint_notes = load_classifier(checkpoint, device)
    checkpoint_config = checkpoint_notes.get('config', {})
    prediction_id = audit.prediction_id.iloc[0]
    validate_checkpoint_provenance(
        checkpoint_config, file_hash(splits), window_mm, prediction_id)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sentinel = output / 'patient_embeddings_all.npz'
    if sentinel.exists() and not overwrite:
        raise FileExistsError(f'{sentinel} already exists; use --overwrite to replace embedding files')
    metadata_path = output / 'embedding_metadata.json'
    if overwrite and metadata_path.exists():
        metadata_path.unlink()

    results = {}
    patient_rows, study_rows = [], []
    for split in ('train', 'val', 'test'):
        dataset = CropDataset(crops, split, mode='eval', augment=False)
        result = _extract_split(model, dataset, device, batch_size, workers)
        results[split] = result
        split_values = [split] * len(result['patient_ids'])
        save_embedding_bank(
            output / f'patient_embeddings_{split}.npz', result['patient_vectors'],
            result['patient_ids'], result['patient_labels'], split_values,
            entity='patient', reference=split == 'train')
        patient_rows.extend(
            {'patient_id': patient, 'split': split, 'label': int(label)}
            for patient, label in zip(result['patient_ids'], result['patient_labels']))
        study_rows.extend(
            {'study_id': study, 'patient_id': patient, 'split': split, 'label': int(label)}
            for study, patient, label in zip(
                result['study_ids'], result['study_patients'], result['study_labels']))

    patient_vectors = np.concatenate([results[s]['patient_vectors'] for s in ('train', 'val', 'test')])
    patient_ids = np.concatenate([results[s]['patient_ids'] for s in ('train', 'val', 'test')])
    patient_labels = np.concatenate([results[s]['patient_labels'] for s in ('train', 'val', 'test')])
    patient_splits = np.concatenate([[s] * len(results[s]['patient_ids']) for s in ('train', 'val', 'test')])
    save_embedding_bank(output / 'patient_embeddings_all.npz', patient_vectors, patient_ids,
                        patient_labels, patient_splits, entity='patient')

    study_vectors = np.concatenate([results[s]['study_vectors'] for s in ('train', 'val', 'test')])
    study_ids = np.concatenate([results[s]['study_ids'] for s in ('train', 'val', 'test')])
    study_labels = np.concatenate([results[s]['study_labels'] for s in ('train', 'val', 'test')])
    study_splits = np.concatenate([[s] * len(results[s]['study_ids']) for s in ('train', 'val', 'test')])
    save_embedding_bank(output / 'study_embeddings_all.npz', study_vectors, study_ids,
                        study_labels, study_splits, entity='study')

    _write_manifest(output / 'patient_embedding_manifest.csv', patient_rows,
                    ['patient_id', 'split', 'label'])
    _write_manifest(output / 'study_embedding_manifest.csv', study_rows,
                    ['study_id', 'patient_id', 'split', 'label'])
    metadata = {
        'model': 'densenet121_512',
        'checkpoint': str(Path(checkpoint)),
        'checkpoint_sha256': file_hash(checkpoint),
        'checkpoint_epoch': checkpoint_notes.get('epoch'),
        'crop_source': 'predicted_pancreas',
        'prediction_id': prediction_id,
        'crop_window_mm': window_mm,
        'splits_sha256': file_hash(splits),
        'dimensions': int(patient_vectors.shape[1]),
        'aggregation': 'mean of L2-normalized slice embeddings; L2-normalized',
        'knn_reference_file': 'patient_embeddings_train.npz',
        'query_files': ['patient_embeddings_val.npz', 'patient_embeddings_test.npz'],
        'counts': {split: {
            'patients': len(results[split]['patient_ids']),
            'studies': len(results[split]['study_ids']),
            'slices': results[split]['n_slices'],
        } for split in ('train', 'val', 'test')},
    }
    metadata_temporary = output / '.embedding_metadata.json.tmp'
    metadata_temporary.write_text(json.dumps(metadata, indent=2))
    os.replace(metadata_temporary, metadata_path)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--crops', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--splits', default='data/splits.csv')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('batch-size must be positive and workers cannot be negative')
    metadata = export_dense_embeddings(
        args.checkpoint, args.crops, args.out, args.splits,
        args.batch_size, args.workers, args.device, args.overwrite)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
