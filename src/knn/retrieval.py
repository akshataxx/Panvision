"""Leakage-safe similar-case retrieval with cosine similarity."""

from pathlib import Path

import numpy as np


def _normalized_rows(embeddings):
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Embeddings must be a finite, non-empty matrix')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('Embeddings must not contain zero vectors')
    return values / norms


def build_index(embeddings, patient_ids, labels):
    """Store training-set embeddings (N x 512) with their patient IDs and labels."""
    vectors = _normalized_rows(embeddings)
    patient_ids = np.asarray(patient_ids, dtype=str)
    labels = np.asarray(labels, dtype=np.int64)
    if len(vectors) != len(patient_ids) or len(vectors) != len(labels):
        raise ValueError('Embedding index columns have different lengths')
    if len(set(patient_ids.tolist())) != len(patient_ids):
        raise ValueError('Patient identifiers must be unique')
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError('Labels must be binary')
    return {'embeddings': vectors, 'patient_ids': patient_ids, 'labels': labels}


def load_reference_bank(path):
    """Load a validated train-only patient embedding bank exported by classification."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as bank:
        required = {'embeddings', 'labels'}
        missing = required.difference(bank.files)
        if missing:
            raise ValueError(f'Embedding bank is missing: {", ".join(sorted(missing))}')
        id_key = 'ids' if 'ids' in bank.files else 'patient_ids'
        split_key = 'split' if 'split' in bank.files else 'splits'
        if id_key not in bank.files or split_key not in bank.files:
            raise ValueError('Embedding bank is missing patient identifiers or splits')
        embeddings = np.asarray(bank['embeddings'], dtype=np.float32)
        patient_ids = np.asarray(bank[id_key], dtype=str)
        labels = np.asarray(bank['labels'], dtype=np.int64)
        splits = np.asarray(bank[split_key], dtype=str)
        entity = str(np.asarray(bank['entity']).item()) if 'entity' in bank.files else 'patient'
        reference = (bool(np.asarray(bank['knn_reference']).item())
                     if 'knn_reference' in bank.files else set(splits.tolist()) == {'train'})
        dimensions = (int(np.asarray(bank['dimensions']).item())
                      if 'dimensions' in bank.files
                      else embeddings.shape[1] if embeddings.ndim == 2 else 0)
        checkpoint_sha256 = (str(np.asarray(bank['checkpoint_sha256']).item())
                             if 'checkpoint_sha256' in bank.files else None)
    if entity != 'patient' or not reference:
        raise ValueError('KNN references require a patient reference bank')
    if dimensions != 512 or embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError('KNN references require 512-dimensional embeddings')
    if set(splits.tolist()) != {'train'}:
        raise ValueError('KNN reference bank must contain training patients only')
    index = build_index(embeddings, patient_ids, labels)
    index['split'] = splits
    index['source'] = str(path)
    index['checkpoint_sha256'] = checkpoint_sha256
    return index


def top_k(index, query_embedding, k=3, exclude_patient_id=None):
    """Return the k most similar training cases as (patient_id, label, similarity).

    exclude_patient_id: the query's own ID, so a case never retrieves itself.
    """
    if k < 1:
        raise ValueError('k must be positive')
    query = np.asarray(query_embedding, dtype=np.float32)
    dimensions = index['embeddings'].shape[1]
    if query.shape != (dimensions,) or not np.isfinite(query).all():
        raise ValueError(f'Query embedding must be a finite {dimensions}-vector')
    norm = np.linalg.norm(query)
    if norm <= 0:
        raise ValueError('Query embedding must not be zero')
    similarities = index['embeddings'] @ (query / norm)
    eligible = np.ones(len(similarities), dtype=bool)
    if exclude_patient_id is not None:
        eligible &= index['patient_ids'] != str(exclude_patient_id)
    order = np.flatnonzero(eligible)
    order = order[np.argsort(-similarities[order], kind='stable')[:k]]
    return [
        {
            'patient_id': str(index['patient_ids'][row]),
            'label': int(index['labels'][row]),
            'similarity': float(similarities[row]),
        }
        for row in order
    ]
