"""End-to-end research inference for one validated uploaded CT."""

import numpy as np

from app.classification import classify_scan
from app.lesion import segment_lesion
from app.pancreas import segment_pancreas
from src.knn.neon_embeddings import query_neon


def query_similar_cases(embedding, model_key, checkpoint_sha256, k=3):
    from app import database as db
    with db.connect() as connection:
        return query_neon(connection, embedding, model_key, checkpoint_sha256, k=k)


def analyze_upload(array, spacing_xyz):
    """Run P3, DenseNet, Grad-CAM and train-only KNN retrieval."""
    pancreas = segment_pancreas(array, spacing_xyz)
    lesion = segment_lesion(array, spacing_xyz, pancreas)
    classification = classify_scan(array, spacing_xyz, pancreas)
    neighbors = query_similar_cases(
        classification['embedding'], classification['model_key'],
        classification['checkpoint_sha256'], k=3,
    )
    result = {key: value for key, value in classification.items() if key != 'embedding'}
    result.update({'pancreas_mask': np.asarray(pancreas, dtype=bool),
                   'lesion_mask': np.asarray(lesion, dtype=bool),
                   'neighbors': neighbors})
    return result
