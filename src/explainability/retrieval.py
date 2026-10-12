"""Top-3 similar-case retrieval using 512-d embeddings and cosine similarity.

TODO (explainability owner): implement with scikit-learn.
"""


def build_index(embeddings, patient_ids, labels):
    """Store training-set embeddings (N x 512) with their patient IDs and labels."""
    raise NotImplementedError("Not implemented yet")


def top_k(index, query_embedding, k=3, exclude_patient_id=None):
    """Return the k most similar training cases as (patient_id, label, similarity).

    exclude_patient_id: the query's own ID, so a case never retrieves itself.
    """
    raise NotImplementedError("Not implemented yet")
