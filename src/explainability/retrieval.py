"""Top-3 similar-case retrieval from 512-number "fingerprints" (embeddings).

Plain-language idea
-------------------
The classifier turns every patient into a list of 512 numbers (a fingerprint).
Two patients whose fingerprints point in a similar direction look similar to the
model. "Cosine similarity" measures that: 1.0 = same direction, 0 = unrelated.
To explain a new case we show the 3 most similar *training* patients.

This file uses only numpy (no torch), so it works the same for the stand-in model
and for the real classifier later.

Rules the team agreed / we are following
----------------------------------------
* The search database holds TRAINING patients only.
* A query patient never retrieves themselves (all of their studies are excluded).
* One fingerprint per patient (the average of their slice/study fingerprints).
"""
import numpy as np


def _normalise(x):
    """Scale each vector to length 1, so a dot product equals cosine similarity."""
    x = np.asarray(x, dtype=np.float32)
    length = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(length, 1e-12)  # the max() avoids dividing by zero


def aggregate_by_patient(embeddings, patient_ids, labels):
    """Average the fingerprints of all rows that belong to the same patient.

    embeddings: (N, 512) one row per study (or per slice).
    Returns (patient_embeddings, unique_patient_ids, patient_labels).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    patient_ids = np.asarray(patient_ids)
    labels = np.asarray(labels)
    unique_ids = sorted(set(patient_ids.tolist()))
    out_emb, out_lab = [], []
    for pid in unique_ids:
        rows = patient_ids == pid
        if len(set(labels[rows].tolist())) != 1:
            raise ValueError(f"Patient {pid} has conflicting labels: {set(labels[rows])}")
        out_emb.append(embeddings[rows].mean(axis=0))
        out_lab.append(labels[rows][0])
    return np.stack(out_emb), np.array(unique_ids), np.array(out_lab)


def build_index(embeddings, patient_ids, labels):
    """Store training-set embeddings (N x 512) with their patient IDs and labels.

    Call this with TRAINING patients only (one row per patient).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D (N, 512), got shape {embeddings.shape}")
    if not (len(embeddings) == len(patient_ids) == len(labels)):
        raise ValueError("embeddings, patient_ids and labels must have the same length")
    return {
        "vectors": _normalise(embeddings),  # length-1 copies, ready for cosine similarity
        "patient_ids": np.asarray(patient_ids),
        "labels": np.asarray(labels),
    }


def top_k(index, query_embedding, k=3, exclude_patient_id=None):
    """Return the k most similar training cases as (patient_id, label, similarity).

    exclude_patient_id: the query's own ID, so a case never retrieves itself.
    """
    query = _normalise(np.asarray(query_embedding, dtype=np.float32).reshape(-1))
    similarity = index["vectors"] @ query  # cosine similarity to every stored patient
    if exclude_patient_id is not None:
        similarity = np.where(index["patient_ids"] == exclude_patient_id, -np.inf, similarity)
    order = np.argsort(-similarity)  # best first
    results = []
    for i in order[:k]:
        if np.isneginf(similarity[i]):
            break  # only excluded rows are left
        results.append((str(index["patient_ids"][i]), int(index["labels"][i]), float(similarity[i])))
    return results


def precision_at_k(index, k=3):
    """Leave-one-patient-out check: how often are the Top-k neighbours the same class?

    Every patient in the index takes a turn as the query (excluded from its own
    search). Precision@k = share of the k neighbours that have the query's label.

    The random baseline is what you would get by picking k patients blindly: for each
    query it is the share of the OTHER patients that have the same label. A model
    only means something if it beats this baseline.
    """
    labels = index["labels"]
    hits, baselines, is_pdac = [], [], []
    for i, pid in enumerate(index["patient_ids"]):
        found = top_k(index, index["vectors"][i], k=k, exclude_patient_id=pid)
        if not found:
            continue
        hits.append(np.mean([lab == labels[i] for _, lab, _ in found]))
        others = np.delete(labels, i)
        baselines.append(np.mean(others == labels[i]))
        is_pdac.append(labels[i] == 1)
    hits, baselines, is_pdac = np.array(hits), np.array(baselines), np.array(is_pdac)
    return {
        "n_queries": int(len(hits)),
        "precision": float(hits.mean()),
        "random_baseline": float(baselines.mean()),
        "pdac_precision": float(hits[is_pdac].mean()) if is_pdac.any() else None,
        "pdac_random_baseline": float(baselines[is_pdac].mean()) if is_pdac.any() else None,
    }

def precision_at_k_queries(index, query_embeddings, query_labels, k=3):
    """Score retrieval for patients who are NOT in the database (e.g. validation patients).

    index:            built from TRAINING patients only (see build_index)
    query_embeddings: (Q, 512) one fingerprint per query patient
    query_labels:     (Q,) 0/1 label of every query patient
    Precision@k = share of the k neighbours that have the same label as the query.

    The random baseline for a query is the share of database patients with the query's
    own label: that is what you would get by picking k patients blindly.
    Returns overall numbers and PDAC-only numbers (PDAC is the class that matters most).
    """
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    query_labels = np.asarray(query_labels)
    db_labels = index["labels"]
    hits, baselines, any_hit = [], [], []
    for vector, label in zip(query_embeddings, query_labels):
        found = top_k(index, vector, k=k)
        same = [lab == label for _, lab, _ in found]
        hits.append(np.mean(same))
        any_hit.append(float(any(same)))
        baselines.append(np.mean(db_labels == label))
    hits, baselines, any_hit = np.array(hits), np.array(baselines), np.array(any_hit)
    pdac = query_labels == 1
    return {
        "n_queries": int(len(hits)),
        "precision": float(hits.mean()),
        "random_baseline": float(baselines.mean()),
        "at_least_one_same": float(any_hit.mean()),
        "pdac_precision": float(hits[pdac].mean()) if pdac.any() else None,
        "pdac_random_baseline": float(baselines[pdac].mean()) if pdac.any() else None,
    }