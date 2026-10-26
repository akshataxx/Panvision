"""Everything the dashboard needs to explain one case, in one function.

    from explain import load_index, explain_study
    model, notes = load_model("models/classifier.pt")
    index = load_index("outputs/embeddings.npz")
    result = explain_study(model, index, "outputs/crops/100000_00001.npz", notes["threshold"])

`result` is a plain dictionary:
    patient_id, score (0-1, average chance of PDAC over the slices), threshold,
    predicted_pdac (True/False),
    key_slices: the most suspicious slices, each with slice_index, probability, crop,
                heatmap and overlay (arrays ready to show),
    similar_cases: the 3 most similar TRAINING patients (patient_id, label, similarity).

It works for any classifier that provides embedding(), forward() and gradcam_layer
(see model_loader.py). For now it reads a study's saved crops; once the segmentation
step is connected, the crops can come from a newly uploaded scan instead.
"""
import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image

from model_loader import to_model_input  # first: it also makes src/classification importable
from evaluation import patient_scores  # from src/classification
from gradcam import generate_heatmap
from retrieval import aggregate_by_patient, build_index, top_k


def load_index(embeddings_path):
    """Search database from outputs/embeddings.npz: TRAINING patients only."""
    data = np.load(embeddings_path)
    train = data["splits"] == "train"
    emb, ids, labels = aggregate_by_patient(
        data["embeddings"][train], data["patient_ids"][train], data["labels"][train])
    return build_index(emb, ids, labels)


def score_slices(model, crops, device="cpu", batch_size=64):
    """Chance of PDAC and the 512-number fingerprint for every saved slice."""
    probs, embeddings = [], []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            x = to_model_input(crops[start:start + batch_size]).to(device)
            embeddings.append(model.embedding(x).cpu().numpy())
            probs.append(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs), np.concatenate(embeddings)


def explain_study(model, index, npz_path, threshold, device="cpu", top_n=3, k=3, aggregation="mean"):
    with np.load(npz_path) as f:
        crops, slice_idx, patient_id = f["crops"], f["slice_idx"], str(f["patient_id"])

    probs, embeddings = score_slices(model, crops, device)
    # Same rule as the scoring on the test patients. The threshold must have been chosen
    # on validation with this SAME rule (the checkpoint's "threshold" is for "mean").
    score = float(patient_scores(probs, np.zeros(len(probs), dtype=int), [0], aggregation)[0][0])

    key_slices = []
    for i in np.argsort(-probs)[:top_n]:  # the most suspicious slices
        x = to_model_input(crops[i:i + 1]).to(device)
        heat = generate_heatmap(model, x, model.gradcam_layer, target_class=1)
        gray = crops[i].astype(np.float32) / 255.0
        overlay = show_cam_on_image(np.stack([gray] * 3, axis=-1), heat, use_rgb=True)
        key_slices.append({"slice_index": int(slice_idx[i]), "probability": float(probs[i]),
                           "crop": crops[i], "heatmap": heat, "overlay": overlay})

    # Similar training patients. A patient never finds themselves.
    similar = top_k(index, embeddings.mean(axis=0), k=k, exclude_patient_id=patient_id)
    return {
        "patient_id": patient_id,
        "score": score,
        "threshold": float(threshold),
        "predicted_pdac": bool(score >= threshold),
        "key_slices": key_slices,
        "similar_cases": [{"patient_id": p, "label": int(l), "similarity": float(s)} for p, l, s in similar],
    }
