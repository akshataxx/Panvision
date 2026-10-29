"""What the dashboard needs to explain one case: the patient score and the key slices with Grad-CAM.

    from explain import explain_study
    model, notes = load_model("models/classifier.pt")
    result = explain_study(model, "outputs/crops/100000_00001.npz", threshold=notes["threshold"])

`result` is a plain dictionary:
    patient_id, score (0-1, patient-level chance of PDAC), threshold,
    predicted_pdac (True/False),
    key_slices: the most suspicious slices, each with slice_index, probability, crop,
                heatmap and overlay (arrays ready to show).

Similar cases are NOT in here: the app gets them from its own KNN (src/knn/). The offline
retrieval evaluation for the report lives in retrieval_eval.py / evaluate_retrieval.py.

It works for any classifier that provides embedding(), forward() and gradcam_layer
(see model_loader.py). For now it reads a study's saved crops.
"""
import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image

from model_loader import to_model_input  # first: it also puts the repo root on the path
from src.classification.evaluation import patient_scores
from gradcam import generate_heatmap


def score_slices(model, crops, device="cpu", batch_size=64):
    """Chance of PDAC and the 512-number fingerprint for every saved slice."""
    probs, embeddings = [], []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            x = to_model_input(crops[start:start + batch_size]).to(device)
            embeddings.append(model.embedding(x).cpu().numpy())
            probs.append(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs), np.concatenate(embeddings)


def explain_study(model, npz_path, threshold, device="cpu", top_n=3, aggregation="mean"):
    with np.load(npz_path) as f:
        crops, slice_idx, patient_id = f["crops"], f["slice_idx"], str(f["patient_id"])

    probs, _ = score_slices(model, crops, device)
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

    return {
        "patient_id": patient_id,
        "score": score,
        "threshold": float(threshold),
        "predicted_pdac": bool(score >= threshold),
        "key_slices": key_slices,
    }
