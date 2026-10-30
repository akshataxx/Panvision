"""Frozen DenseNet inference for one uploaded CT and its predicted pancreas mask."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
import SimpleITK as sitk

from app.pancreas import SPACING_XYZ, WINDOW_HU
from PIL import Image

from src.classification.make_crops import crop_slice, window_box
from src.classification.make_predicted_crops import selected_pancreas_slices


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT / 'data/private/models/classification/best.pt'
DEFAULT_METADATA = PROJECT / 'data/private/models/classification/embedding_metadata.json'
MODEL_KEY = 'densenet121_512'
DEFAULT_THRESHOLD = 0.4195498807063755
WINDOW_MM = 144.0
MAX_PANCREAS_SLICES = 24
_MODEL_LOCK = Lock()


class ClassifierUnavailable(RuntimeError):
    pass


def checkpoint_file():
    configured = os.environ.get('PANVISION_CLASSIFIER_CHECKPOINT')
    if configured:
        return Path(configured).expanduser().resolve()
    folder = os.environ.get('PANVISION_CLASSIFIER_DIR')
    return Path(folder).expanduser().resolve() / 'best.pt' if folder else DEFAULT_CHECKPOINT


def metadata_file():
    configured = os.environ.get('PANVISION_CLASSIFIER_METADATA')
    if configured:
        return Path(configured).expanduser().resolve()
    folder = os.environ.get('PANVISION_CLASSIFIER_DIR')
    if not folder:
        return DEFAULT_METADATA
    root = Path(folder).expanduser().resolve()
    embedding_metadata = root / 'embedding_metadata.json'
    return embedding_metadata if embedding_metadata.is_file() else root / 'frozen_evaluation_config.json'


def model_available():
    return checkpoint_file().is_file() and metadata_file().is_file()


@lru_cache(maxsize=1)
def model_configuration():
    if not model_available():
        raise ClassifierUnavailable('Configure the DenseNet checkpoint and embedding metadata.')
    metadata = json.loads(metadata_file().read_text())
    checkpoint_hash = hashlib.sha256(checkpoint_file().read_bytes()).hexdigest()
    if metadata.get('checkpoint_sha256') != checkpoint_hash:
        raise ClassifierUnavailable('Classifier checkpoint does not match its embedding metadata.')
    dimensions = metadata.get('dimensions', metadata.get('embedding_dimensions', 0))
    if int(dimensions) != 512:
        raise ClassifierUnavailable('Classifier metadata does not declare a 512-D embedding.')
    threshold = float(os.environ.get(
        'PANVISION_CLASSIFIER_THRESHOLD', metadata.get('classification_threshold', DEFAULT_THRESHOLD)))
    if not 0 < threshold < 1:
        raise ClassifierUnavailable('Classifier threshold must be between zero and one.')
    return {'model_key': str(metadata.get('model') or MODEL_KEY),
            'checkpoint_sha256': checkpoint_hash, 'threshold': threshold}


def _prepare_classifier_inputs(array, spacing_xyz, pancreas_mask):
    """Return the training-policy crops plus their prepared-image geometry."""
    array = np.asarray(array, dtype=np.float32)
    pancreas_mask = np.asarray(pancreas_mask, dtype=bool)
    if array.ndim != 3 or pancreas_mask.shape != array.shape:
        raise ValueError('Pancreas mask must match the uploaded 3D CT.')
    source = sitk.GetImageFromArray(array)
    source.SetSpacing(tuple(float(value) for value in spacing_xyz))
    mask_image = sitk.GetImageFromArray(pancreas_mask.astype(np.uint8))
    mask_image.CopyInformation(source)
    from src.preprocessing.pipeline import preprocess
    prepared, prepared_mask = preprocess(source, mask_image, SPACING_XYZ, WINDOW_HU)
    ct = sitk.GetArrayFromImage(prepared)
    mask = sitk.GetArrayFromImage(prepared_mask).astype(bool)
    slices = selected_pancreas_slices(mask, MAX_PANCREAS_SLICES)
    if not len(slices):
        raise ValueError('The pancreas prediction is empty; classification cannot run.')
    window_px = int(round(WINDOW_MM / float(prepared.GetSpacing()[0])))
    boxes = [window_box(mask[z], window_px)[:4] for z in slices]
    crops = [crop_slice(ct[z], mask[z], window_px)[0] for z in slices]
    return prepared, np.stack(crops), np.asarray(slices, dtype=int), boxes


def prepare_classifier_crops(array, spacing_xyz, pancreas_mask):
    """Recreate the predicted-pancreas crop policy used to train DenseNet."""
    _, crops, slices, _ = _prepare_classifier_inputs(array, spacing_xyz, pancreas_mask)
    return crops, slices


def _project_gradcam(array, spacing_xyz, prepared, slices, boxes, prepared_slice, heatmap):
    source = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    source.SetSpacing(tuple(float(value) for value in spacing_xyz))
    matches = np.where(slices == int(prepared_slice))[0]
    if not len(matches):
        raise ValueError('Grad-CAM slice is not part of the classifier crop set.')
    r0, r1, c0, c1 = boxes[int(matches[0])]
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.shape != (224, 224) or not np.isfinite(heatmap).all():
        raise ValueError('Grad-CAM output has invalid dimensions or values.')
    resized = np.asarray(Image.fromarray(np.clip(heatmap, 0, 1), mode='F').resize(
        (c1 - c0, r1 - r0), Image.Resampling.BILINEAR), dtype=np.float32)
    prepared_heat = np.zeros(tuple(reversed(prepared.GetSize())), dtype=np.float32)
    prepared_heat[int(prepared_slice), r0:r1, c0:c1] = resized
    heat_image = sitk.GetImageFromArray(prepared_heat)
    heat_image.CopyInformation(prepared)
    restored = sitk.Resample(heat_image, source, sitk.Transform(), sitk.sitkLinear,
                             0.0, sitk.sitkFloat32)
    restored_array = sitk.GetArrayFromImage(restored)
    source_slice = int(restored_array.sum(axis=(1, 2)).argmax())
    return np.clip(restored_array[source_slice], 0, 1), source_slice


def project_gradcam_to_source(array, spacing_xyz, pancreas_mask, prepared_slice, heatmap):
    """Map a crop-space Grad-CAM back to the matching original CT slice."""
    prepared, _, slices, boxes = _prepare_classifier_inputs(array, spacing_xyz, pancreas_mask)
    return _project_gradcam(array, spacing_xyz, prepared, slices, boxes,
                            prepared_slice, heatmap)


def aggregate_predictions(probabilities, embeddings, threshold):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if probabilities.ndim != 1 or embeddings.shape != (len(probabilities), 512):
        raise ValueError('Classifier outputs have invalid dimensions.')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(embeddings).all():
        raise ValueError('Classifier produced an invalid embedding.')
    patient_embedding = (embeddings / norms).mean(axis=0)
    patient_embedding /= np.linalg.norm(patient_embedding)
    probability = float(probabilities.mean())
    return {'pdac_probability': probability,
            'predicted_class': 'PDAC' if probability >= threshold else 'Non-PDAC',
            'threshold': float(threshold), 'embedding': patient_embedding.astype(np.float32)}


@lru_cache(maxsize=1)
def _classifier():
    try:
        import torch
        from src.classification.densenet_model import load_classifier
    except ImportError as exc:
        raise ClassifierUnavailable('Install app/inference-requirements.txt for DenseNet inference.') from exc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, checkpoint = load_classifier(checkpoint_file(), device)
    config = checkpoint.get('config', {})
    crop_source = config.get('crop_source')
    if crop_source and crop_source != 'predicted_pancreas':
        raise ClassifierUnavailable('Classifier checkpoint was not trained on predicted-pancreas crops.')
    recorded_window = config.get('window_mm')
    if recorded_window is not None and not np.isclose(float(recorded_window), WINDOW_MM):
        raise ClassifierUnavailable('Classifier checkpoint uses a different crop window.')
    return model, device


def classify_scan(array, spacing_xyz, pancreas_mask):
    """Return patient PDAC score, embedding, and Grad-CAM for one uploaded scan."""
    try:
        import torch
        from src.classification.dataset import MEAN, STD
        from src.explainability.gradcam import generate_heatmap
    except ImportError as exc:
        raise ClassifierUnavailable('Install app/inference-requirements.txt for DenseNet inference.') from exc
    config = model_configuration()
    prepared, crops, slices, boxes = _prepare_classifier_inputs(array, spacing_xyz, pancreas_mask)
    tensor = torch.from_numpy(crops).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
    tensor = (tensor - MEAN) / STD
    model, device = _classifier()
    with _MODEL_LOCK:
        with torch.no_grad():
            batch = tensor.to(device)
            probabilities = torch.softmax(model(batch), dim=1)[:, 1]
            embeddings = model.embedding(batch)
        hottest = int(probabilities.argmax().item())
        heatmap = generate_heatmap(model, tensor[hottest:hottest + 1].to(device),
                                   model.gradcam_layer, target_class=1)
    projected, source_slice = _project_gradcam(
        array, spacing_xyz, prepared, slices, boxes, int(slices[hottest]), heatmap)
    result = aggregate_predictions(probabilities.cpu().numpy(), embeddings.cpu().numpy(),
                                   config['threshold'])
    result.update({'gradcam': projected, 'gradcam_slice': source_slice,
                   # The same 224 x 224 crop the model saw, its heat map, and that slice's own chance of PDAC
                   # (shown as: crop | heat map | overlay).
                   'gradcam_crop': np.asarray(crops[hottest], dtype=np.uint8),
                   'gradcam_crop_heat': np.asarray(heatmap, dtype=np.float32),
                   'gradcam_probability': float(probabilities[hottest].item()), **config})
    return result

