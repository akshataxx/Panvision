"""Run the frozen DenseNet-121 PDAC classifier on crops around P3's pancreas mask."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
import SimpleITK as sitk

from app.pancreas import MAX_INFERENCE_VOXELS, ModelUnavailable, ScanTooLarge, prepare_volume


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT / 'data/private/models/densenet121_predicted'
# make_predicted_crops.py defaults used to build the training crops.
MAX_PANCREAS_SLICES = 24
WINDOW_MM = 144.0
_PREDICT_LOCK = Lock()


class NoPancreasRegion(ValueError):
    pass


def model_folder():
    configured = os.environ.get('PANVISION_CLASSIFIER_DIR')
    return Path(configured).expanduser().resolve() if configured else DEFAULT_MODEL


def model_available():
    folder = model_folder()
    return all(path.is_file() for path in (folder / 'best.pt', folder / 'frozen_evaluation_config.json'))


def frozen_config():
    """Aggregation and threshold chosen on validation patients; never re-tuned here."""
    if not model_available():
        raise ModelUnavailable('Classifier files are missing. Configure PANVISION_CLASSIFIER_DIR.')
    config = json.loads((model_folder() / 'frozen_evaluation_config.json').read_text())
    if config.get('crop_source') != 'predicted_pancreas':
        raise ModelUnavailable('Classifier was not trained on predicted-pancreas crops.')
    threshold = float(config['classification_threshold'])
    if not 0 <= threshold <= 1:
        raise ModelUnavailable('Classifier threshold is invalid.')
    return {'aggregation': str(config['classification_aggregation']), 'threshold': threshold,
            'checkpoint_sha256': str(config['checkpoint_sha256'])}


@lru_cache(maxsize=1)
def _classifier():
    config = frozen_config()
    checkpoint = model_folder() / 'best.pt'
    with checkpoint.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != config['checkpoint_sha256']:
            raise ModelUnavailable('Classifier checkpoint does not match its frozen evaluation config.')
    try:
        import torch
        from src.classification.densenet_model import load_classifier
    except ImportError as exc:
        raise ModelUnavailable('Install app/inference-requirements.txt for PDAC classification.') from exc
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    model, _ = load_classifier(checkpoint, device='cpu')
    return model, config


def classify_pdac(array, spacing_xyz, pancreas_mask):
    """Return the patient-level PDAC probability and frozen-threshold decision for one upload."""
    if array.size > MAX_INFERENCE_VOXELS:
        raise ScanTooLarge('Scan exceeds the local classification size limit.')
    if pancreas_mask.shape != array.shape:
        raise ValueError('Pancreas mask does not match the uploaded CT.')
    try:
        import torch
        from src.classification.dataset import MEAN, STD
        from src.classification.evaluation import patient_scores
        from src.classification.make_crops import crop_slice
        from src.classification.make_predicted_crops import selected_pancreas_slices
    except ImportError as exc:
        raise ModelUnavailable('Install app/inference-requirements.txt for PDAC classification.') from exc

    # Same grid as training: the preprocessed CT and the P3 mask resampled onto it.
    source, prepared = prepare_volume(array, spacing_xyz)
    pancreas_image = sitk.GetImageFromArray(pancreas_mask.astype(np.uint8))
    pancreas_image.CopyInformation(source)
    prepared_pancreas = sitk.Resample(pancreas_image, prepared, sitk.Transform(),
                                      sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    ct = sitk.GetArrayFromImage(prepared)
    pancreas = sitk.GetArrayFromImage(prepared_pancreas).astype(bool)
    slices = selected_pancreas_slices(pancreas, MAX_PANCREAS_SLICES)
    if not len(slices):
        raise NoPancreasRegion('No clear pancreas region to classify.')
    window_px = int(round(WINDOW_MM / float(prepared.GetSpacing()[0])))
    crops = np.stack([crop_slice(ct[z], pancreas[z], window_px)[0] for z in slices])

    x = torch.from_numpy(crops).float().div(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    x = (x - MEAN) / STD
    with _PREDICT_LOCK:
        model, config = _classifier()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[:, 1].numpy()
    scores, _ = patient_scores(probs, np.zeros(len(probs), dtype=int), [1], config['aggregation'])
    probability = float(scores[0])
    return {'probability': probability, 'threshold': config['threshold'],
            'pdac': probability >= config['threshold'], 'slices': int(len(slices))}
