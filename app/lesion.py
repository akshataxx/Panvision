"""Run the trained L2 lesion model using P3's pancreas mask as its ROI."""

from functools import lru_cache
import os
from pathlib import Path
from threading import Lock

import numpy as np
import SimpleITK as sitk

from app.pancreas import MAX_INFERENCE_VOXELS, ModelUnavailable, ScanTooLarge, prepare_volume


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT / 'data/private/models/L2_attention_unet_roi_full/best.pt'
_PREDICT_LOCK = Lock()


def model_file():
    configured = os.environ.get('PANVISION_LESION_CHECKPOINT')
    return Path(configured).expanduser().resolve() if configured else DEFAULT_MODEL


def model_available():
    return model_file().is_file()


@lru_cache(maxsize=1)
def _segmentor():
    if not model_available():
        raise ModelUnavailable('Lesion model files are missing. Configure PANVISION_LESION_CHECKPOINT.')
    try:
        from src.segmentation.infer import LesionSegmentor
    except ImportError as exc:
        raise ModelUnavailable('Install app/inference-requirements.txt for lesion segmentation.') from exc
    return LesionSegmentor(model_file())


def segment_lesion(array, spacing_xyz, pancreas_mask):
    """Return an L2 binary mask on the uploaded CT grid, via the training preprocessing grid."""
    if array.size > MAX_INFERENCE_VOXELS:
        raise ScanTooLarge('Scan exceeds the local lesion inference size limit.')
    if pancreas_mask.shape != array.shape:
        raise ValueError('Pancreas mask does not match the uploaded CT.')
    source, prepared = prepare_volume(array, spacing_xyz)
    pancreas_image = sitk.GetImageFromArray(pancreas_mask.astype(np.uint8))
    pancreas_image.CopyInformation(source)
    prepared_pancreas = sitk.Resample(pancreas_image, prepared, sitk.Transform(),
                                      sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    with _PREDICT_LOCK:
        predicted, _ = _segmentor().predict(
            sitk.GetArrayFromImage(prepared),
            sitk.GetArrayFromImage(prepared_pancreas).astype(bool),
            prepared.GetSpacing(),
        )
    if predicted.shape != sitk.GetArrayFromImage(prepared).shape:
        raise RuntimeError('Lesion mask shape does not match the model input.')
    lesion_image = sitk.GetImageFromArray(predicted.astype(np.uint8))
    lesion_image.CopyInformation(prepared)
    aligned = sitk.Resample(lesion_image, source, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    mask = sitk.GetArrayFromImage(aligned).astype(bool)
    if mask.shape != array.shape:
        raise RuntimeError('Lesion mask shape does not match the uploaded scan.')
    return mask
