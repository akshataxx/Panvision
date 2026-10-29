"""Run the P3 nnU-Net pancreas model on a validated upload's LPS voxel array."""

from functools import lru_cache
import os
from pathlib import Path
from threading import Lock

import numpy as np
import SimpleITK as sitk


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (PROJECT / 'data/private/models/P3_nnunet_2d_full'
                 / 'nnUNetTrainer_PanVision__nnUNetPlans__2d')
SPACING_XYZ = (1.5, 1.5, 3.0)
WINDOW_HU = (-100, 300)
MAX_INFERENCE_VOXELS = int(os.environ.get('PANVISION_MAX_INFERENCE_VOXELS', '96000000'))
_PREDICT_LOCK = Lock()


class ModelUnavailable(RuntimeError):
    pass


class ScanTooLarge(ValueError):
    pass


def model_folder():
    configured = os.environ.get('PANVISION_PANCREAS_MODEL_DIR')
    return Path(configured).expanduser().resolve() if configured else DEFAULT_MODEL


def model_available():
    folder = model_folder()
    return all(path.is_file() for path in (folder / 'plans.json', folder / 'dataset.json',
                                         folder / 'fold_0/checkpoint_best.pth'))


@lru_cache(maxsize=1)
def _predictor():
    folder = model_folder()
    if not model_available():
        raise ModelUnavailable('Pancreas model files are missing. Configure PANVISION_PANCREAS_MODEL_DIR.')

    # The checkpoint names a custom trainer. Discover our repo copy without
    # modifying the installed nnU-Net package or requiring its training branch.
    trainer_folder = Path(__file__).parent / 'nnunet_ext'
    external = os.environ.get('nnUNet_extTrainer', '')
    paths = [path for path in external.split(os.pathsep) if path]
    if str(trainer_folder) not in paths:
        os.environ['nnUNet_extTrainer'] = os.pathsep.join([*paths, str(trainer_folder)])

    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as exc:
        raise ModelUnavailable('Install app/inference-requirements.txt for pancreas segmentation.') from exc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                                use_mirroring=False,
                                perform_everything_on_device=device.type == 'cuda',
                                device=device, verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        str(folder), use_folds=(0,), checkpoint_name='checkpoint_best.pth')
    return predictor


def prepare_volume(array, spacing_xyz):
    """Use the exact orientation, spacing and HU window used for training."""
    try:
        from src.preprocessing.pipeline import preprocess
    except ImportError as exc:
        raise ModelUnavailable('Install app/inference-requirements.txt for pancreas segmentation.') from exc

    source = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    source.SetSpacing(tuple(float(value) for value in spacing_xyz))
    prepared, _ = preprocess(source, None, SPACING_XYZ, WINDOW_HU)
    return source, prepared


def segment_pancreas(array, spacing_xyz):
    """Return a binary mask in the uploaded preview's original voxel grid."""
    if array.size > MAX_INFERENCE_VOXELS:
        raise ScanTooLarge('Scan exceeds the local pancreas inference size limit.')
    source, prepared = prepare_volume(array, spacing_xyz)
    volume = sitk.GetArrayFromImage(prepared)
    with _PREDICT_LOCK:
        predicted = _predictor().predict_single_npy_array(
            volume[None].astype(np.float32),
            {'spacing': list(prepared.GetSpacing()[::-1])})
    if predicted.shape != volume.shape:
        raise RuntimeError('Pancreas mask shape does not match the model input.')
    mask_image = sitk.GetImageFromArray((predicted > 0).astype(np.uint8))
    mask_image.CopyInformation(prepared)
    aligned = sitk.Resample(mask_image, source, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    mask = sitk.GetArrayFromImage(aligned).astype(bool)
    if mask.shape != array.shape:
        raise RuntimeError('Pancreas mask shape does not match the uploaded scan.')
    return mask
