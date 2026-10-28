"""Read completed preprocessing outputs without modifying the shared dataset."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


PROJECT = Path(__file__).resolve().parents[1]


def default_root():
    if os.environ.get('PANVISION_DATA_DIR'):
        return Path(os.environ['PANVISION_DATA_DIR']).expanduser()
    config = PROJECT / 'data/storage.json'
    if config.exists():
        return Path(json.loads(config.read_text())['data_root']).expanduser()
    return PROJECT / 'data'


def case_index(root):
    """Only expose completed full-run studies, not partial or older duplicates."""
    root = Path(root)
    cases, errors = [], []
    for receipt in sorted((root / 'processed/full').glob('*/complete.json')):
        if receipt.parent.name.startswith('.'):
            continue
        manifest = receipt.parent / 'volumes/manifest.csv'
        try:
            completion = json.loads(receipt.read_text())
            frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
            if len(frame) != 1 or frame.iloc[0].study_id != completion['study_id']:
                raise ValueError('Study ID does not match its completion receipt')
            row = frame.iloc[0].to_dict()
            for field in ('processed_image', 'processed_labels'):
                value = row.get(field, '')
                target = (manifest.parent / value).resolve() if value else None
                if target is not None and not target.is_relative_to(receipt.parent.resolve()):
                    raise ValueError('Output path escapes study directory')
                if target is not None and not target.is_file():
                    raise ValueError(f'Missing {field}')
                row[field] = str(target) if target else ''
            if row.get('label') not in ('0', '1'):
                raise ValueError('Unrecognized dataset label')
            row['diagnosis'] = 'PDAC' if row['label'] == '1' else 'Non-PDAC'
            cases.append(row)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            errors.append(f'{receipt.parent.name}: {exc}')
    return cases, errors


def load_case(image_path, labels_path=''):
    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    if image.GetDimension() != 3:
        raise ValueError('Expected a 3D volume')
    array = sitk.GetArrayFromImage(image)
    if not np.isfinite(array).all():
        raise ValueError('Image contains non-finite values')
    labels = None
    if labels_path:
        mask = sitk.ReadImage(str(labels_path))
        if mask.GetDimension() != 3 or mask.GetSize() != image.GetSize() or any(
            not np.allclose(getattr(image, key)(), getattr(mask, key)(), atol=1e-4, rtol=0)
            for key in ('GetSpacing', 'GetOrigin', 'GetDirection')
        ):
            raise ValueError('Image and annotation geometry do not match')
        labels = sitk.GetArrayFromImage(mask)
        if not np.isin(np.unique(labels), range(7)).all():
            raise ValueError('Unexpected annotation label values')
    return array, labels, image.GetSpacing()


def rgb_slice(array, labels, z, mode='ct', opacity=0.45, hu=False):
    gray = np.clip((array[z] + 100) / 400 if hu else array[z], 0, 1)
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
    if labels is not None and mode != 'ct':
        values = [(4, (75, 216, 135))] if mode == 'pancreas' else [(1, (250, 101, 113))]
        for value, color in values:
            region = labels[z] == value
            rgb[region] = ((1-opacity)*rgb[region] + opacity*np.array(color)).astype(np.uint8)
    return rgb


def initial_slice(array, labels):
    if labels is not None:
        for value in (1, 4):
            areas = (labels == value).sum(axis=(1, 2))
            if areas.max() > 0:
                return int(areas.argmax())
    return len(array) // 2


def annotation_volume(labels, spacing):
    return None if labels is None else float((labels == 1).sum() * np.prod(spacing) / 1000)


def upload_display_mode(array):
    """A display heuristic only; a narrow range does not establish CT calibration."""
    return 'normalized' if float(array.min()) >= 0 and float(array.max()) <= 1 else 'hu'


def upload_rgb_slice(array, z, display_mode='auto'):
    if display_mode == 'auto':
        display_mode = upload_display_mode(array)
    if display_mode not in ('hu', 'normalized'):
        raise ValueError('Unknown display mode')
    return rgb_slice(array, None, z, hu=display_mode == 'hu')


def upload_pancreas_slice(array, mask, z, display_mode='auto', opacity=0.45):
    """Overlay a model pancreas mask on the same uploaded CT voxel grid."""
    if mask.shape != array.shape:
        raise ValueError('Pancreas mask does not align with the uploaded CT.')
    rgb = upload_rgb_slice(array, z, display_mode)
    region = mask[z].astype(bool)
    rgb[region] = ((1 - opacity) * rgb[region] +
                   opacity * np.array([75, 216, 135])).astype(np.uint8)
    return rgb


def upload_lesion_slice(array, mask, z, display_mode='auto', opacity=0.55):
    """Overlay a predicted lesion mask in coral on the uploaded CT grid."""
    if mask.shape != array.shape:
        raise ValueError('Lesion mask does not align with the uploaded CT.')
    rgb = upload_rgb_slice(array, z, display_mode)
    region = mask[z].astype(bool)
    rgb[region] = ((1 - opacity) * rgb[region] +
                   opacity * np.array([250, 101, 113])).astype(np.uint8)
    return rgb
