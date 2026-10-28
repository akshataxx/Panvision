"""Lesion measurements from predicted masks using physical voxel spacing (model-derived estimates)."""

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _max_distance(points):
    if len(points) < 2:
        return 0.0
    if len(points) > 3:
        try:
            points = points[ConvexHull(points).vertices]
        except QhullError:  # collinear points
            pass
    diffs = points[:, None, :] - points[None, :, :]
    return float(np.sqrt((diffs ** 2).sum(-1)).max())


def max_axial_diameter_mm(mask, spacing_xyz):
    """Largest in-plane extent over all axial slices, measured between voxel centres plus one voxel."""
    best = 0.0
    for z in np.flatnonzero(mask.any(axis=(1, 2))):
        ys, xs = np.nonzero(mask[z])
        points = np.stack([xs * spacing_xyz[0], ys * spacing_xyz[1]], axis=1)
        best = max(best, _max_distance(points) + spacing_xyz[0])
    return best


def location(lesion, pancreas, spacing_xyz):
    """head/body/tail by thirds along the pancreas principal axis.

    Arrays are LPS-oriented (x index increases towards patient left); the head lies on the patient's right,
    so the axis is oriented from low x (head) to high x (tail).
    """
    if not lesion.any() or not pancreas.any():
        return 'unknown'
    scale = np.array(spacing_xyz[::-1])  # zyx
    gland = np.argwhere(pancreas) * scale
    mean = gland.mean(axis=0)
    axis = np.linalg.svd(gland - mean, full_matrices=False)[2][0]
    if axis[2] < 0:
        axis = -axis
    projected = (gland - mean) @ axis
    position = ((np.argwhere(lesion) * scale).mean(axis=0) - mean) @ axis
    low, high = np.quantile(projected, [1 / 3, 2 / 3])
    return 'head' if position < low else 'body' if position < high else 'tail'


def lesion_measurements(lesion, pancreas, spacing_xyz):
    lesion = lesion.astype(bool)
    count = int(lesion.sum())
    return {'lesion_detected': count > 0, 'lesion_voxel_count': count,
            'lesion_volume_cm3': count * float(np.prod(spacing_xyz)) / 1000.0,
            'lesion_max_axial_diameter_mm': max_axial_diameter_mm(lesion, spacing_xyz),
            'lesion_location': location(lesion, pancreas.astype(bool), spacing_xyz)}
