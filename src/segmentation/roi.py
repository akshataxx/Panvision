"""Pancreas ROI: largest component, bounding box with a physical margin, fixed-size crop and paste-back."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage


def largest_component(mask):
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def remove_small(mask, min_voxels):
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask.astype(bool)
    sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, count + 1))
    return np.isin(labels, np.flatnonzero(sizes >= min_voxels) + 1)


@dataclass
class ROI:
    """Slices z0:z1 of a (size x size) in-plane window centred on (cy, cx); voxel coordinates, zyx order."""
    z0: int
    z1: int
    cy: int
    cx: int
    size: int

    def to_dict(self):
        return asdict(self)


def make_roi(mask, spacing_xyz, size=160, margin_mm=10.0):
    """ROI around a binary (Z, Y, X) mask, or None when the mask is empty."""
    if not mask.any():
        return None
    zs, ys, xs = np.nonzero(mask)
    margin_z = int(np.ceil(margin_mm / spacing_xyz[2]))
    return ROI(z0=max(0, int(zs.min()) - margin_z), z1=min(mask.shape[0], int(zs.max()) + 1 + margin_z),
               cy=(int(ys.min()) + int(ys.max()) + 1) // 2, cx=(int(xs.min()) + int(xs.max()) + 1) // 2,
               size=size)


def _window(center, size, length):
    """Source range in the volume and destination range in the crop for one axis."""
    start = center - size // 2
    src0, src1 = max(0, start), min(length, start + size)
    return src0, src1, src0 - start, src1 - start


def crop(volume, roi, fill=0):
    """(z1-z0, size, size) crop; parts outside the volume are filled with `fill`."""
    out = np.full((roi.z1 - roi.z0, roi.size, roi.size), fill, dtype=volume.dtype)
    ys0, ys1, yd0, yd1 = _window(roi.cy, roi.size, volume.shape[1])
    xs0, xs1, xd0, xd1 = _window(roi.cx, roi.size, volume.shape[2])
    out[:, yd0:yd1, xd0:xd1] = volume[roi.z0:roi.z1, ys0:ys1, xs0:xs1]
    return out


def paste(patch, roi, shape, fill=0):
    """Inverse of crop: place a crop back into a zero volume of `shape`."""
    out = np.full(shape, fill, dtype=patch.dtype)
    ys0, ys1, yd0, yd1 = _window(roi.cy, roi.size, shape[1])
    xs0, xs1, xd0, xd1 = _window(roi.cx, roi.size, shape[2])
    out[roi.z0:roi.z1, ys0:ys1, xs0:xs1] = patch[:, yd0:yd1, xd0:xd1]
    return out
