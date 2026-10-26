"""Bounded CT validation shared by the UI preview and backend upload endpoint."""
from pathlib import Path
import numpy as np
import SimpleITK as sitk

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_VOXELS = 128 * 1024 * 1024


def read_ct(path):
    path = Path(path)
    name = path.name.lower()
    if not name.endswith(('.nii', '.nii.gz', '.mha')):
        raise ValueError('Use a .nii, .nii.gz, or self-contained .mha volume.')
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError('CT upload exceeds 512 MiB.')
    if name.endswith('.mha'):
        with path.open('rb') as source:
            header = source.read(65536).split(b'ElementDataFile', 1)
        if len(header) != 2 or header[1].splitlines()[0].strip() != b'= LOCAL':
            raise ValueError('MHA must contain LOCAL image data, not an external file reference.')
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.SetImageIO('MetaImageIO' if name.endswith('.mha') else 'NiftiImageIO')
    reader.ReadImageInformation()
    size = reader.GetSize()
    if reader.GetDimension() != 3 or min(size) < 2 or int(np.prod(size, dtype=object)) > MAX_VOXELS:
        raise ValueError('Expected a nonempty 3D CT with at most 128 million voxels.')
    if not np.isfinite(reader.GetOrigin()).all() or not np.isfinite(reader.GetDirection()).all():
        raise ValueError('Invalid CT origin or orientation.')
    spacing = reader.GetSpacing()
    if not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError('Invalid CT spacing.')
    reader.SetOutputPixelType(sitk.sitkFloat32)
    image = sitk.DICOMOrient(reader.Execute(), 'LPS')
    array = sitk.GetArrayFromImage(image)
    if not np.isfinite(array).all():
        raise ValueError('CT contains non-finite intensities.')
    return array, image.GetSpacing()
