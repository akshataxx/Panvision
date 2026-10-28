"""Slice-selection policy shared by crop-based classifiers."""

import numpy as np

CROP_SOURCES = {'annotation_assisted', 'predicted_pancreas'}


def select_rows(has_lesion, label, mode, drop_unlocalised_pdac=False,
                crop_source='annotation_assisted'):
    """Return crop rows allowed for a training or evaluation sample.

    Predicted-pancreas crops never use lesion metadata. Annotation-assisted PDAC
    training may use known lesion slices; evaluation always uses all crop rows.
    """
    has_lesion = np.asarray(has_lesion, dtype=bool)
    if crop_source not in CROP_SOURCES:
        raise ValueError(f'Unknown crop source: {crop_source}')
    if crop_source == 'predicted_pancreas':
        return np.arange(len(has_lesion))
    if mode == 'train' and label == 1:
        if has_lesion.any():
            return np.where(has_lesion)[0]
        if drop_unlocalised_pdac:
            return np.array([], dtype=int)
    return np.arange(len(has_lesion))
