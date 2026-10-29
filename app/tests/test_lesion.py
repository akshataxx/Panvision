"""L2 upload-grid alignment without running the full model."""

import unittest
from unittest.mock import patch

import numpy as np

from app.lesion import segment_lesion


class LesionGeometryTests(unittest.TestCase):
    def test_prediction_receives_preprocessed_pancreas_and_returns_upload_grid(self):
        class Segmentor:
            def predict(self, image, pancreas, spacing):
                assert spacing == (1.5, 1.5, 3.0)
                assert image.shape == pancreas.shape
                assert pancreas.any()
                return pancreas, None

        array = np.full((6, 16, 16), 50, dtype=np.float32)
        pancreas = np.zeros(array.shape, dtype=bool)
        pancreas[2:4, 4:12, 4:12] = True
        with patch('app.lesion._segmentor', return_value=Segmentor()):
            mask = segment_lesion(array, (1, 1, 2), pancreas)
        self.assertEqual(mask.shape, array.shape)
        self.assertTrue(mask.any())
        self.assertEqual(mask.dtype, np.bool_)

    def test_rejects_misaligned_pancreas_mask(self):
        with self.assertRaises(ValueError):
            segment_lesion(np.zeros((4,8,8)), (1,1,1), np.zeros((3,8,8)))
