"""Pancreas integration geometry without loading a model checkpoint."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from app.pancreas import ScanTooLarge, prepare_volume, segment_pancreas


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Install app/inference-requirements.txt')
class PancreasGeometryTests(unittest.TestCase):
    def test_preprocessing_matches_training_spacing_and_window(self):
        array = np.full((4, 8, 8), 50, dtype=np.float32)
        source, prepared = prepare_volume(array, (1, 1, 2))
        self.assertEqual(source.GetSpacing(), (1.0, 1.0, 2.0))
        self.assertEqual(prepared.GetSpacing(), (1.5, 1.5, 3.0))
        self.assertAlmostEqual(float(np.median(sitk.GetArrayFromImage(prepared))), 0.375)

    def test_prediction_returns_mask_in_uploaded_grid(self):
        class Predictor:
            def predict_single_npy_array(self, image, properties):
                self_spacing = properties['spacing']
                assert self_spacing == [3.0, 1.5, 1.5]
                return np.ones(image.shape[1:], dtype=np.uint8)

        array = np.full((4, 8, 8), 50, dtype=np.float32)
        with patch('app.pancreas._predictor', return_value=Predictor()):
            mask = segment_pancreas(array, (1, 1, 2))
        self.assertEqual(mask.shape, array.shape)
        self.assertTrue(mask.any())
        self.assertEqual(mask.dtype, np.bool_)

    def test_large_scan_is_rejected_before_loading_model(self):
        huge = np.broadcast_to(np.array(0, dtype=np.float32), (2, 2, 24_000_001))
        with patch('app.pancreas._predictor') as predictor:
            with self.assertRaises(ScanTooLarge):
                segment_pancreas(huge, (1, 1, 1))
            predictor.assert_not_called()

    def test_standard_high_resolution_scan_reaches_preprocessing(self):
        scan = np.broadcast_to(np.array(0, dtype=np.float32), (60, 768, 768))
        with patch('app.pancreas.prepare_volume', side_effect=RuntimeError('entered preprocessing')):
            with self.assertRaisesRegex(RuntimeError, 'entered preprocessing'):
                segment_pancreas(scan, (1, 1, 1))
