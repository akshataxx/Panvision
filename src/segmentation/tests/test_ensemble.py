import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.segmentation import ensemble
from src.segmentation.roi import ROI
from src.segmentation.tests.fake_data import make_root


class _Pancreas:
    def predict_case(self, case):
        return case['labels'] > 0


class _Lesion:
    def __init__(self, lesion_probability, background_probability):
        self.lesion_probability = lesion_probability
        self.background_probability = background_probability
        self.calls = 0

    def predict_probabilities(self, image, pancreas, spacing):
        self.calls += 1
        probabilities = np.zeros_like(image)
        probabilities[image > 0.45] = self.background_probability
        probabilities[(image > 0.3) & (image < 0.4)] = self.lesion_probability
        return probabilities, None


class EnsembleTests(unittest.TestCase):
    def test_average_requires_aligned_rois(self):
        a = np.ones((2, 4, 4), np.float32)
        b = np.zeros_like(a)
        roi = ROI(0, 2, 2, 2, 4)
        probabilities, result_roi = ensemble.average_probabilities((a, roi), (b, roi))
        self.assertTrue(np.all(probabilities == 0.5))
        self.assertEqual(result_roi, roi)
        with self.assertRaisesRegex(ValueError, 'different pancreas ROIs'):
            ensemble.average_probabilities((a, roi), (b, ROI(0, 2, 3, 2, 4)))

    def test_validation_averages_once_per_model_and_records_measurement_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, splits = make_root(temporary)
            output = Path(temporary) / 'ensemble.csv'
            args = argparse.Namespace(first='first.pt', second='second.pt', pancreas_checkpoint=None,
                                      pancreas_predictions='unused', thresholds=[0.5, 0.8],
                                      min_voxels=[1], data_root=str(root), splits=str(splits), out=str(output))
            first, second = _Lesion(0.9, 0.7), _Lesion(0.9, 0.1)
            with patch.object(ensemble, 'LesionSegmentor', side_effect=[first, second]), \
                    patch.object(ensemble, 'pancreas_source', return_value=_Pancreas()):
                ensemble.run(args)
            all_results = pd.read_csv(output)
            self.assertEqual(set(all_results.model), {'L2', 'L5', 'L2_L5_ensemble'})
            summary = all_results[all_results.model == 'L2_L5_ensemble'].set_index('threshold')
            self.assertEqual((first.calls, second.calls), (2, 2))
            self.assertEqual(summary.loc[0.5, 'false_positive_rate'], 0)
            self.assertEqual(summary.loc[0.5, 'detection_rate'], 1)
            self.assertIn('diameter_abs_error_mm', summary.columns)
            self.assertTrue((output.with_name('ensemble_cases.csv')).exists())


if __name__ == '__main__':
    unittest.main()
