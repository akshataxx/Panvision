import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.segmentation import tune
from src.segmentation.tests.fake_data import make_root


class _Pancreas:
    def predict_case(self, case):
        return case['labels'] > 0


class _Lesion:
    def __init__(self, checkpoint):
        self.calls = 0

    def predict_probabilities(self, image, pancreas, spacing):
        self.calls += 1
        probs = np.zeros_like(image)
        probs[image > 0.45] = 0.65  # healthy pancreas triggers a false positive at 0.5
        probs[(image > 0.3) & (image < 0.4)] = 0.9  # lesion survives at 0.8
        return probs, None


class TuningTests(unittest.TestCase):
    def test_sweep_scores_both_classes_and_runs_inference_once_per_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, splits = make_root(temporary)
            output = Path(temporary) / 'tuning.csv'
            args = argparse.Namespace(checkpoint='unused.pt', pancreas_checkpoint=None,
                                      pancreas_predictions='unused', thresholds=[0.5, 0.8],
                                      min_voxels=[1], data_root=str(root), splits=str(splits), out=str(output))
            model = _Lesion('unused.pt')
            with patch.object(tune, 'LesionSegmentor', return_value=model), \
                    patch.object(tune, 'pancreas_source', return_value=_Pancreas()):
                tune.run(args)
            summary = pd.read_csv(output).set_index('threshold')
            self.assertEqual(model.calls, 2)
            self.assertEqual(summary.loc[0.5, 'pdac_cases'], 1)
            self.assertEqual(summary.loc[0.5, 'non_pdac_cases'], 1)
            self.assertEqual(summary.loc[0.5, 'false_positive_rate'], 1.0)
            self.assertEqual(summary.loc[0.8, 'false_positive_rate'], 0.0)
            self.assertGreater(summary.loc[0.8, 'dice'], 0)


if __name__ == '__main__':
    unittest.main()
