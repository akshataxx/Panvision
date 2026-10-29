"""PDAC classifier crop geometry, frozen decision rule and checkpoint guard."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.classification import NoPancreasRegion, classify_pdac, frozen_config, model_available

INFERENCE = all(importlib.util.find_spec(name) for name in ('torch', 'torchvision', 'sklearn'))


def scan_with_pancreas():
    array = np.full((12, 64, 64), 40, dtype=np.float32)
    pancreas = np.zeros(array.shape, dtype=bool)
    pancreas[3:9, 20:44, 16:48] = True
    return array, pancreas


@unittest.skipUnless(INFERENCE, 'Install app/inference-requirements.txt')
class ClassificationTests(unittest.TestCase):
    def test_crops_match_training_and_use_frozen_rule(self):
        import torch

        seen = {}

        class Model(torch.nn.Module):
            def forward(self, x):
                seen['shape'] = tuple(x.shape)
                # Slice probabilities 0.2 and 0.6 alternate; their mean is 0.4.
                p = torch.tensor([0.2, 0.6]).repeat(len(x))[:len(x)]
                return torch.stack([torch.log(1 - p), torch.log(p)], dim=1)

        array, pancreas = scan_with_pancreas()
        config = {'aggregation': 'mean', 'threshold': 0.35, 'checkpoint_sha256': 'x'}
        with patch('app.classification._classifier', return_value=(Model(), config)):
            result = classify_pdac(array, (1, 1, 2), pancreas)
        self.assertEqual(seen['shape'][1:], (3, 224, 224))
        self.assertEqual(seen['shape'][0], result['slices'])
        self.assertAlmostEqual(result['probability'], 0.4, places=5)
        self.assertEqual(result['threshold'], 0.35)
        self.assertTrue(result['pdac'])

    def test_empty_or_misaligned_pancreas_is_rejected(self):
        array, pancreas = scan_with_pancreas()
        with self.assertRaises(NoPancreasRegion):
            classify_pdac(array, (1, 1, 2), np.zeros_like(pancreas))
        with self.assertRaises(ValueError):
            classify_pdac(array, (1, 1, 2), pancreas[:-1])

    @unittest.skipUnless(model_available(), 'Classifier checkpoint is not installed locally')
    def test_real_checkpoint_loads_and_returns_probability(self):
        array, pancreas = scan_with_pancreas()
        result = classify_pdac(array, (1, 1, 2), pancreas)
        self.assertTrue(0 <= result['probability'] <= 1)
        self.assertEqual(result['pdac'], result['probability'] >= result['threshold'])


class FrozenConfigTests(unittest.TestCase):
    def write_model(self, folder, **overrides):
        config = {'crop_source': 'predicted_pancreas', 'classification_aggregation': 'mean',
                  'classification_threshold': 0.42, 'checkpoint_sha256': 'abc', **overrides}
        (Path(folder) / 'frozen_evaluation_config.json').write_text(json.dumps(config))
        (Path(folder) / 'best.pt').write_bytes(b'not a checkpoint')

    def test_reads_threshold_and_rejects_wrong_crop_source(self):
        from app.pancreas import ModelUnavailable
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'PANVISION_CLASSIFIER_DIR': folder}):
            self.write_model(folder)
            self.assertEqual(frozen_config()['threshold'], 0.42)
            self.write_model(folder, crop_source='annotation_assisted')
            with self.assertRaises(ModelUnavailable):
                frozen_config()

    def test_checkpoint_must_match_frozen_hash(self):
        from app.classification import _classifier
        from app.pancreas import ModelUnavailable
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'PANVISION_CLASSIFIER_DIR': folder}):
            self.write_model(folder)
            _classifier.cache_clear()
            self.addCleanup(_classifier.cache_clear)
            with self.assertRaises(ModelUnavailable):
                _classifier()
