"""Checks for the isolated lesion experiment path."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.segmentation import train_experiments
from src.segmentation.experimental_data import augment
from src.segmentation.tests.fake_data import make_root


class LesionExperimentTests(unittest.TestCase):
    def test_strong_augmentation_preserves_binary_mask(self):
        image = np.full((3, 32, 32), 0.5, np.float32)
        target = np.zeros((32, 32), np.uint8)
        target[10:20, 10:20] = 1
        changed, mask = augment(image, target, np.random.default_rng(7), 'strong')
        self.assertEqual(changed.shape, image.shape)
        self.assertTrue(np.isfinite(changed).all())
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))
        self.assertGreater(mask.sum(), 0)

    def test_false_positive_penalty_and_settings(self):
        args = train_experiments.parse_args(
            ['--stage', 'lesion', '--run', 'trial', '--selection-fp-weight', '0.2'])
        a = {'val_dice': 0.35, 'val_false_positive': 0.8}
        b = {'val_dice': 0.34, 'val_false_positive': 0.4}
        self.assertGreater(train_experiments.selection_score(args, b),
                           train_experiments.selection_score(args, a))
        self.assertTrue(train_experiments.same_training_settings(vars(args), vars(args) | {'stop_after': 500}))
        self.assertFalse(train_experiments.same_training_settings(
            vars(args), vars(args) | {'lesion_negative_fraction': 0.5}))

    def test_experimental_train_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, splits = make_root(tmp)
            command = ['--stage', 'lesion', '--model', 'attention_unet', '--run', 'trial',
                       '--data-root', str(root), '--splits', str(splits), '--patch', '32',
                       '--batch', '2', '--workers', '0', '--iters', '4', '--val-every', '2',
                       '--lesion-negative-fraction', '0.5', '--lesion-augmentation', 'strong',
                       '--selection-fp-weight', '0.1', '--cache-dir', str(Path(tmp) / 'cache')]
            train_experiments.train(train_experiments.parse_args(command + ['--stop-after', '2']))
            train_experiments.train(train_experiments.parse_args(command))
            run = root / 'models/segmentation/trial'
            self.assertTrue((run / 'best.pt').exists())
            self.assertEqual(pd.read_csv(run / 'log.csv').step.tolist(), [2, 4])


if __name__ == '__main__':
    unittest.main()
