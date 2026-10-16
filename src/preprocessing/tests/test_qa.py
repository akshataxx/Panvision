import unittest

import numpy as np

from src.preprocessing.qa import render, select_slices


class QATests(unittest.TestCase):
    def test_selection_includes_largest_lesion_and_preserves_input(self):
        labels = np.zeros((10, 8, 8), dtype=np.uint8)
        labels[2:8, 2:6, 2:6] = 4
        labels[5, 3:5, 3:5] = 1
        original = labels.copy()
        slices = select_slices(labels, 10)
        self.assertIn(5, slices)
        montage = render(np.full(labels.shape, 100), labels, slices)
        self.assertEqual(montage.shape, (8 * len(slices), 16, 3))
        self.assertEqual(montage.dtype, np.uint8)
        np.testing.assert_array_equal(labels, original)
        self.assertTrue(np.any(montage[:, 8:, 0] != montage[:, 8:, 1]))
