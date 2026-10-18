import unittest

import numpy as np
import pandas as pd

from src.segmentation.measure import lesion_measurements, location
from src.segmentation.metrics import overlap, paired_wilcoxon, summarize
from src.segmentation.roi import crop, largest_component, make_roi, paste, remove_small


class RoiTests(unittest.TestCase):
    def test_crop_paste_round_trip_at_border(self):
        volume = np.arange(10 * 20 * 30, dtype=np.float32).reshape(10, 20, 30)
        mask = np.zeros(volume.shape, bool)
        mask[4:6, 0:3, 25:30] = True  # touches the image edge, so the window must be padded
        roi = make_roi(mask, (1.5, 1.5, 3.0), size=16, margin_mm=3.0)
        self.assertEqual((roi.z0, roi.z1), (3, 7))
        patch = crop(volume, roi)
        self.assertEqual(patch.shape, (4, 16, 16))
        restored = paste(patch, roi, volume.shape)
        np.testing.assert_array_equal(restored[mask], volume[mask])

    def test_empty_mask_has_no_roi(self):
        self.assertIsNone(make_roi(np.zeros((4, 8, 8), bool), (1, 1, 1)))

    def test_component_filters(self):
        mask = np.zeros((5, 10, 10), bool)
        mask[1:4, 1:5, 1:5] = True
        mask[0, 8, 8] = True
        self.assertEqual(largest_component(mask).sum(), 48)
        self.assertEqual(remove_small(mask, 2).sum(), 48)


class MetricTests(unittest.TestCase):
    def test_overlap(self):
        a = np.zeros((2, 4, 4), bool)
        b = a.copy()
        a[0, :2] = True   # 8 voxels
        b[0, 1:3] = True  # 8 voxels, 4 shared
        result = overlap(a, b)
        self.assertAlmostEqual(result['dice'], 0.5)
        self.assertAlmostEqual(result['iou'], 1 / 3)
        self.assertAlmostEqual(result['recall'], 0.5)
        self.assertEqual(overlap(np.zeros(3), np.zeros(3))['dice'], 1.0)

    def test_summary_and_wilcoxon(self):
        a = pd.DataFrame({'study_id': list('abcdef'), 'dice': [0.8, 0.7, 0.9, 0.6, 0.75, 0.85]})
        b = a.assign(dice=a.dice - 0.1)
        summary = summarize(a, ['dice'])
        self.assertAlmostEqual(summary.loc[0, 'mean'], a.dice.mean())
        self.assertLessEqual(summary.loc[0, 'ci_low'], summary.loc[0, 'ci_high'])
        self.assertAlmostEqual(paired_wilcoxon(a, b, 'dice')['mean_diff'], 0.1)


class MeasureTests(unittest.TestCase):
    def test_sphere(self):
        spacing = (1.5, 1.5, 3.0)
        z, y, x = np.indices((20, 40, 40))
        sphere = ((z - 10) * 3) ** 2 + ((y - 20) * 1.5) ** 2 + ((x - 20) * 1.5) ** 2 <= 10 ** 2
        result = lesion_measurements(sphere, sphere, spacing)
        self.assertAlmostEqual(result['lesion_volume_cm3'], 4 / 3 * np.pi * 1.0, delta=0.4)
        self.assertAlmostEqual(result['lesion_max_axial_diameter_mm'], 20, delta=2.5)
        self.assertTrue(result['lesion_detected'])

    def test_head_body_tail_along_patient_left(self):
        pancreas = np.zeros((6, 10, 60), bool)
        pancreas[2:4, 4:6, :] = True
        for x, expected in ((5, 'head'), (30, 'body'), (55, 'tail')):
            lesion = np.zeros_like(pancreas)
            lesion[2:4, 4:6, x - 2:x + 2] = True
            self.assertEqual(location(lesion, pancreas, (1, 1, 1)), expected)
        self.assertEqual(location(np.zeros_like(pancreas), pancreas, (1, 1, 1)), 'unknown')


if __name__ == '__main__':
    unittest.main()
