import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from app.data import annotation_volume, case_index, initial_slice, load_case, rgb_slice


class DashboardDataTests(unittest.TestCase):
    def test_annotation_measurements_and_overlays(self):
        image = np.full((4, 8, 8), 0.5, dtype=np.float32)
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[2, 2:4, 2:4] = 1
        self.assertEqual(initial_slice(image, mask), 2)
        self.assertAlmostEqual(annotation_volume(mask, (2, 2, 5)), 0.08)
        self.assertIsNone(annotation_volume(None, (1, 1, 1)))
        rgb = rgb_slice(image, mask, 2, 'lesion')
        self.assertGreater(rgb[2, 2, 0], rgb[2, 2, 1])
        self.assertTrue(np.all(image == 0.5))

    def test_index_ignores_partial_and_rejects_missing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / 'processed/full/s1'
            (study / 'volumes').mkdir(parents=True)
            (root / 'processed/full/.partial-s2').mkdir()
            (root / 'processed/full/.partial-s2/complete.json').write_text('{invalid json')
            image = study / 'volumes/image.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((3, 4, 5), dtype=np.float32)), str(image))
            (study / 'complete.json').write_text(json.dumps({'study_id': 's1'}))
            pd.DataFrame([{'study_id': 's1', 'patient_id': 'p1', 'label': '1',
                           'processed_image': 'image.nii.gz', 'processed_labels': ''}]).to_csv(
                               study / 'volumes/manifest.csv', index=False)
            cases, errors = case_index(root)
            self.assertEqual(len(cases), 1)
            self.assertEqual(errors, [])
            array, mask, spacing = load_case(cases[0]['processed_image'])
            self.assertEqual(array.shape, (3, 4, 5))
            self.assertIsNone(mask)
            image.unlink()
            cases, errors = case_index(root)
            self.assertEqual(cases, [])
            self.assertEqual(len(errors), 1)

    def test_gradcam_panels_show_crop_heat_map_and_overlay(self):
        from app.data import gradcam_panels
        crop = np.full((16, 16), 128, dtype=np.uint8)
        heat = np.zeros((16, 16), dtype=np.float32)
        heat[6:10, 6:10] = 1.0
        picture, heat_map, overlay = gradcam_panels(crop, heat)
        for rendered in (picture, heat_map, overlay):
            self.assertEqual(rendered.shape, (16, 16, 3))
            self.assertEqual(rendered.dtype, np.uint8)
        self.assertGreater(heat_map[8, 8, 0], heat_map[8, 8, 2])   # hot spot is red
        self.assertGreater(heat_map[0, 0, 2], heat_map[0, 0, 0])   # cold area is blue
        self.assertGreater(overlay[8, 8, 0], overlay[0, 0, 0])
        with self.assertRaises(ValueError):
            gradcam_panels(crop, np.zeros((8, 8), dtype=np.float32))

    def test_normalized_upload_keeps_display_contrast(self):
        from app.data import upload_gradcam_slice, upload_rgb_slice, upload_pancreas_slice
        array = np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 8, 8)
        rendered = upload_rgb_slice(array, 0)
        self.assertEqual(int(rendered.min()), 0)
        self.assertEqual(int(rendered.max()), 255)
        hu = array * 400 - 100
        np.testing.assert_allclose(upload_rgb_slice(hu, 0), rendered, atol=1)
        mask = np.zeros_like(array, dtype=bool)
        mask[0, 2:4, 2:4] = True
        overlay = upload_pancreas_slice(array, mask, 0)
        self.assertGreater(overlay[2, 2, 1], rendered[2, 2, 1])
        np.testing.assert_array_equal(overlay[0, 0], rendered[0, 0])
        heatmap = np.zeros((8, 8), dtype=np.float32)
        heatmap[3:5, 3:5] = 1
        attention = upload_gradcam_slice(array, heatmap, 0)
        self.assertGreater(attention[3, 3, 0], rendered[3, 3, 0])
        np.testing.assert_array_equal(attention[0, 0], rendered[0, 0])
