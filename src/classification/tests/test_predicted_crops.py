import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.classification.make_predicted_crops import build_predicted_crops


def write_image(path, array, spacing=(1.5, 1.5, 3.0), origin=(0.0, 0.0, 0.0)):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    sitk.WriteImage(image, str(path))


class PredictedCropTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.studies = self.root / 'studies'
        self.predictions = self.root / 'predictions'
        self.output = self.root / 'crops'
        self.study = self.studies / '100001_00001'
        volume = self.study / 'volumes' / 'train' / self.study.name
        volume.mkdir(parents=True)
        (self.study / 'complete.json').write_text('{}\n')
        ct = np.linspace(0, 1, 5 * 64 * 64, dtype=np.float32).reshape(5, 64, 64)
        write_image(volume / 'image.nii.gz', ct)
        mask_folder = self.predictions / self.study.name
        mask_folder.mkdir(parents=True)
        mask = np.zeros_like(ct, dtype=np.uint8)
        mask[1:4, 20:40, 18:42] = 1
        write_image(mask_folder / 'pancreas.nii.gz', mask)
        self.splits = self.root / 'splits.csv'
        with self.splits.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['patient_id', 'split', 'label'])
            writer.writeheader()
            writer.writerow({'patient_id': '100001', 'split': 'train', 'label': '1'})

    def tearDown(self):
        self.temp.cleanup()

    def test_builds_compatible_crops_without_lesion_assistance(self):
        counts = build_predicted_crops(
            self.studies, self.predictions, 'P2_train_only', self.splits, self.output)
        self.assertEqual(counts['saved'], 1)
        with np.load(self.output / '100001_00001.npz', allow_pickle=False) as item:
            self.assertEqual(item['crops'].shape, (3, 224, 224))
            self.assertEqual(item['crops'].dtype, np.uint8)
            self.assertEqual(item['slice_idx'].tolist(), [1, 2, 3])
            self.assertFalse(item['has_lesion'].any())
            self.assertFalse(bool(item['lesion_available']))
            self.assertEqual(str(item['crop_source']), 'predicted_pancreas')
            self.assertEqual(str(item['prediction_id']), 'P2_train_only')

    def test_rejects_mask_on_a_different_grid(self):
        path = self.predictions / self.study.name / 'pancreas.nii.gz'
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        write_image(path, mask, origin=(1.0, 0.0, 0.0))
        counts = build_predicted_crops(
            self.studies, self.predictions, 'P2_train_only', self.splits, self.output)
        self.assertEqual(counts['grid_mismatch'], 1)
        self.assertFalse((self.output / '100001_00001.npz').exists())

    def test_refuses_to_mix_changed_predictions(self):
        build_predicted_crops(
            self.studies, self.predictions, 'P2_train_only', self.splits, self.output)
        path = self.predictions / self.study.name / 'pancreas.nii.gz'
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        mask[0, 10:20, 10:20] = 1
        write_image(path, mask)
        with self.assertRaisesRegex(ValueError, 'different prediction'):
            build_predicted_crops(
                self.studies, self.predictions, 'P2_train_only', self.splits, self.output)


if __name__ == '__main__':
    unittest.main()
