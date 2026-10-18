import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.segmentation import evaluate, infer, train
from src.segmentation.data import (LesionSlices, PancreasSlices, find_studies, lesion_crop, lesion_rows,
                                   load_cases, pancreas_band)
from src.segmentation.tests.fake_data import make_root


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root, self.splits = make_root(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_discovery_and_lesion_supervision(self):
        studies = find_studies(self.root, self.splits)
        self.assertEqual(len(studies), 7)
        self.assertTrue(Path(studies.processed_image.iloc[0]).exists())
        positives, negatives = lesion_rows(studies, 'train')
        self.assertEqual(sorted(positives.study_id), ['100_1', '101_1'])  # automatic lesion mask excluded
        self.assertEqual(negatives.study_id.tolist(), ['103_1'])

    def test_interrupted_copy_is_ignored(self):
        import shutil
        shutil.copytree(self.root / 'processed/full/100_1', self.root / 'processed/full/.partial-100_1')
        self.assertEqual(len(find_studies(self.root, self.splits)), 7)

    def test_split_disagreement_fails(self):
        frame = pd.read_csv(self.splits, dtype=str)
        frame.loc[frame.patient_id == '100', 'split'] = 'test'
        frame.to_csv(self.splits, index=False)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            find_studies(self.root, self.splits)

    def test_dataset_shapes(self):
        studies = find_studies(self.root, self.splits)
        positives, negatives = lesion_rows(studies, 'train')
        bands = load_cases(positives, transform=pancreas_band)
        image, target = PancreasSlices(bands, 4, patch=32, context=3)[0]
        self.assertEqual((tuple(image.shape), tuple(target.shape)), ((3, 32, 32), (1, 32, 32)))
        crops = load_cases(positives, transform=lambda c: lesion_crop(c, size=32, jitter=4))
        negative_crops = load_cases(negatives, transform=lambda c: lesion_crop(c, size=32, jitter=4))
        dataset = LesionSlices(crops, negative_crops, 8, size=32, jitter=4, context=3)
        self.assertEqual(tuple(dataset[3][0].shape), (3, 32, 32))
        self.assertTrue(any(dataset[i][1].sum() > 0 for i in range(8)))


class SmokeTests(unittest.TestCase):
    """Train -> resume -> evaluate -> export on synthetic data (a few steps; checks wiring, not accuracy)."""

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, splits = make_root(tmp)
            common = ['--data-root', str(root), '--splits', str(splits), '--patch', '32', '--batch', '2',
                      '--workers', '0', '--val-every', '2', '--cache-dir', str(Path(tmp) / 'cache')]
            for stage, run, extra in (('pancreas', 'P', ['--context', '3']), ('lesion', 'L', ['--model', 'attention_unet']),
                                      ('lesion', 'L0', ['--roi', 'none'])):
                command = ['--stage', stage, '--run', run, '--iters', '4'] + common + extra
                train.train(train.parse_args(command + ['--stop-after', '2']))
                train.train(train.parse_args(command))
                log = pd.read_csv(root / f'models/segmentation/{run}/log.csv')
                self.assertEqual(log.step.tolist(), [2, 4])  # second call resumed rather than restarting
            models = root / 'models/segmentation'
            base = ['--data-root', str(root), '--splits', str(splits)]
            evaluate.main(['pancreas', '--checkpoint', str(models / 'P/best.pt')] + base)
            evaluate.main(['lesion', '--checkpoint', str(models / 'L/best.pt'),
                           '--pancreas-checkpoint', str(models / 'P/best.pt')] + base)
            cases = pd.read_csv(root / 'outputs/segmentation/results/L_val_cases.csv')
            self.assertEqual(sorted(cases['mode'].unique()), ['end_to_end', 'oracle'])
            evaluate.main(['lesion', '--checkpoint', str(models / 'L0/best.pt')] + base)
            self.assertFalse(pd.read_csv(root / 'outputs/segmentation/results/L0_val_cases.csv').roi_found.any())
            infer.main(['--pancreas-checkpoint', str(models / 'P/best.pt'),
                        '--lesion-checkpoint', str(models / 'L/best.pt'), '--split', 'test'] + base)
            out = root / 'outputs/segmentation'
            self.assertTrue((out / '106_1/pancreas.nii.gz').exists())
            measurements = pd.read_csv(out / 'measurements.csv')
            self.assertEqual(measurements.study_id.tolist(), ['106_1'])
            self.assertTrue(np.isfinite(measurements.lesion_volume_cm3).all())


class MaskFolderAndNnunetExportTests(unittest.TestCase):
    """nnU-Net (P3) path without nnU-Net installed: export layout and scoring of saved prediction folders."""

    def test_export_and_folder_evaluation(self):
        import json
        import SimpleITK as sitk
        from src.segmentation import nnunet_export
        with tempfile.TemporaryDirectory() as tmp:
            root, splits = make_root(tmp)
            base = ['--data-root', str(root), '--splits', str(splits)]
            raw, pre, inputs = Path(tmp) / 'raw', Path(tmp) / 'pre', Path(tmp) / 'inputs'
            nnunet_export.main(['dataset', '--raw', str(raw)] + base)
            nnunet_export.main(['splits', '--raw', str(raw), '--preprocessed', str(pre)] + base)
            dataset = raw / nnunet_export.DATASET
            self.assertEqual(len(list((dataset / 'imagesTr').glob('*_0000.nii.gz'))), 6)
            label = sitk.GetArrayFromImage(sitk.ReadImage(str(dataset / 'labelsTr/100_1.nii.gz')))
            self.assertEqual(set(np.unique(label)), {0, 1})  # lesion (1) folded into the pancreas target
            split = json.loads((pre / nnunet_export.DATASET / 'splits_final.json').read_text())[0]
            self.assertEqual(split['val'], ['104_1', '105_1'])
            nnunet_export.main(['inputs', '--out', str(inputs)] + base)
            self.assertEqual(sorted(p.name for p in inputs.iterdir()),
                             ['104_1_0000.nii.gz', '105_1_0000.nii.gz', '106_1_0000.nii.gz'])
            # Perfect "predictions" (the reference labels) must score Dice 1 through the folder path.
            predictions = Path(tmp) / 'pred'
            predictions.mkdir()
            for study in ('104_1', '105_1'):
                image = next((root / 'processed/full' / study).rglob('labels.nii.gz'))
                labels = sitk.ReadImage(str(image))
                mask = sitk.GetImageFromArray((sitk.GetArrayFromImage(labels) > 0).astype(np.uint8))
                mask.CopyInformation(labels)
                sitk.WriteImage(mask, str(predictions / f'{study}.nii.gz'))
            evaluate.main(['pancreas', '--predictions', str(predictions), '--name', 'P3', '--split', 'val'] + base)
            cases = pd.read_csv(root / 'outputs/segmentation/results/P3_val_cases.csv')
            self.assertTrue(np.allclose(cases.dice, 1.0))


if __name__ == '__main__':
    unittest.main()
