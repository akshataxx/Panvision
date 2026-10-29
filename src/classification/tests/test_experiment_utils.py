import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.classification.experiment_utils import audit_crops, score_metrics, threshold_on_validation


class ExperimentTests(unittest.TestCase):
    def test_metrics_match_confusion_counts(self):
        result = score_metrics([.1, .8, .7, .9], [0, 0, 1, 1], .5)
        self.assertEqual([result[k] for k in ['tn', 'fp', 'fn', 'tp']], [1, 1, 0, 2])
        self.assertEqual(result['sensitivity'], 1)
        self.assertEqual(result['specificity'], .5)

    def test_threshold_is_finite_for_constant_scores(self):
        self.assertTrue(np.isfinite(threshold_on_validation([.2, .2], [0, 1])))
        with self.assertRaises(ValueError):
            threshold_on_validation([.2, .3], [1, 1])

    def test_audit_rejects_split_and_label_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root/'splits.csv'
            split.write_text('patient_id,split,label\n100000,train,1\n')
            crop = root/'100000_00001.npz'
            values = dict(patient_id='100000', study_id='100000_00001', split='train', label=1,
                          crops=np.zeros((2,224,224), np.uint8), has_lesion=np.array([True,False]),
                          slice_idx=np.array([0,1]), window_mm=144.)
            np.savez(crop, **values)
            frame, window = audit_crops(root, split)
            self.assertEqual(len(frame), 1)
            self.assertEqual(window, 144)
            self.assertEqual(frame.crop_source.tolist(), ['annotation_assisted'])
            for field, bad in [('split','test'), ('label',0)]:
                with self.subTest(field=field):
                    np.savez(crop, **{**values, field:bad})
                    with self.assertRaisesRegex(ValueError, 'mismatch'):
                        audit_crops(root, split)

    def test_audit_rejects_unknown_crop_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / 'splits.csv'
            split.write_text('patient_id,split,label\n100000,train,1\n')
            np.savez(root / '100000_00001.npz', patient_id='100000',
                     study_id='100000_00001', split='train', label=1,
                     crops=np.zeros((1, 224, 224), np.uint8),
                     has_lesion=np.array([False]), slice_idx=np.array([0]),
                     window_mm=144., crop_source='predicted_pancrease')
            with self.assertRaisesRegex(ValueError, 'Unknown crop source'):
                audit_crops(root, split)
