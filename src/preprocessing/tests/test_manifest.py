import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing.build_manifest import build_manifest


class ManifestTests(unittest.TestCase):
    def test_metadata_join_and_manual_preference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'annotations/manual_labels').mkdir(parents=True)
            (root / 'annotations/automatic_labels').mkdir()
            (root / 'images/00123_00001_0000.nii.gz').touch()
            for tier in ('manual', 'automatic'):
                (root / f'annotations/{tier}_labels/00123_00001.nii.gz').touch()
            metadata = pd.DataFrame([
                dict(PANORAMA_patient_id='00123', PANORAMA_study_id='00123_00001',
                     label='PDAC', level='pathology'),
                dict(PANORAMA_patient_id='00999', PANORAMA_study_id='00999_00001',
                     label='non-PDAC', level='NIH_dataset'),
            ])
            metadata.to_excel(root / 'clinical.xlsx', index=False)
            frame, missing = build_manifest(root / 'images', root / 'annotations',
                                            root / 'clinical.xlsx')
            self.assertEqual(missing, 1)
            self.assertEqual(frame.iloc[0].patient_id, '00123')
            self.assertEqual(frame.iloc[0].label, '1')
            self.assertEqual(frame.iloc[0].annotation_tier, 'manual')
            (root / 'images/unknown_0000.nii.gz').touch()
            with self.assertRaisesRegex(ValueError, 'No clinical metadata'):
                build_manifest(root / 'images', root / 'annotations', root / 'clinical.xlsx')
