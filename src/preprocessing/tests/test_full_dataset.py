import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.preprocessing.full_dataset import completed, run
from src.preprocessing.pipeline import patient_splits


class FullDatasetTests(unittest.TestCase):
    def test_resume_and_low_disk_with_full_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'raw/images').mkdir(parents=True)
            (root / 'raw/annotations/automatic_labels').mkdir(parents=True)
            clinical, siblings = [], []
            for i in range(20):
                study = f'{i:06d}_00001'
                clinical.append({'PANORAMA_patient_id': f'{i:06d}',
                                 'PANORAMA_study_id': study,
                                 'label': 'PDAC' if i % 2 else 'non-PDAC', 'level': 'radiology'})
                image = sitk.GetImageFromArray(np.full((4, 4, 4), 100, dtype=np.int16))
                image.SetSpacing((1.5, 1.5, 3))
                mask = sitk.GetImageFromArray(np.full((4, 4, 4), 4, dtype=np.uint8))
                mask.CopyInformation(image)
                name = f'images/{study}_0000.nii.gz'
                sitk.WriteImage(image, str(root / 'raw' / name))
                sitk.WriteImage(mask, str(root / 'raw/annotations/automatic_labels' / f'{study}.nii.gz'))
                siblings.append(NS(rfilename=name, size=(root / 'raw' / name).stat().st_size))
            pd.DataFrame(clinical).to_excel(root / 'raw/annotations/clinical_information.xlsx', index=False)
            cohort = pd.DataFrame({'patient_id': [r['PANORAMA_patient_id'] for r in clinical],
                                   'label': [str(i % 2) for i in range(20)]})
            split_path = root / 'splits.csv'
            patient_splits(cohort).to_csv(split_path, index=False)
            api = NS(dataset_info=lambda *a, **k: NS(sha='test_revision', siblings=siblings))
            downloads = []
            download = lambda **kwargs: downloads.append(kwargs['filename'])
            args = NS(data_root=root, splits=split_path, run=False, reserve_gib=10,
                      batch_size=2, max_batches=1)
            self.assertEqual(run(args, api, download), 0)
            self.assertEqual(downloads, [])
            args.run = True
            self.assertEqual(run(args, api, download), 0)
            self.assertEqual(len(downloads), 2)
            self.assertEqual(run(args, api, download), 0)
            self.assertEqual(len(downloads), 4)
            self.assertEqual(len(set(downloads)), 4)
            final = root / 'processed/full/000000_00001'
            receipt = json.loads((final / 'complete.json').read_text())
            self.assertTrue(completed(final, receipt['signature']))
            output_manifest = pd.read_csv(final / 'volumes/manifest.csv')
            image_path = output_manifest.iloc[0].processed_image
            self.assertFalse(Path(image_path).is_absolute())
            self.assertTrue((final / 'volumes' / image_path).is_file())
            with patch('src.preprocessing.full_dataset.shutil.disk_usage', return_value=NS(free=0)):
                self.assertEqual(run(args, api, download), 1)
            self.assertEqual(len(downloads), 4)
            progress = json.loads((root / 'manifests/full_progress.json').read_text())
            self.assertEqual(progress['state'], 'needs_attention')
            with self.assertRaisesRegex(ValueError, 'differs'):
                completed(final, 'changed-signature')
            (final / 'volumes' / image_path).unlink()
            with self.assertRaisesRegex(ValueError, 'missing or changed'):
                completed(final, receipt['signature'])
