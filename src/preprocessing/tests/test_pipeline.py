import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.preprocessing.pipeline import (
    load_manifest, patient_splits, preprocess, read_pair, run, validate_splits,
)


class PipelineTests(unittest.TestCase):
    def pair(self):
        data = np.full((6, 8, 10), 100, dtype=np.float32)
        data[0, 0, 0] = -1000
        data[-1, -1, -1] = 1000
        image = sitk.GetImageFromArray(data)
        image.SetSpacing((2, 2, 4))
        image.SetOrigin((10, 20, 30))
        mask = sitk.GetImageFromArray(np.zeros(data.shape, dtype=np.uint8))
        mask.CopyInformation(image)
        return image, mask

    def test_grouped_deterministic_split_and_mismatch(self):
        frame = pd.DataFrame([
            {'patient_id': f'p{i}', 'label': str(i % 2)}
            for i in range(40) for _ in range(2)])
        split = patient_splits(frame)
        pd.testing.assert_frame_equal(split, patient_splits(frame.sample(frac=1)))
        self.assertEqual(len(split), 40)
        self.assertEqual(set(split.split), {'train', 'val', 'test'})
        validate_splits(frame, split)
        validate_splits(frame.iloc[:4], split, allow_subset=True)
        with self.assertRaises(ValueError):
            validate_splits(frame.iloc[:4], split)
        unknown = frame.iloc[:1].copy()
        unknown['patient_id'] = 'unknown'
        with self.assertRaises(ValueError):
            validate_splits(unknown, split, allow_subset=True)
        with self.assertRaises(ValueError):
            validate_splits(frame, pd.concat([split, split.iloc[:1]]))
        changed = split.copy()
        changed.iloc[0, changed.columns.get_loc('label')] = 'wrong'
        with self.assertRaises(ValueError):
            validate_splits(frame, changed)

    def test_resampling_preserves_geometry_and_labels(self):
        image, mask = self.pair()
        values = sitk.GetArrayFromImage(mask)
        values[2:4, 2:6, 2:8] = 4
        values[3, 3, 3] = 1
        mask = sitk.GetImageFromArray(values)
        mask.CopyInformation(image)
        image, mask = preprocess(image, mask, [1, 1, 2], [-100, 300])
        self.assertEqual(image.GetSpacing(), (1, 1, 2))
        self.assertEqual(image.GetOrigin(), (10, 20, 30))
        self.assertEqual(image.GetSize(), mask.GetSize())
        self.assertEqual(set(np.unique(sitk.GetArrayFromImage(mask))), {0, 1, 4})
        data = sitk.GetArrayFromImage(image)
        self.assertEqual(data.min(), 0)
        self.assertEqual(data.max(), 1)

    def test_geometry_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image, mask = self.pair()
            mask.SetOrigin((100, 20, 30))
            sitk.WriteImage(image, str(directory / 'image.nii.gz'))
            sitk.WriteImage(mask, str(directory / 'mask.nii.gz'))
            row = Namespace(image_path=str(directory / 'image.nii.gz'),
                            mask_path=str(directory / 'mask.nii.gz'), label='1')
            with self.assertRaisesRegex(ValueError, 'geometry'):
                read_pair(row)

    def test_end_to_end_preserves_missing_masks_and_existing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows = []
            for i in range(20):
                image, mask = self.pair()
                image_path = f'ct{i}.nii.gz'
                mask_path = f'mask{i}.nii.gz' if i else ''
                sitk.WriteImage(image, str(directory / image_path))
                if mask_path:
                    sitk.WriteImage(mask, str(directory / mask_path))
                rows.append(dict(study_id=f's{i}', patient_id=f'p{i}', label=str(i % 2),
                                 image_path=image_path, mask_path=mask_path,
                                 annotation_tier='automatic' if mask_path else 'none',
                                 external_source='synthetic'))
            pd.DataFrame(rows).to_csv(directory / 'manifest.csv', index=False)
            args = Namespace(command='prepare', manifest=directory / 'manifest.csv',
                             splits=directory / 'splits.csv', output=directory / 'processed',
                             seed=42, spacing=[2, 2, 4], window=[-100, 300])
            with self.assertRaisesRegex(ValueError, 'Shared split is missing'):
                run(args)
            patient_splits(pd.DataFrame(rows)).to_csv(args.splits, index=False)
            run(args)
            before = args.splits.read_bytes()
            result = pd.read_csv(args.output / 'manifest.csv', keep_default_na=False)
            missing = result[result.study_id == 's0'].iloc[0]
            self.assertEqual(missing.processed_labels, '')
            self.assertFalse((Path(missing.processed_image).parent / 'lesion.nii.gz').exists())
            args.seed = 999
            args.output = directory / 'second_run'
            run(args)
            self.assertEqual(before, args.splits.read_bytes())
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                run(args)
            manifest = load_manifest(args.manifest)
            manifest.loc[1, 'label'] = 'unknown'
            manifest.to_csv(directory / 'bad.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'label must'):
                load_manifest(directory / 'bad.csv')


if __name__ == '__main__':
    unittest.main()
