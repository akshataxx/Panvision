import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.classification.crop_policy import select_rows
from src.classification.export_embeddings import (mean_l2_embeddings,
                                                   save_embedding_bank,
                                                   validate_checkpoint_provenance)


class EmbeddingTests(unittest.TestCase):
    def test_predicted_crops_never_use_lesion_slice_selection(self):
        lesion_flags = np.array([False, True, False])
        rows = select_rows(lesion_flags, label=1, mode='train',
                           crop_source='predicted_pancreas')
        np.testing.assert_array_equal(rows, [0, 1, 2])

    def test_annotation_assisted_pdac_uses_localised_slices_for_training(self):
        lesion_flags = np.array([False, True, False])
        rows = select_rows(lesion_flags, label=1, mode='train',
                           crop_source='annotation_assisted')
        np.testing.assert_array_equal(rows, [1])

    def test_unknown_crop_source_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'Unknown crop source'):
            select_rows([False, True], label=1, mode='train',
                        crop_source='predicted_pancrease')

    def test_patient_embeddings_are_mean_aggregated_and_normalised(self):
        vectors = np.array([[1., 0.], [0., 1.], [2., 0.]], dtype=np.float32)
        groups = np.array([0, 0, 1])
        result = mean_l2_embeddings(vectors, groups, 2)
        np.testing.assert_allclose(result[0], [2**-.5, 2**-.5], atol=1e-6)
        np.testing.assert_allclose(result[1], [1., 0.], atol=1e-6)

    def test_saved_bank_records_reference_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'patient_embeddings_train.npz'
            save_embedding_bank(path, np.eye(2, dtype=np.float32),
                                ids=['p1', 'p2'], labels=[0, 1], splits=['train', 'train'],
                                entity='patient', reference=True)
            with np.load(path, allow_pickle=False) as bank:
                self.assertEqual(str(bank['entity']), 'patient')
                self.assertTrue(bool(bank['knn_reference']))
                self.assertEqual(set(bank['split'].tolist()), {'train'})

    def test_annotation_assisted_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not trained on predicted-pancreas'):
            validate_checkpoint_provenance(
                {'crop_source': 'annotation_assisted'}, 'split-hash', 144., 'P3')

    def test_checkpoint_prediction_run_must_match(self):
        config = {'crop_source': 'predicted_pancreas', 'splits_sha256': 'split-hash',
                  'window_mm': 144., 'prediction_id': 'P2'}
        with self.assertRaisesRegex(ValueError, 'different segmentation predictions'):
            validate_checkpoint_provenance(config, 'split-hash', 144., 'P3')


if __name__ == '__main__':
    unittest.main()
