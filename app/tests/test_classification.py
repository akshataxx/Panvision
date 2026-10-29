import unittest

import numpy as np

from app.classification import (aggregate_predictions, prepare_classifier_crops,
                                project_gradcam_to_source)


class ClassificationInferenceTests(unittest.TestCase):
    def test_predicted_pancreas_crops_match_training_shape(self):
        array = np.linspace(-100, 300, 8 * 32 * 32, dtype=np.float32).reshape(8, 32, 32)
        mask = np.zeros_like(array, dtype=bool)
        mask[2:7, 10:22, 11:23] = True

        crops, slices = prepare_classifier_crops(array, (1.5, 1.5, 3.0), mask)

        self.assertEqual(crops.shape[1:], (224, 224))
        self.assertEqual(crops.dtype, np.uint8)
        self.assertGreater(len(crops), 0)
        self.assertEqual(len(crops), len(slices))

    def test_patient_score_and_embedding_use_frozen_aggregation(self):
        probabilities = np.asarray([0.2, 0.8], dtype=np.float32)
        embeddings = np.zeros((2, 512), dtype=np.float32)
        embeddings[0, 0] = 1
        embeddings[1, 1] = 1

        result = aggregate_predictions(probabilities, embeddings, threshold=0.42)

        self.assertAlmostEqual(result['pdac_probability'], 0.5)
        self.assertEqual(result['predicted_class'], 'PDAC')
        self.assertAlmostEqual(float(np.linalg.norm(result['embedding'])), 1.0, places=6)

    def test_gradcam_crop_is_projected_to_original_scan(self):
        array = np.zeros((5, 32, 40), dtype=np.float32)
        mask = np.zeros_like(array, dtype=bool)
        mask[2, 10:20, 14:24] = True
        heatmap = np.ones((224, 224), dtype=np.float32)

        projected, source_slice = project_gradcam_to_source(
            array, (1.5, 1.5, 3.0), mask, 2, heatmap)

        self.assertEqual(projected.shape, array.shape[1:])
        self.assertEqual(source_slice, 2)
        self.assertGreater(float(projected.max()), 0.9)
        self.assertEqual(float(projected[0, 0]), 0.0)

