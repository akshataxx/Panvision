import unittest
from unittest.mock import patch

import numpy as np

from app.inference import analyze_upload


class InferencePipelineTests(unittest.TestCase):
    def test_pipeline_runs_pancreas_classifier_knn_and_returns_ui_payload(self):
        array = np.ones((4, 8, 8), dtype=np.float32)
        pancreas = np.ones_like(array, dtype=bool)
        lesion = np.zeros_like(array, dtype=bool)
        lesion[2, 3:5, 3:5] = True
        classification = {
            'predicted_class': 'PDAC', 'pdac_probability': 0.82, 'threshold': 0.42,
            'embedding': np.ones(512, dtype=np.float32), 'model_key': 'dense',
            'checkpoint_sha256': 'abc', 'gradcam': np.ones((224, 224), dtype=np.float32),
            'gradcam_slice': 2,
        }
        neighbors = [{'patient_id': 'p2', 'label': 1, 'similarity': 0.91}]
        with patch('app.inference.segment_pancreas', return_value=pancreas), \
             patch('app.inference.segment_lesion', return_value=lesion), \
             patch('app.inference.classify_scan', return_value=classification), \
             patch('app.inference.query_similar_cases', return_value=neighbors):
            result = analyze_upload(array, (1, 1, 2))

        self.assertEqual(result['predicted_class'], 'PDAC')
        self.assertEqual(result['neighbors'], neighbors)
        self.assertNotIn('embedding', result)
        np.testing.assert_array_equal(result['lesion_mask'], lesion)
        np.testing.assert_array_equal(result['pancreas_mask'], pancreas)


if __name__ == '__main__':
    unittest.main()
