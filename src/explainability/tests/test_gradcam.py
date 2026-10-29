import unittest
import importlib.util

import numpy as np

from src.explainability.gradcam import generate_heatmap


class GradCAMTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch') and
                         importlib.util.find_spec('pytorch_grad_cam'),
                         'Grad-CAM inference dependencies are optional')
    def test_heatmap_is_finite_normalized_and_image_sized(self):
        import torch
        from torch import nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Conv2d(1, 2, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Linear(2, 2)

            def forward(self, value):
                return self.head(self.pool(self.features(value)).flatten(1))

        model = TinyModel().eval()
        image = torch.rand(1, 1, 16, 16)

        heatmap = generate_heatmap(model, image, model.features, target_class=1)

        self.assertEqual(heatmap.shape, (16, 16))
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertGreaterEqual(float(heatmap.min()), 0)
        self.assertLessEqual(float(heatmap.max()), 1)


if __name__ == '__main__':
    unittest.main()
