"""Grad-CAM heat maps for the PDAC classifier.

Reusable functions: the dashboard and the experiment scripts both import from here.
"""
import numpy as np


def generate_heatmap(model, image_tensor, target_layer, target_class=1):
    """Return a heatmap (2D array scaled 0-1) for one preprocessed ROI image.

    model:        trained PyTorch classifier
    image_tensor: tensor of shape (1, C, H, W)
    target_layer: last convolutional layer of the model
    target_class: 1 = PDAC
    """
    # Keep the optional Grad-CAM dependency lazy so the rest of the API can
    # start and report model availability even when inference extras are not
    # installed yet.
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = GradCAM(model=model, target_layers=[target_layer])
    return cam(input_tensor=image_tensor,
               targets=[ClassifierOutputTarget(target_class)])[0]


def restrict_to_mask(heat, mask):
    """Keep heat only inside a same-sized pancreas mask."""
    mask = np.asarray(mask) > 0
    if mask.shape != heat.shape:
        raise ValueError(f'mask shape {mask.shape} does not match heat map shape {heat.shape}')
    restricted = np.where(mask, heat, 0.0).astype(np.float32)
    total = float(heat.sum())
    inside_fraction = float(restricted.sum()) / total if total > 0 else 0.0
    return restricted, inside_fraction
