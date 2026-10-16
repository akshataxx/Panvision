"""Grad-CAM heat maps for the PDAC classifier.

Reusable functions: the dashboard and the experiment scripts both import from here.
"""
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def generate_heatmap(model, image_tensor, target_layer, target_class=1):
    """Return a heat map for one preprocessed ROI image.

    model:        trained PyTorch classifier (in eval mode)
    image_tensor: tensor of shape (1, 3, H, W)
    target_layer: last convolutional layer of the model
    target_class: which output to explain; 1 = PDAC
    Returns a 2D float array of shape (H, W) with values between 0 and 1.
    """
    cam = GradCAM(model=model, target_layers=[target_layer])
    return cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(target_class)])[0]


def restrict_to_mask(heat, mask):
    """Keep heat only where the mask is on (e.g. inside the pancreas); zero elsewhere.

    heat: 2D array (H, W) from generate_heatmap
    mask: 2D array (H, W), non-zero (or True) inside the region to keep
    Returns (restricted_heat, inside_fraction), where inside_fraction is the share of
    the original total heat that lay inside the mask (0 to 1). A low value means the
    model was mostly looking OUTSIDE the pancreas.
    """
    mask = np.asarray(mask) > 0
    if mask.shape != heat.shape:
        raise ValueError(f"mask shape {mask.shape} does not match heat map shape {heat.shape}")
    restricted = np.where(mask, heat, 0.0).astype(np.float32)
    total = float(heat.sum())
    inside_fraction = float(restricted.sum()) / total if total > 0 else 0.0
    return restricted, inside_fraction