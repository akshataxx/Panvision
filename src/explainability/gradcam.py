"""Grad-CAM heatmaps for the PDAC classifier.

TODO (explainability owner): implement with the `grad-cam` package.
"""


def generate_heatmap(model, image_tensor, target_layer, target_class=1):
    """Return a heatmap (2D array scaled 0-1) for one preprocessed ROI image.

    model:        trained PyTorch classifier
    image_tensor: tensor of shape (1, C, H, W)
    target_layer: last convolutional layer of the model
    target_class: 1 = PDAC
    """
    raise NotImplementedError("Not implemented yet")
