"""One place to load the classifier the team picked, and to turn saved crops into model input.

Every classifier used by the explainability code must offer three things:
    model.embedding(x)   the 512-number fingerprint of an image batch (B, 3, 224, 224) -> (B, 512)
    model(x)             the two scores [non-PDAC, PDAC] -> (B, 2)
    model.gradcam_layer  the last convolutional block, which Grad-CAM looks at
The ResNet-50, the DenseNet-121 and the small CNN (all in src/classification/) do, so
load_model() picks the right one from the checkpoint.
"""
import sys
from pathlib import Path

import numpy as np
import torch

# The classification code is imported as "src.classification...", like the rest of the repo,
# so the repository root must be on Python's path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.dataset import MEAN, STD  # noqa: E402  (same normalisation as training)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path, device=None):
    """Load a saved classifier. Returns (model in eval mode, notes dictionary).

    The notes hold the best epoch, the validation score and the suggested threshold.
    """
    device = device or default_device()
    checkpoint = torch.load(path, map_location="cpu")
    arch = checkpoint.get("arch") if isinstance(checkpoint, dict) else None
    if arch == "resnet50_512":
        from src.classification.resnet_model import load_classifier
    elif arch == "densenet121_512":
        from src.classification.densenet_model import load_classifier
    elif arch == "simplecnn_512":
        from src.classification.cnn_model import load_classifier
    else:
        # A new network: give it the three things listed at the top of this file and add a branch.
        raise ValueError(f"Unknown classifier type {arch!r} in {path}. Add it to load_model().")
    return load_classifier(path, device)


def to_model_input(crops):
    """Saved crops (N, 224, 224) of whole numbers 0-255 -> a model input tensor (N, 3, 224, 224).

    Does exactly what CropDataset does when it feeds the model: scale to 0-1,
    copy into 3 colour channels, normalise.
    """
    x = torch.from_numpy(np.asarray(crops)).float() / 255.0
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    return (x - MEAN) / STD
