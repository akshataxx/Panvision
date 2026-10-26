"""One place to load the classifier the team picked, and to turn saved crops into model input.

Every classifier used by the explainability code must offer three things:
    model.embedding(x)   the 512-number fingerprint of an image batch (B, 3, 224, 224) -> (B, 512)
    model(x)             the two scores [non-PDAC, PDAC] -> (B, 2)
    model.gradcam_layer  the last convolutional block, which Grad-CAM looks at
The ResNet-50 (src/classification/resnet_model.py), the DenseNet-121 and the small CNN
(research/classification/) all do, so load_model() picks the right one from the checkpoint.
"""
import sys
from pathlib import Path

import numpy as np
import torch

# The classification code lives in a sibling folder, and the CNN / DenseNet models in
# research/classification; make both importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_DIR = REPO_ROOT / "src" / "classification"
RESEARCH_DIR = REPO_ROOT / "research" / "classification"
for folder in (CLASSIFICATION_DIR, RESEARCH_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from dataset import MEAN, STD  # noqa: E402  (same normalisation the model was trained with)


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
        from resnet_model import load_classifier
    elif arch == "densenet121_512":
        from densenet_model import load_classifier
    elif arch == "simplecnn_512":
        from cnn_model import load_classifier
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
