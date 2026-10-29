"""The ResNet-50 PDAC classifier, in the shape the rest of PanVision expects.

It has exactly the same layout as the stand-in used for Grad-CAM and retrieval
(src/explainability/standin_model.py):
    ResNet-50 body (2048 numbers) -> 512-number layer (the "fingerprint") -> 2 scores
so Grad-CAM and retrieval work on it without any change.

Saved file format: a small dictionary holding the weights plus notes (best epoch,
validation score, threshold). This is more robust than saving the whole Python object,
which breaks as soon as a file moves. Always load with load_classifier().
"""
import os

import torch
from torch import nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Start from a ResNet-50 that already learned to see shapes in ordinary photos.
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        backbone.fc = nn.Identity()  # remove the photo categories; keep the 2048 features
        self.backbone = backbone
        self.embed = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())  # 2048 -> 512
        self.out = nn.Linear(512, 2)  # 512 -> [non-PDAC, PDAC]

    def embedding(self, x):
        """The 512-number fingerprint of an image batch (B, 3, H, W) -> (B, 512)."""
        return self.embed(self.backbone(x))

    def forward(self, x):
        return self.out(self.embedding(x))

    @property
    def gradcam_layer(self):
        """The last convolutional block, which Grad-CAM looks at."""
        return self.backbone.layer4[-1]


def save_checkpoint(model, path, **notes):
    """Save weights plus notes (use plain numbers/text for the notes)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"arch": "resnet50_512", "state_dict": model.state_dict(), **notes}, path)


def load_classifier(path, device="cpu"):
    """Load a saved classifier. Returns (model ready for use, notes dictionary)."""
    checkpoint = torch.load(path, map_location=device)
    model = ResNet50Classifier(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
