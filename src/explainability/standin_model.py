"""A STAND-IN classifier, used only until the team's real PDAC classifier exists.

It copies the design in the proposal: a ResNet-50 backbone whose output is squeezed
2048 -> 512 numbers (the "embedding" used for similar-case retrieval) and then
-> 2 scores (non-PDAC, PDAC).

IMPORTANT: the backbone has ordinary photo-trained weights and the 512 -> 2 head is
RANDOM. Its predictions and heat maps mean nothing medically. Its only job is to let
us build and test Grad-CAM and retrieval code that will later work, unchanged, on
the real model (as long as the real model keeps the same shape and layer names).
"""
import torch
from torch import nn
from torchvision import models


class StandInClassifier(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)  # makes the random head the same every run
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Identity()  # drop the photo classes; keep the 2048 features
        self.backbone = backbone
        self.embed = nn.Sequential(nn.Linear(2048, 512), nn.ReLU())  # 2048 -> 512
        self.out = nn.Linear(512, 2)  # 512 -> [non-PDAC, PDAC]
        self.eval()

    def embedding(self, x):
        """The 512-number fingerprint of an image batch (B, 3, H, W) -> (B, 512)."""
        return self.embed(self.backbone(x))

    def forward(self, x):
        return self.out(self.embedding(x))

    @property
    def gradcam_layer(self):
        """The last convolutional block, which Grad-CAM looks at."""
        return self.backbone.layer4[-1]
