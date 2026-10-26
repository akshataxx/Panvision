"""DenseNet-121 PDAC baseline with a 512-dimensional retrieval embedding."""
import os
import torch
from torch import nn
from torchvision import models


class DenseNet121Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights)
        self.backbone.classifier = nn.Identity()
        self.embed = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.out = nn.Linear(512, 2)

    def embedding(self, x):
        # torchvision DenseNet forward includes final ReLU and global average pooling.
        return self.embed(self.backbone(x))

    def forward(self, x):
        return self.out(self.embedding(x))

    @property
    def gradcam_layer(self):
        return self.backbone.features.denseblock4


def save_checkpoint(model, path, **notes):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'arch': 'densenet121_512', 'state_dict': model.state_dict(), **notes}, path)


def load_classifier(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('arch') != 'densenet121_512':
        raise ValueError('Expected a DenseNet-121 checkpoint; do not load ResNet weights here')
    model = DenseNet121Classifier(pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model, checkpoint
