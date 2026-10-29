"""Small CNN trained from scratch, with 512-dimensional retrieval features."""
import os
import torch
from torch import nn


class SimpleCNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        incoming = 3
        for outgoing in (32, 64, 128, 256):
            blocks.append(nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(inplace=False), nn.MaxPool2d(2)))
            incoming = outgoing
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Sequential(nn.Linear(256, 512), nn.ReLU())
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(512, 2)

    def embedding(self, x):
        return self.embed(self.pool(self.features(x)).flatten(1))

    def forward(self, x):
        return self.out(self.dropout(self.embedding(x)))

    @property
    def gradcam_layer(self):
        return self.features[-1][0]


def save_checkpoint(model, path, **notes):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'arch': 'simplecnn_512', 'state_dict': model.state_dict(), **notes}, path)


def load_classifier(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('arch') != 'simplecnn_512':
        raise ValueError('Expected a Simple CNN checkpoint')
    model = SimpleCNNClassifier()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model, checkpoint
