"""Candidate 2D segmentation networks. One output channel (logit); 2.5D input means in_channels = context."""

from monai.networks.nets import AttentionUnet, UNet

CHANNELS = (32, 64, 128, 256, 512)
STRIDES = (2, 2, 2, 2)
MODELS = ('unet', 'attention_unet')


def build_model(name, in_channels):
    if name == 'unet':
        return UNet(spatial_dims=2, in_channels=in_channels, out_channels=1, channels=CHANNELS,
                    strides=STRIDES, num_res_units=2, norm='batch')
    if name == 'attention_unet':
        return AttentionUnet(spatial_dims=2, in_channels=in_channels, out_channels=1,
                             channels=CHANNELS, strides=STRIDES)
    raise ValueError(f'Unknown model {name!r}; choose from {MODELS}')
