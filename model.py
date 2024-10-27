import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Squeeze-and-Excitation block
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise attention.
    
    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for SE block, typically set to 16.
    """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Modify the ResNetBlock to include the SE block
class ResNetBlock(nn.Module):
    """
    Residual block with an SE block for improved feature representation.

    Args:
        in_channels (int): Number of input channels.
        num_filters (int): Number of filters for convolutional layers.
        kernel_size (int): Kernel size for convolutional layers, typically set to 3.
    """
    def __init__(self, in_channels, num_filters, kernel_size=3):
        super(ResNetBlock, self).__init__()
        self.resnet_block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1),
        )
        self.se_block = SEBlock(num_filters)

    def forward(self, x):
        residual = x
        x = self.resnet_block(x)
        x = self.se_block(x)
        return x + residual


class Model(nn.Module):
    """
    EDSR-based model with modified SE blocks.

    Args:
        in_channels (int): Number of input channels.
        factor (int): Upscaling factor.
        scale (int): Scaling applied for upsampling layers.
        num_of_residual_blocks (int): Number of residual blocks in the model.
        num_filters (int): Number of filters in convolutional layers.
        kernel_size (int): Kernel size for convolutional layers.
    """
    def __init__(
        self, in_channels=4, factor=2, scale=3, num_of_residual_blocks=20, num_filters=64, kernel_size=3, **kwargs
    ):
        super(Model, self).__init__()
        self.res_blocks = nn.Sequential(
            *[ResNetBlock(in_channels=in_channels, num_filters=num_filters, kernel_size=kernel_size)]
            * num_of_residual_blocks
        )

        self.upsample = nn.Sequential(
            *[nn.Conv2d(num_filters, num_filters * (factor**2), kernel_size=kernel_size, padding=1, **kwargs),
              nn.PixelShuffle(upscale_factor=factor)]
            * scale
        )
        self.resnet_input = nn.Conv2d(in_channels, num_filters, kernel_size=1)
        self.output_layer = nn.Conv2d(num_filters, in_channels, kernel_size=3, padding=1)
        self.resnet_out = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1)

    def forward(self, x):
        x = self.resnet_input(x)
        x_res = self.res_blocks(x)
        x_res = self.resnet_out(x_res)
        out = x + x_res
        out = self.upsample(out)
        return self.output_layer(out)
