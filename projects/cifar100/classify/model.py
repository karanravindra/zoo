"""Compact ConvNeXt for native-resolution CIFAR images."""

import torch
from torch import nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, 4 * channels), nn.GELU(),
                                 nn.Linear(4 * channels, channels))
        self.scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x):
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.mlp(self.norm(x)) * self.scale
        return residual + x.permute(0, 3, 1, 2)


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=100, widths=(32, 64, 128), depths=(2, 2, 4)):
        super().__init__()
        if not widths or len(widths) != len(depths) or min(*widths, *depths) < 1:
            raise ValueError("widths and depths must be positive lists of equal length")
        if len(widths) > 5:
            raise ValueError("At most five stages fit 32×32 images")
        stages = []
        channels = 3
        for width, depth in zip(widths, depths):
            stages.append(nn.Conv2d(channels, width, kernel_size=2, stride=2))
            stages.extend(ConvNeXtBlock(width) for _ in range(depth))
            channels = width
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, num_classes))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(-2, -1)))
