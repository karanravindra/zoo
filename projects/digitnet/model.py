import torch
import torch.nn as nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 5x5 convs (BN + GELU) followed by 2x2 max-pool, halving spatial size."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 5, padding=2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, 5, padding=2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
        nn.MaxPool2d(2),
    )


class DigitNet(nn.Module):
    """Small CNN for 28x28 digit classification: 28 -> 14 -> 7 -> global pool -> logits."""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, width: int = 32, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, width),
            conv_block(width, width * 2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
