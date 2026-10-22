import torch
from torch import nn
from torchinfo import summary


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        self.scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.mlp(self.norm(x)) * self.scale
        return residual + x.permute(0, 3, 1, 2)


class ConvNeXt(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2, stride=2),
            ConvNeXtBlock(8),
            ConvNeXtBlock(8),
            nn.Conv2d(8, 32, kernel_size=2, stride=2),
            ConvNeXtBlock(32),
            ConvNeXtBlock(32),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
        )
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(dim=(-2, -1)))


if __name__ == "__main__":
    summary(ConvNeXt(), input_size=(1, 1, 28, 28))
