"""rFID: the Fréchet distance between Inception-v3 features of images and of their reconstructions.

Features are the 2048-d pool features of torch-fidelity's port of the TF Inception that the standard FID uses (the
weights are downloaded on first use, to `TORCH_HOME`). Images are quantized to uint8 first, as saved images would be, and
the statistics are kept in float64.

With a few images the score is biased upwards and noisy (the covariance of 1,000 images in 2048 dimensions is rank
deficient), so compare runs on the same images; the numbers are not comparable with an rFID over 50k images.
"""

from __future__ import annotations

import torch
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

FEATURE_DIM = 2048


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float images -> uint8."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


class Inception:
    """Callable: a batch of images in [-1, 1] -> (B, 2048) features."""

    def __init__(self, device):
        self.net = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"]).to(device)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(to_uint8(x))[0]


class FeatureStats:
    """Running mean and covariance of feature vectors, kept on the device in float64."""

    def __init__(self, device, dim: int = FEATURE_DIM):
        self.n = 0
        self.sum = torch.zeros(dim, dtype=torch.float64, device=device)
        self.outer = torch.zeros(dim, dim, dtype=torch.float64, device=device)
        self._root: torch.Tensor | None = None

    def update(self, feats: torch.Tensor) -> None:
        f = feats.double()
        self.n += len(f)
        self.sum += f.sum(0)
        self.outer += f.T @ f
        self._root = None

    def mean_cov(self) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.sum / self.n
        return mu, (self.outer - self.n * torch.outer(mu, mu)) / (self.n - 1)

    def root_cov(self) -> torch.Tensor:
        """The symmetric square root of the covariance, C^(1/2). It costs an eigendecomposition, so it is computed once
        and kept until the next `update`: the statistics of a fixed set of real images are compared with the
        reconstructions of every epoch."""
        if self._root is None:
            w, v = torch.linalg.eigh(self.mean_cov()[1])
            self._root = (v * w.clamp_min(0).sqrt()) @ v.T
        return self._root


def frechet_distance(a: FeatureStats, b: FeatureStats) -> float:
    """|mu_a - mu_b|^2 + tr(C_a + C_b - 2 (C_a C_b)^(1/2)). The trace term is the sum of the square roots of the
    eigenvalues of C_a^(1/2) C_b C_a^(1/2), which has the same eigenvalues as C_a C_b but is symmetric, so everything
    stays real. `a` is the reference: its square root is cached on the object (`root_cov`), so pass the same `a` when
    comparing one image set with many."""
    mu_a, c_a = a.mean_cov()
    mu_b, c_b = b.mean_cov()
    root_a = a.root_cov()
    m = root_a @ c_b @ root_a
    tr_sqrt = torch.linalg.eigvalsh((m + m.T) / 2).clamp_min(0).sqrt().sum()
    return float((mu_a - mu_b).pow(2).sum() + c_a.trace() + c_b.trace() - 2 * tr_sqrt)
