"""MNIST-family datasets."""

from __future__ import annotations

from torchvision import datasets

from zoo.data.image import ImageDataset


class MNIST(ImageDataset):
    num_classes = 10
    image_shape = (1, 28, 28)
    mean = (0.1307,)
    std = (0.3081,)

    @classmethod
    def _load(cls, root, train, download):
        return datasets.MNIST(root, train=train, download=download)


class QMNIST(MNIST):
    """QMNIST: MNIST reconstruction with a 60k-image test set (vs. 10k)."""

    @classmethod
    def _load(cls, root, train, download):
        return datasets.QMNIST(root, what="train" if train else "test", compat=True, download=download)
