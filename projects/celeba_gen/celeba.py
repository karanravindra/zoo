"""CelebA-HQ at 256x256, cached as uint8 on disk.

`CelebAHQ256` is an `ImageDataset`: the first construction resizes every image to 256x256 (bicubic, antialiased) and
writes `<root>/cache/CelebAHQ256/{train,test}/`, so later runs read memory-mapped arrays instead of decoding the 1024px
originals again. The size is part of the class name because the cache key is the class name: a different resolution
needs a different class, never a different value passed to the same cache.

Splits follow `zoo.data.stargan`: `train=True` is the StarGAN train split (28,000 images), `train=False` is its val
split (2,000 images). Training scripts carve their own validation set out of the train split.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torchvision import datasets
from torchvision.transforms.v2.functional import pil_to_tensor

from zoo.data import stargan
from zoo.data.image import ImageDataset

NAME = "celeba_hq"
SIZE = 256


class CelebAHQ256(ImageDataset):
    num_classes = 2  # the folders under each split (female / male); the autoencoder ignores labels
    image_shape = (3, SIZE, SIZE)
    mean = (0.5, 0.5, 0.5)  # with std 0.5 the default transform maps uint8 [0, 255] to [-1, 1]
    std = (0.5, 0.5, 0.5)

    @classmethod
    def preprocess(cls, img: Image.Image):
        # Pillow's resize antialiases when downsampling, so this is a proper bicubic downscale from 1024px.
        img = img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        return pil_to_tensor(img)

    @classmethod
    def _load(cls, root, train, download):
        root = Path(root)
        base = stargan.download(NAME, root) if download else root / NAME
        split = stargan.splits(NAME)[0 if train else 1]
        return datasets.ImageFolder(str(base / split))
