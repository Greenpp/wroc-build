from typing import Iterable

import torch
from PIL import Image
from torchvision import transforms as T


def split_img(
    img: Image.Image,
) -> Iterable[Image.Image]:
    """Split single 512x512 image into 4 256x256 tiles.

    Returns:
        Four PIL Images, upper-left, upper-right, lower-left, lower-right.
    """
    up_left = img.crop((0, 0, 256, 256))
    up_right = img.crop((256, 0, 512, 256))
    down_left = img.crop((0, 256, 256, 512))
    down_right = img.crop((256, 256, 512, 512))

    return up_left, up_right, down_left, down_right


def create_batch(
    images: Iterable[Image.Image],
    transform: T.Compose,
) -> torch.Tensor:
    """Transform images and stack them into tensor.

    Returns:
        PyTorch tensor of stacked images.
    """
    tensors = []

    for img in images:
        transformed_img = transform(img)
        tensors.append(transformed_img)

    batch_tensor = torch.stack(tensors)

    return batch_tensor
