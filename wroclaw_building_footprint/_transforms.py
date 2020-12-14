import torch
from torchvision import transforms as T

COLOR_MEAN = torch.Tensor([77.63 / 255, 89.57 / 255, 101.36 / 255]).reshape((3, 1, 1))


def get_preprocessing_transform() -> T.Compose:
    """Create a preprocessing transform for the segmentation model."""
    composition = T.Compose(
        [
            T.ToTensor(),
            T.Lambda(lambda x: x - x.mean(dim=(1, 2)).reshape((3, 1, 1)) + COLOR_MEAN),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    return composition
