import os
from distutils.sysconfig import get_python_lib

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from segmentation_models_pytorch import DeepLabV3Plus

from ._postprocess import combine_masks, create_binary_mask, merge_masks
from ._preprocess import create_batch, split_img
from ._transforms import get_preprocessing_transform
from .extractor import MapExtractor

BASE_DIR = None
if os.path.isfile(get_python_lib() + '/plateDetect'):
    BASE_DIR = get_python_lib() + '/plateDetect'
else:
    BASE_DIR = os.path.dirname(__file__)


class Segmentation:
    """Class representing the result of the map segmentation.

    Attributes:
        longitude: The selected longitude.
        latitude: The selected latitude.
        img: The extracted orthophoto.
        probability_mask: The generated probability mask with values in range 0-1.
        binary_mask: The generated binary mask with values 0 or 1.

    """

    def __init__(
        self,
        longitude: float,
        latitude: float,
        img: Image.Image,
        probability_mask: np.ndarray,
        binary_mask: np.ndarray,
    ) -> None:
        self.longitude = longitude
        self.latitude = latitude
        self.img = img
        self.probability_mask = probability_mask
        self.binary_mask = binary_mask

    def get_building_probability(self, size: int = 2) -> float:
        """Calculate the probability of a building being at the selected coordinates.

        The returned probability is a mean of all probabilities within a square with a given side length from the center.
        With default segmentation parameters, the generated binary masks mark areas with probability >30% as buildings.

        Args:
            size: An integer length of the calculation square side.

        Returns:
            A float representation of the probability of building at the selected coordinates in the range 0-1.
        """
        radius = size // 2

        min_y = 127 - (radius - 1)
        max_y = 128 + (radius - 1)

        min_x = min_y
        max_x = max_y

        check_area = self.probability_mask[min_y : max_y + 1, min_x : max_x + 1]
        probability = check_area.mean()

        return probability

    def show_img(self) -> None:
        """Show the segmented orthophoto with matplotlib"""
        plt.imshow(self.img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def show_probability_mask(self) -> None:
        """Show the generated probability mask with matplotlib"""
        plt.imshow(self.probability_mask, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def show_binary_mask(self) -> None:
        """Show the generated binary mask with matplotlib"""
        plt.imshow(self.binary_mask, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class Segmentator:
    """Class used to generate building footprint masks for Wrocław."""

    def __init__(self) -> None:
        self.map_extractor = MapExtractor()
        self.preprocess_transform = get_preprocessing_transform()

        self._load_model()

    def _load_model(self) -> None:
        """Setup the segmentation model"""
        self.model = DeepLabV3Plus(encoder_name='resnet34', activation='sigmoid')

        self.model.load_state_dict(torch.load(f'{BASE_DIR}/model/seg_model.pt'))
        self.model.eval()

    def segment(
        self,
        longitude: float,
        latitude: float,
        area_size: float = 153.6,
        combination_beta1: float = 1,
        combination_beta2: float = 1.5,
        binary_threshold: float = 0.3,
        closing_kernel_size: int = 5,
        opening_kernel_size: int = 5,
    ) -> Segmentation:
        """Generate segmentation mask for a square area around given coordinates.

        Args:
            longitude: Longitude of the center of the area to segment.
            latitude: Latitude of the center of the area to segment.
            area_size: Length of the side of the area in meters.
            combination_beta1: Weight applied to the whole image mask during combination.
            combination_beta2: Weight applied to the tiled image mask during combination.
            binary_threshold: The minimal probability to classify an area as a building during the binary mask creation, must be in the range 0-1.
            closing_kernel_size: Size of the closing operation kernel.
            opening_kernel_size: Size of the opening operation kernel.

        Returns:
            An object of Segmentation class.
        """
        img = self.map_extractor.get_area_at(longitude, latitude, area_size)
        img_tiles = split_img(img)

        img = img.resize((256, 256), Image.BILINEAR)

        img_tensor = self.preprocess_transform(img).unsqueeze(dim=0)
        tiles_batch = create_batch(img_tiles, self.preprocess_transform)

        batch = torch.cat([img_tensor, tiles_batch])
        segmentation_output = self.model(batch).detach().numpy()[:, 0, :]

        img_mask, *tiles_masks = segmentation_output
        tiles_mask = merge_masks(tiles_masks)
        tiles_mask = cv2.resize(tiles_mask, (256, 256))

        combined_mask = combine_masks(
            img_mask,
            tiles_mask,
            combination_beta1,
            combination_beta2,
        )
        binary_mask = create_binary_mask(
            combined_mask,
            binary_threshold,
            closing_kernel_size,
            opening_kernel_size,
        )

        return Segmentation(
            longitude,
            latitude,
            img,
            combined_mask,
            binary_mask,
        )
