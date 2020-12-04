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

    def get_building_probability(self, radius: int = 1) -> float:
        """
        Calculate the mean probability of building being in the center of the selected location, based on probability mask
        """
        min_y = 127 - (radius - 1)
        max_y = 128 + (radius - 1)

        min_x = min_y
        max_x = max_y

        check_area = self.probability_mask[min_y : max_y + 1, min_x : max_x + 1]
        probability = check_area.mean()

        return probability

    def show_img(self) -> None:
        plt.imshow(self.img)
        plt.show()

    def show_probability_mask(self) -> None:
        plt.imshow(self.probability_mask, cmap='gray')
        plt.show()

    def show_binary_mask(self) -> None:
        plt.imshow(self.binary_mask, cmap='gray')
        plt.show()


class Segmentator:
    def __init__(self) -> None:
        self.map_extractor = MapExtractor()
        self.preprocess_transform = get_preprocessing_transform()

        self._load_model()

    def _load_model(self) -> None:
        """
        Setup segmentation model
        """
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
        """
        Create Segmentation object, containing coordinates, image and masks, for given location
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
