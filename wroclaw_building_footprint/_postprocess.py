import cv2
import numpy as np


def merge_masks(
    masks: np.ndarray,
) -> np.ndarray:
    """
    Merge masks of split tiles back into a single image
    """
    up_left, up_right, down_left, down_right = masks

    upper = np.concatenate([up_left, up_right], axis=1)
    lower = np.concatenate([down_left, down_right], axis=1)

    mask = np.concatenate([upper, lower], axis=0)

    return mask


def combine_masks(
    mask1: np.ndarray,
    mask2: np.ndarray,
    beta1: float = 1,
    beta2: float = 1,
) -> np.ndarray:
    """
    Add both masks and normalize to 1
    """
    mask_stack = np.stack([mask1 * beta1, mask2 * beta2])
    combined_mask = np.sum(mask_stack, axis=0)

    normalized_mask = combined_mask / (beta1 + beta2)

    return normalized_mask


def create_binary_mask(
    mask: np.ndarray,
    threshold: float = 0.5,
    close_kernel_size: int = 5,
    open_kernel_size: int = 3,
) -> np.ndarray:
    """
    Transform segmentation mask into a binary mask and apply optional morphological transforms
    """
    binary_mask = np.where(mask >= threshold, 1, 0).astype('uint8')

    if open_kernel_size is not None:
        open_kernel = np.ones((open_kernel_size, open_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel)

    if close_kernel_size is not None:
        close_kernel = np.ones((close_kernel_size, close_kernel_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)

    return binary_mask
