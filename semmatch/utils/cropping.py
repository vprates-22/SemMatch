"""
Module: cropping
----------------

This module provides utility functions for cropping image regions based on binary masks.
It is primarily designed for semantic segmentation or object detection workflows, where
a localized object needs to be cropped and resized around its mask for further processing,
such as similarity computation or patch-based classification.

Functions
---------
crop_square_around_mask(image, mask, output_size=(256, 256))
    Extracts a square crop around the binary mask of an object and resizes it to a fixed size.
"""

import cv2
import numpy as np


def crop_square_around_mask(image, mask, output_size=(256, 256)):
    """
    Crop a square region around the binary mask in the image and resize it to a given size.

    This function finds the bounding box of the binary mask, expands it into a square
    centered on the object, and resizes the crop to the specified output dimensions.
    If the crop exceeds image bounds, the image is padded with white (255) pixels.

    Parameters
    ----------
    image : np.ndarray
        The input image (H, W, C) or (H, W) where the object is located.
    mask : np.ndarray
        Binary mask (H, W) indicating the location of the object to crop.
    output_size : tuple of int, optional
        The desired output size (height, width) of the cropped image. Default is (256, 256).

    Returns
    -------
    np.ndarray
        A square image patch cropped around the object and resized to the specified dimensions.

    Raises
    ------
    ValueError
        If the provided mask does not contain any foreground pixels.
    """
    # Ensure mask is boolean
    mask = np.asarray(mask).astype(bool)

    # Get the coordinates of the mask
    y_indices, x_indices = np.where(mask)

    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Mask is empty. No object found.")

    # Bounding box
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Center of the object
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    # Determine square size (max of width/height)
    size = max(x_max - x_min, y_max - y_min)
    half = size // 2 + 1  # +1 to include full object

    # Initial crop coordinates
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    # Pad if crop is out of image bounds
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - image.shape[1])
    pad_bottom = max(0, y2 - image.shape[0])

    # Pad image if needed
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        image = np.pad(image,
                       ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)) if image.ndim == 3 else
                       ((pad_top, pad_bottom), (pad_left, pad_right)),
                       mode='constant', constant_values=255)

    # Adjust crop after padding
    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    # Final square crop
    cropped = image[y1:y2, x1:x2]

    # Resize to fixed shape
    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LINEAR)

    return resized
