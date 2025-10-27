"""
Module: io
----------

This module provides utility functions for image loading, resizing, and dictionary merging
used in image processing pipelines, particularly with PyTorch.

Functions
---------
- load_image(path, gray=False)
    Loads an RGB or grayscale image from a file path and returns it as a torch tensor.
- load_depth(path)
    Loads a depth image from a file path and returns it as a torch tensor in meters.
- reshape(image, shape)
    Resizes an image to an exact shape and returns the new image along with the scale factors.
- resize_long_edge(image, max_size)
    Resizes an image so that its longest edge matches `max_size`, preserving aspect ratio.
- combine_dicts(dict1, dict2)
    Merges two dictionaries, preserving only truthy values and giving precedence to the second dictionary.
"""

from typing import Tuple, Union

import cv2
import torch
import torchvision
from torchvision import transforms

to_tensor = transforms.ToTensor()


def load_image(path, gray=False) -> torch.Tensor:
    """
    Load an image from a file path and return it as a torch tensor.

    Parameters
    ----------
    path : str
        Path to the image file.
    gray : bool, optional
        If True, converts the image to grayscale. Default is False.

    Returns
    -------
    torch.Tensor
        Tensor of shape (1 or 3, H, W) with float32 values in [0, 1].
    """
    image_cv = cv2.imread(path)
    image = to_tensor(image_cv)

    if gray:
        image = transforms.functional.rgb_to_grayscale(image)

    return image


def load_depth(path) -> torch.Tensor:
    """
    Load a depth image and convert it to meters.

    Parameters
    ----------
    path : str
        Path to the depth image file (usually 16-bit PNG).

    Returns
    -------
    torch.Tensor
        A single-channel depth image tensor with shape (H, W) in meters.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) / 1000
    image = torch.tensor(image)

    return image


def reshape(image: torch.Tensor, shape: Union[int, tuple, list]) -> Tuple[torch.Tensor, Union[float, tuple]]:
    """
    Resize an image tensor to a specified shape using bilinear interpolation.

    Parameters
    ----------
    image : torch.Tensor
        The image tensor of shape (C, H, W).
    shape : tuple or int
        The desired shape as (H, W). If int, both height and width are set to this value.

    Returns
    -------
    tuple
        - Resized image tensor of shape (C, new_H, new_W).
        - Scale factors as a tuple (scale_h, scale_w).
    """
    h, w = image.shape[-2:]
    new_h, new_w = shape
    scale_h = new_h / h
    scale_w = new_w / w

    return torch.nn.functional.interpolate(image[None], size=shape, mode='bilinear', align_corners=False)[0], (scale_h, scale_w)


def resize_long_edge(image: torch.Tensor, max_size: int) -> Tuple[torch.Tensor, float]:
    """
    Resize an image so that its longest edge matches `max_size`, preserving the aspect ratio.

    Parameters
    ----------
    image : torch.Tensor
        The image tensor of shape (C, H, W).
    max_size : int
        The desired size of the longest edge.

    Returns
    -------
    tuple
        - Resized image tensor of shape (C, new_H, new_W).
        - Scale factor (float) applied to the height.
    """
    h, w = image.shape[-2:]
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)

    scale = new_h / h

    return torch.nn.functional.interpolate(image[None], size=(new_h, new_w), mode='bilinear', align_corners=False)[0], scale


def combine_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Combine two dictionaries, preserving only truthy values and preferring values from dict2.

    If both dictionaries contain a key:
    - Use the value from dict2 if it is truthy.
    - Otherwise, fall back to the value from dict1.

    Parameters
    ----------
    dict1 : dict
        The base dictionary.
    dict2 : dict
        The overriding dictionary.

    Returns
    -------
    dict
        The combined dictionary with only truthy values preserved.
    """
    keys = set(dict1) | set(dict2)
    result = {}
    for key in keys:
        v1 = dict1.get(key)
        v2 = dict2.get(key)
        result[key] = v2 if v2 else v1
        if not result[key]:
            result[key] = v2 if key in dict2 else v1
    return result
