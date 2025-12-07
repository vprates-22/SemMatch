"""
Module: utils.image
--------------------

This module provides a collection of utility functions for image processing
within the SemMatch project. It includes functionalities for loading images
and depth maps, reshaping and resizing images, and cropping regions based on masks.

Functions:
- load_image: Loads an image from a file path into a PyTorch tensor.
- load_depth: Loads a depth map from a file path into a PyTorch tensor.
- reshape: Resizes an image tensor to a specified shape.
- resize_long_edge: Resizes an image tensor such that its long edge matches a given size.
- crop_square_around_mask: Crops a square region around a binary mask in an image.
"""
from typing import Tuple, Union

import cv2
import h5py
import torch

import numpy as np

from torch import Tensor
from torchvision import transforms

to_tensor = transforms.ToTensor()

# def to_tensor(img_np: np.ndarray) -> torch.Tensor:
#     """
#     Converts a NumPy image array to a normalized PyTorch tensor.

#     Parameters
#     ----------
#     img_np : np.ndarray
#         Input image as a NumPy array of shape (H, W, C) or (H, W).

#     Returns
#     -------
#     torch.Tensor
#         Image as a PyTorch tensor of shape (C, H, W) with float32 values in [0, 1].
#     """
#     if isinstance(img_np, torch.Tensor):
#         return img_np

#     img_np = img_np.astype(np.float32) / 255.0
#     img_np = img_np * 2 - 1
#     img_np = np.transpose(img_np, (2, 0, 1))
#     img_tensor = torch.from_numpy(img_np).unsqueeze(0)
#     return img_tensor


def to_cv(torch_image, convert_color=False, batch_idx=0, to_gray=False):
    """
    Converts a PyTorch image tensor to a NumPy array in OpenCV format.

    Parameters
    ----------
    torch_image : torch.Tensor or np.ndarray
        Input image tensor, can be shape (C, H, W), (B, C, H, W), or already NumPy.
    convert_color : bool, optional
        If True, converts RGB to BGR (OpenCV format).
    batch_idx : int, optional
        Index to select if image is batched (shape [B, C, H, W]).
    to_gray : bool, optional
        If True, converts the final image to grayscale.

    Returns
    -------
    np.ndarray
        Image as a NumPy array, possibly in BGR or grayscale.
    """
    if isinstance(torch_image, torch.Tensor):
        if torch_image.dim() == 4:
            torch_image = torch_image[batch_idx]

        if torch_image.dim() == 2:
            torch_image = torch_image.unsqueeze(0)

        if torch_image.dim() != 3:
            raise ValueError(f"Unsupported tensor shape: {torch_image.shape}")

        C, H, W = torch_image.shape

        img = torch_image.detach().cpu().float()

        if img.max() <= 1.0:
            img = img * 255.0

        img = img.permute(1, 2, 0).numpy().astype(np.uint8)

    else:
        if img.max() <= 1.0:
            img = img * 255.0

        img = np.array(torch_image, dtype=np.uint8)

    if convert_color and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if to_gray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def load_image(path, gray=False) -> Tensor:
    """
    Load an image from a file path and return it as a torch tensor.

    Parameters
    ----------
    path : str
        Path to the image file.
    gray : bool, optional
        If True, converts the image to grayscale. Defaults to False.

    Returns
    -------
    torch.Tensor
        Tensor of shape (1 or 3, H, W) with float32 values in [0, 1].
    """
    image_cv = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
    image = to_tensor(image_cv)

    if gray:
        image = transforms.functional.rgb_to_grayscale(image)

    return image


def load_depth(path) -> Tensor:
    """
    Load a depth image and convert it to meters.

    Parameters
    ----------
    path : str
        Path to the depth image file (e.g., 16-bit PNG or HDF5).

    Returns
    -------
    torch.Tensor
        A single-channel depth image tensor with shape (H, W) in meters.
    """
    if str(path).endswith('.h5'):
        with h5py.File(path, 'r') as depth_file:
            return Tensor(np.array(depth_file['depth']))

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) / 1000
    image = Tensor(image)

    return image


def reshape(image: Tensor, shape: Union[int, tuple, list]) -> Tuple[Tensor, Union[float, tuple]]:
    """
    Resize an image tensor to a specified shape using bilinear interpolation.

    Parameters
    ----------
    image : Tensor
        The image tensor of shape (C, H, W).
    shape : tuple or int
        The desired output shape as (H, W). If an int, both height and width are set to this value.

    Returns
    -------
    tuple[torch.Tensor, Union[float, tuple[float, float]]]
        A tuple containing:
        - Resized image tensor of shape (C, new_H, new_W).
        - Scale factors as a tuple (scale_h, scale_w).
    """
    h, w = image.shape[-2:]
    new_h, new_w = shape
    scale_h = new_h / h
    scale_w = new_w / w

    return torch.nn.functional.interpolate(image[None], size=shape, mode='bilinear', align_corners=False)[0], (scale_h, scale_w)


def resize_long_edge(image: Tensor, max_size: int) -> tuple[Tensor, float]:
    """
    Resize a PyTorch image tensor so that the long edge is `max_size`, preserving the aspect ratio.

    Parameters
    ----------
    image : Tensor
        Input image tensor of shape (C, H, W).
    max_size : int
        The desired size for the long edge of the image.

    Returns
    -------
    resized_image : Tensor
        The resized image tensor of shape (C, new_H, new_W).
    scale : float
        The scaling factor applied to the image dimensions.
    """
    h, w = image.shape[-2:]
    if h > w:
        new_h, new_w = max_size, int(w * max_size / h)
    else:
        new_w, new_h = max_size, int(h * max_size / w)

    scale = new_h / h
    resized = torch.nn.functional.interpolate(
        image[None], size=(new_h, new_w),
        mode='bilinear', align_corners=False
    )[0]
    return resized, scale


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
    numpy.ndarray
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
