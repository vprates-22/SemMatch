"""
Module: helpers
----------------

This module provides utility functions for loading and processing images, converting
between PyTorch and OpenCV formats, and estimating inliers from keypoint matches.

Functions:
----------
- load_image(path, gray=False): 
    Loads an RGB or grayscale image from disk into a normalized PyTorch tensor.

- load_depth(path): 
    Loads a depth image from disk into a PyTorch tensor.

- get_inliers(mkpts0, mkpts1, K0, K1): 
    Estimates inliers from matched keypoints using epipolar geometry via pose estimation.

- to_cv(torch_image, convert_color=True, batch_idx=0, to_gray=False): 
    Converts a PyTorch image tensor to a NumPy array in OpenCV-compatible format.

- to_tensor(img_np): 
    Converts a NumPy image array to a normalized PyTorch tensor suitable for model input.
"""

import cv2
import torch
import torchvision
import numpy as np

import poselib
from semmatch.utils.geometry import intrinsics_to_camera


def load_image_torch(path, gray=False) -> torch.Tensor:
    """
    Loads an RGB or grayscale image from disk as a PyTorch tensor.

    Parameters
    ----------
    path : str or Path
        File path to the image.
    gray : bool, optional
        If True, converts the image to grayscale.

    Returns
    -------
    torch.Tensor
        Tensor of shape (C, H, W) with values in [0, 1], where C is 1 (grayscale) or 3 (RGB).
    """
    image = torchvision.io.read_image(str(path)).float() / 255
    if gray:
        image = torchvision.transforms.functional.rgb_to_grayscale(image)
    return image


def load_image_numpy(path, gray=False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR_RGB
    image = cv2.imread(path, flag)

    return image


def load_depth(path) -> torch.Tensor:
    """
    Loads a depth image from disk as a single-channel PyTorch tensor.

    Parameters
    ----------
    path : str or Path
        Path to the depth image file (typically .png with 16-bit depth).

    Returns
    -------
    torch.Tensor
        Tensor of shape (1, H, W) with depth values as float32.
    """
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    image = torch.tensor(image).unsqueeze(0)
    return image


def get_inliers(mkpts0: np.ndarray,
                mkpts1: np.ndarray,
                K0: np.ndarray,
                K1: np.ndarray) -> np.ndarray:
    """
    Estimates the relative pose between two sets of matched keypoints and returns inliers.

    Uses pose estimation with RANSAC to identify geometrically consistent matches between two views.

    Parameters
    ----------
    mkpts0 : np.ndarray
        Matched keypoints from the first image, shape (N, 2).
    mkpts1 : np.ndarray
        Matched keypoints from the second image, shape (N, 2).
    K0 : np.ndarray
        Intrinsics matrix for the first camera (3x3).
    K1 : np.ndarray
        Intrinsics matrix for the second camera (3x3).

    Returns
    -------
    np.ndarray
        Boolean array of shape (N,) indicating which keypoint matches are inliers.
    """
    _, details = poselib.estimate_relative_pose(
        mkpts0.tolist(),
        mkpts1.tolist(),
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
        ransac_opt={
            'max_iterations': 10000,
            'success_prob': 0.99999,
            'max_epipolar_error': 6.0,
        }
    )

    return np.array(details['inliers']).astype(bool)


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


def to_tensor(img_np):
    """
    Converts a NumPy image array (H, W, C) into a normalized PyTorch tensor.

    Pixel values are scaled to [-1, 1] and permuted to (1, C, H, W).

    Parameters
    ----------
    img_np : np.ndarray or torch.Tensor
        Input image. If already a tensor, returns as-is.

    Returns
    -------
    torch.Tensor
        Normalized image tensor of shape (1, C, H, W).
    """
    if isinstance(img_np, torch.Tensor):
        return img_np

    img_np = img_np.astype(np.float32) / 255.0
    img_np = img_np * 2 - 1
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    return img_tensor
