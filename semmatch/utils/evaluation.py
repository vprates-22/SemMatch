"""
Module: utils.evaluation
---------------------------------------------

This module provides a set of utilities for camera intrinsics manipulation, 
image resizing, and depth-based point matching across multiple cameras. 
It includes functions for converting camera intrinsics to COLMAP-style 
parameters, resizing images while preserving their aspect ratios, and 
projecting points from one camera to another using depth maps and transformations.

Functions:
----------
- intrinsics_to_camera: Convert a 3x3 camera intrinsics matrix to a dictionary 
  in COLMAP format, including focal lengths, principal point, and image dimensions.
  
- resize_long_edge: Resize an image tensor such that the long edge is scaled 
  to a given maximum size, preserving the aspect ratio.
  
- project_points_between_cameras: Project 2D points with known depth from one camera 
  into another camera's image space and validate the projections based on depth consistency.

"""

import torch
import numpy as np

from typing import Tuple
from numpy.typing import NDArray
from semmatch.evaluation.Scannet1500.utils import map_color_to_depth_coordinates


def intrinsics_to_camera(K: np.ndarray) -> dict:
    """
    Convert a 3x3 intrinsics matrix to a COLMAP-style camera dictionary.

    This function takes a camera intrinsics matrix (K), which contains the focal lengths 
    and principal point coordinates, and converts it into a dictionary representation 
    compatible with the COLMAP format. The dictionary includes basic camera parameters 
    such as the model, image width and height, and intrinsic parameters.

    Parameters
    ----------
    K : np.ndarray
        A 3x3 camera intrinsics matrix.
        The matrix should be of the form:
        [[fx, 0, cx],
         [0, fy, cy],
         [0,  0, 1]]

    Returns
    -------
    dict
        A dictionary containing COLMAP-style camera parameters:
        {
            'model': 'PINHOLE',
            'width': int, 
            'height': int, 
            'params': [fx, fy, cx, cy]
        }

    Notes
    -----
    - The image width and height are inferred from the principal point (cx, cy).
    - The model type is fixed to "PINHOLE", which assumes a pinhole camera model.
    """
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }


def resize_long_edge(image: torch.Tensor, max_size: int) -> tuple:
    """
    Resize a PyTorch image tensor so that the long edge is `max_size`, preserving the aspect ratio.

    This function resizes an image tensor such that the longer side of the image is scaled 
    to the specified `max_size`, and the aspect ratio is maintained.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape (C, H, W), where C is the number of channels, 
        H is the height, and W is the width of the image.
    max_size : int
        The desired size for the long edge (height or width) of the image.

    Returns
    -------
    resized_image : torch.Tensor
        The resized image tensor of shape (C, new_H, new_W), where new_H and new_W are 
        the new height and width of the image after resizing.
    scale : float
        The scaling factor applied to the height (or width) of the image.

    Notes
    -----
    - This function uses bilinear interpolation to resize the image.
    - The `scale` can be used to transform coordinates or to maintain aspect ratio when 
      processing images.
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


def project_points_between_cameras(
    mkpts0: NDArray[np.float32],
    depth0_img: NDArray[np.float32],
    depth1_img: NDArray[np.float32],
    K0: NDArray[np.float32],
    K1: NDArray[np.float32],
    T_0to1: NDArray[np.float32],
    img_shape: tuple[int, int],
    max_depth_diff: float = 0.2
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """
    Projects 2D points with known depth from camera 0 into camera 1 and validates the projections
    using camera 1's depth map and image bounds.

    This function takes 2D points in camera 0, projects them into 3D space using depth 
    information, and then projects them back into camera 1's image space. It then checks 
    whether the projected points are valid (i.e., they lie inside camera 1's image and are 
    depth-consistent).

    Parameters
    ----------
    mkpts0 : NDArray[np.float32]
        (N, 2) Array of points in the image from camera 0, where N is the number of points 
        and each point is represented as (x, y) coordinates in camera 0's image space.
    depth0_img : NDArray[np.float32]
        (H_d, W_d) Depth image from camera 0. Each pixel contains the depth value at that point.
    depth1_img : NDArray[np.float32]
        (H_d, W_d) Depth image from camera 1.
    K0 : NDArray[np.float32]
        (3x3) Intrinsic matrix for camera 0.
    K1 : NDArray[np.float32]
        (3x3) Intrinsic matrix for camera 1.
    T_0to1 : NDArray[np.float32]
        (4x4) Transformation matrix from camera 0 to camera 1. It includes both rotation and translation.
    img_shape : tuple[int, int]
        (height, width) of camera 1's image.
    max_depth_diff : float, optional
        Maximum allowed difference between projected and measured depth in camera 1 (default is 0.2).

    Returns
    -------
    projected_points : NDArray[np.float32]
        (N, 2) Array of (x, y) pixel coordinates in camera 1's image space. Invalid projections 
        are set to NaN.
    valid : NDArray[np.bool_]
        (N,) Boolean mask indicating whether each projection is valid (inside image bounds and depth-consistent).

    Notes
    -----
    - The function assumes the depth images are already in the same resolution as the input images.
    - The transformation matrix `T_0to1` should be in the standard 4x4 homogeneous coordinates format.
    - Depth consistency is checked using the difference between projected and actual depth values.
    """
    H_d, W_d = depth0_img.shape
    H, W = img_shape

    x0s = np.asarray(mkpts0[:, 0])
    y0s = np.asarray(mkpts0[:, 1])
    N = x0s.shape[0]

    map_color_to_depth_coordinates_vetorized = np.vectorize(
        map_color_to_depth_coordinates)

    # Mapeia coordenadas coloridas para coordenadas da imagem de profundidade
    x0_depth, y0_depth = map_color_to_depth_coordinates_vetorized(x0s, y0s)
    x0_depth = x0_depth.astype(int)
    y0_depth = y0_depth.astype(int)

    in_bounds0 = (
        (x0_depth >= 0) & (x0_depth < W_d) &
        (y0_depth >= 0) & (y0_depth < H_d)
    )

    z0s = np.zeros(N, dtype=np.float32)
    z0s[in_bounds0] = depth0_img[y0_depth[in_bounds0], x0_depth[in_bounds0]]
    valid = (z0s > 0) & in_bounds0

    # Reprojeção 3D
    K0_inv = np.linalg.inv(K0)
    x0_h = np.stack([x0s, y0s, np.ones_like(x0s)], axis=1)
    X_c0 = (K0_inv @ x0_h.T).T * z0s[:, np.newaxis]

    R = T_0to1[:3, :3]
    t = T_0to1[:3, 3]
    X_c1 = (R @ X_c0.T).T + t
    X_c1_z = X_c1[:, 2]
    valid &= X_c1_z > 0

    X_c1_norm = np.zeros_like(X_c1)
    X_c1_norm[valid] = X_c1[valid] / X_c1_z[valid, np.newaxis]
    x1_proj = (K1 @ X_c1_norm.T).T
    projected = x1_proj[:, :2]
    x1s, y1s = projected[:, 0], projected[:, 1]

    x1s_rounded = np.round(x1s).astype(int)
    y1s_rounded = np.round(y1s).astype(int)

    in_bounds1 = (
        (x1s_rounded >= 0) & (x1s_rounded < W) &
        (y1s_rounded >= 0) & (y1s_rounded < H)
    )
    valid &= in_bounds1

    # Mapeia novamente para espaço da imagem de profundidade (se necessário)
    x1_depth, y1_depth = map_color_to_depth_coordinates_vetorized(
        x1s_rounded, y1s_rounded)
    x1_depth = x1_depth.astype(int)
    y1_depth = y1_depth.astype(int)

    z1s = np.zeros(N, dtype=np.float32)
    z1s[valid] = depth1_img[y1_depth[valid], x1_depth[valid]]

    depth_consistent = np.abs(X_c1_z - z1s) < max_depth_diff
    valid &= (z1s > 0) & depth_consistent

    projected_points = np.full((N, 2), np.nan, dtype=np.float32)
    projected_points[valid] = projected[valid]

    return projected_points, valid


def apply_homography_to_point(x: int, y: int, H: NDArray) -> Tuple[int, int]:
    point_img0 = np.array([x, y, 1.0])

    point_img1_hom = H @ point_img0

    x_img1 = point_img1_hom[0] / point_img1_hom[2]
    y_img1 = point_img1_hom[1] / point_img1_hom[2]

    return int(x_img1), int(y_img1)


def rescale_homography(H: NDArray, scale_img0: NDArray, scale_img1: NDArray) -> NDArray:
    scale0_x, scale0_y = scale_img0
    scale1_x, scale1_y = scale_img1

    S1 = np.array([[scale0_x, 0, 0],
                   [0, scale0_y, 0],
                   [0, 0, 1]], dtype=float)

    S2 = np.array([[scale1_x, 0, 0],
                   [0, scale1_y, 0],
                   [0, 0, 1]], dtype=float)

    H_scaled = S2 @ H @ np.linalg.inv(S1)

    return H_scaled
