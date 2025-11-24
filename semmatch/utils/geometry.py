"""
Module: utils.geometry
---------------------------------------------

This module provides a set of utilities for geometric computer vision, including
camera intrinsics manipulation, image resizing, and transformations for 2D and 3D points.

It includes functions for:
- Converting camera intrinsics to different formats.
- Resizing images while preserving aspect ratio.
- Projecting points between camera views using depth information.
- Applying and rescaling homography transformations.
- Finding inliers in point correspondences using RANSAC.
"""

import poselib
import numpy as np

from typing import Iterable
from numpy.typing import NDArray


def intrinsics_to_camera(K: NDArray) -> dict:
    """
    Convert a 3x3 intrinsics matrix to a COLMAP-style camera dictionary.

    This function takes a camera intrinsics matrix (K) and converts it into a
    dictionary representation compatible with the COLMAP format.

    Parameters
    ----------
    K : NDArray
        A 3x3 camera intrinsics matrix of the form:
        [[fx, 0, cx],
         [0, fy, cy],
         [0,  0, 1]]

    Returns
    -------
    dict
        A dictionary containing COLMAP-style camera parameters:
        {'model': 'PINHOLE', 'width': int, 'height': int, 'params': [fx, fy, cx, cy]}

    Notes
    -----
    - The image width and height are inferred from the principal point (cx, cy).
    - The camera model is assumed to be "PINHOLE".
    """
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }


def map_color_to_depth_coordinates(
    x: NDArray,
    y: NDArray,
    color_shape: tuple[int, int],
    depth_shape: tuple[int, int]
) -> tuple[NDArray, NDArray]:
    """
    Vectorized mapping from color image coordinates to depth image coordinates.

    Parameters
    ----------
    x, y : NDArray
        Pixel coordinates in the color image space.
    color_shape : tuple[int, int]
        Resolution of the RGB image (H_color, W_color).
    depth_shape : tuple[int, int]
        Resolution of the depth image (H_depth, W_depth).

    Returns
    -------
    x_d, y_d : NDArray
        Corresponding coordinates in the depth map as integer arrays.
    """
    Hc, Wc = color_shape
    Hd, Wd = depth_shape

    x_d = (x * (Wd / Wc)).astype(np.int32, copy=False)
    y_d = (y * (Hd / Hc)).astype(np.int32, copy=False)

    return x_d, y_d


def unscale_points(mkpts: NDArray[np.float32], scale: Iterable[float]) -> NDArray[np.float32]:
    """
    Unscale 2D points according to a given image scale.

    Parameters
    ----------
    mkpts : NDArray[np.float32]
        (N, 2) Array of points in the image.
    scale : Iterable[float]
        (scale_h, scale_w) scaling factors.

    Returns
    -------
    NDArray[np.float32]
        Unscaled points.
    """
    scale_h, scale_w = scale
    return np.column_stack((mkpts[:, 0] / scale_w, mkpts[:, 1] / scale_h))


def depth_lookup(points: NDArray[np.float32], depth_img: NDArray[np.float32],
                 img_shape: tuple[int, int]) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """
    Retrieve depth values from a depth image for given points.

    Parameters
    ----------
    points : NDArray[np.float32]
        (N, 2) Array of 2D points.
    depth_img : NDArray[np.float32]
        Depth image.
    img_shape : tuple[int, int]
        Shape of the original color image (H, W).

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.bool_]]
        A tuple containing depth values and a validity mask.
    """
    H, W = img_shape
    Hd, Wd = depth_img.shape

    x, y = points[:, 0], points[:, 1]
    x_d, y_d = map_color_to_depth_coordinates(x, y, img_shape, depth_img.shape)

    in_bounds = (
        (x >= 0) & (x < W) &
        (y >= 0) & (y < H) &
        (x_d >= 0) & (x_d < Wd) &
        (y_d >= 0) & (y_d < Hd)
    )

    z = np.zeros(points.shape[0], np.float32)
    z[in_bounds] = depth_img[y_d[in_bounds], x_d[in_bounds]]

    valid = (z > 0) & in_bounds
    return z, valid


def backproject_points(points: NDArray[np.float32], depths: NDArray[np.float32],
                       K: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Backproject 2D points to 3D space using camera intrinsics.

    Parameters
    ----------
    points : NDArray[np.float32]
        (N, 2) 2D points.
    depths : NDArray[np.float32]
        Depth values for each point.
    K : NDArray[np.float32]
        Camera intrinsic matrix.

    Returns
    -------
    NDArray[np.float32]
        (N, 3) 3D points in camera coordinates.
    """
    K_inv = np.linalg.inv(K)
    N = points.shape[0]
    pts_h = np.column_stack((points, np.ones(N, np.float32)))
    return (K_inv @ pts_h.T).T * depths[:, None]


def transform_points(P: NDArray[np.float32], T: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Transform 3D points from one camera to another using a 4x4 transformation.

    Parameters
    ----------
    P : NDArray[np.float32]
        (N, 3) 3D points.
    T : NDArray[np.float32]
        (4, 4) Transformation matrix.

    Returns
    -------
    NDArray[np.float32]
        Transformed 3D points.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ P.T).T + t


def project_points(P: NDArray[np.float32], K: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Project 3D points into 2D image space using camera intrinsics.

    Parameters
    ----------
    P : NDArray[np.float32]
        (N, 3) 3D points.
    K : NDArray[np.float32]
        Camera intrinsic matrix.

    Returns
    -------
    NDArray[np.float32]
        (N, 2) 2D projected points.
    """
    Z = P[:, 2:3]
    P_norm = P / (Z + 1e-8)  # Avoid division by zero
    pix = (K @ P_norm.T).T
    return pix[:, :2]


def project_points_between_cameras(
    mkpts0: NDArray[np.float32],
    depth0_img: NDArray[np.float32],
    depth1_img: NDArray[np.float32],
    K0: NDArray[np.float32],
    K1: NDArray[np.float32],
    T_0to1: NDArray[np.float32],
    img0_shape: tuple[int, int],
    img1_shape: tuple[int, int],
    scale_img0: Iterable[float] = (1.0, 1.0),
    scale_img1: Iterable[float] = (1.0, 1.0),
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
        (N, 2) Array of points in the image from camera 0.
    depth0_img : NDArray[np.float32]
        (H_d, W_d) Depth image from camera 0.
    depth1_img : NDArray[np.float32]
        (H_d, W_d) Depth image from camera 1.
    K0 : NDArray[np.float32]
        (3x3) Intrinsic matrix for camera 0.
    K1 : NDArray[np.float32]
        (3x3) Intrinsic matrix for camera 1.
    T_0to1 : NDArray[np.float32]
        (4x4) Transformation matrix from camera 0 to camera 1.
    img0_shape : tuple[int, int]
        (height, width) of camera 0's image.
    img1_shape : tuple[int, int]
        (height, width) of camera 1's image.
    scale_img0 : Iterable[float], optional
        Scaling factors for image 0 as (scale_h, scale_w). Defaults to (1.0, 1.0).
    scale_img1 : Iterable[float], optional
        Scaling factors for image 1 as (scale_h, scale_w). Defaults to (1.0, 1.0).
    max_depth_diff : float, optional
        Maximum allowed difference between projected and measured depth in camera 1. Defaults to 0.2.

    Returns
    -------
    projected_points : NDArray[np.float32]
        (N, 2) Array of (x, y) pixel coordinates in camera 1's image space. Invalid projections are NaN.
    valid : NDArray[np.bool_]
        (N,) Boolean mask indicating whether each projection is valid.
    """
    # Convert inputs to NumPy arrays with float32 type
    mkpts0 = np.asarray(mkpts0, np.float32)
    depth0_img = np.asarray(depth0_img, np.float32)
    depth1_img = np.asarray(depth1_img, np.float32)
    K0 = np.asarray(K0, np.float32)
    K1 = np.asarray(K1, np.float32)
    T_0to1 = np.asarray(T_0to1, np.float32)

    scale_h1, scale_w1 = scale_img1

    # 1. Unscale points
    points0 = unscale_points(mkpts0, scale_img0)

    # 2. Depth lookup in camera 0
    z0, valid = depth_lookup(points0, depth0_img, img0_shape)
    if not np.any(valid):
        return np.full((len(points0), 2), np.nan, np.float32), valid

    # 3. Backproject to 3D in camera 0's coordinate system
    P0 = backproject_points(points0, z0, K0)

    # 4. Transform points to camera 1's coordinate system
    P1 = transform_points(P0, T_0to1)

    # 5. Project to 2D in camera 1
    x1 = project_points(P1, K1)

    # 6. Check bounds and depth validity
    H1, W1 = img1_shape
    in_bounds = (
        (x1[:, 0] >= 0) & (x1[:, 0] < W1) &
        (x1[:, 1] >= 0) & (x1[:, 1] < H1) &
        (P1[:, 2] > 0)
    )
    valid &= in_bounds

    # 7. Depth consistency check
    x1d, y1d = map_color_to_depth_coordinates(
        x1[:, 0], x1[:, 1], img1_shape, depth1_img.shape)
    idx = valid & (x1d >= 0) & (x1d < depth1_img.shape[1]) & (
        y1d >= 0) & (y1d < depth1_img.shape[0])

    z1 = np.zeros(len(points0), np.float32)
    z1[idx] = depth1_img[y1d[idx], x1d[idx]]

    valid &= (z1 > 0) & (np.abs(P1[:, 2] - z1) < max_depth_diff)

    # 8. Scale output points for final result
    proj = np.full((len(points0), 2), np.nan, np.float32)
    proj[valid, 0] = x1[valid, 0] * scale_w1
    proj[valid, 1] = x1[valid, 1] * scale_h1

    return proj, valid


def map_points_between_images(
    points: NDArray,
    H: NDArray,
    img1_shape: tuple[int, int],
    scale_img0: Iterable[float] = [1.0, 1.0],
    scale_img1: Iterable[float] = [1.0, 1.0]
) -> tuple[NDArray, NDArray]:
    """
    Map 2D points from one image to another using a homography matrix.

    Parameters
    ----------
    points : NDArray
        (N, 2) Array of 2D points in the source image.
    H : NDArray
        (3x3) Homography matrix for mapping points between images.
    img1_shape : tuple[int, int]
        (height, width) of the target image.
    scale_img0 : Iterable[float], optional
        Scaling factors for the source image as (scale_h, scale_w). Defaults to (1.0, 1.0).
    scale_img1 : Iterable[float], optional
        Scaling factors for the target image as (scale_h, scale_w). Defaults to (1.0, 1.0).

    Returns
    -------
    mapped : NDArray
        (N, 2) Array of mapped 2D points in the target image.
    valid : NDArray
        (N,) Boolean mask indicating whether each mapped point is valid (inside bounds).
    """
    # Convert inputs to NumPy arrays with float32 type
    H = np.asarray(H, np.float32)
    points = np.asarray(points, np.float32)
    scale_img0 = np.asarray(scale_img0, np.float32)
    scale_img1 = np.asarray(scale_img1, np.float32)

    # Rescale homography if necessary
    if not np.all(scale_img0 == 1.0) or not np.all(scale_img1 == 1.0):
        H = rescale_homography(H, scale_img0, scale_img1)

    H_img, W_img = img1_shape
    scale_h1, scale_w1 = scale_img1

    # Scale bounds
    H_img_scaled = int(H_img * scale_h1)
    W_img_scaled = int(W_img * scale_w1)

    # Apply homography to all points
    mapped = apply_homography(points, H)

    # Validate points inside scaled bounds
    x, y = mapped[:, 0], mapped[:, 1]
    valid = (x >= 0) & (x < W_img_scaled) & (y >= 0) & (y < H_img_scaled)

    # Points outside bounds are marked as NaN
    mapped[~valid] = np.nan

    return mapped, valid


def apply_homography(points: NDArray, H: NDArray) -> NDArray:
    """
    Apply a homography transformation to a set of 2D points.

    Parameters
    ----------
    points : NDArray
        (N, 2) Array of 2D points to be transformed.
    H : NDArray
        (3x3) Homography matrix.

    Returns
    -------
    NDArray
        (N, 2) Array of transformed 2D points.
    """
    n = points.shape[0]
    pts_h = np.hstack([points, np.ones((n, 1))])
    mapped = pts_h @ H.T
    mapped /= (mapped[:, 2:3] + 1e-8)  # Avoid division by zero
    return mapped[:, :2]


def rescale_homography(H: NDArray, scale_img0: NDArray, scale_img1: NDArray) -> NDArray:
    """
    Rescale a homography matrix to account for differences in image scaling.

    Parameters
    ----------
    H : NDArray
        (3x3) Homography matrix to be rescaled.
    scale_img0 : NDArray
        (2,) Scaling factors for the source image (scale_h, scale_w).
    scale_img1 : NDArray
        (2,) Scaling factors for the target image (scale_h, scale_w).

    Returns
    -------
    NDArray
        (3x3) Rescaled homography matrix.
    """
    scale0_h, scale0_w = scale_img0
    scale1_h, scale1_w = scale_img1

    S0 = np.diag([scale0_w, scale0_h, 1.0])
    S1 = np.diag([scale1_w, scale1_h, 1.0])

    return S1 @ H @ np.linalg.inv(S0)


def get_inliers_ransac(
    mkpts0: NDArray,
    mkpts1: NDArray,
    K0: NDArray,
    K1: NDArray,
    threshold: float = 6.0,
) -> NDArray:
    """
    Estimates the relative pose between two sets of matched keypoints and returns inliers.

    Uses pose estimation with RANSAC to identify geometrically consistent matches between two views.

    Parameters
    ----------
    mkpts0 : NDArray
        Matched keypoints from the first image, shape (N, 2).
    mkpts1 : NDArray
        Matched keypoints from the second image, shape (N, 2).
    K0 : NDArray
        Intrinsics matrix for the first camera (3x3).
    K1 : NDArray
        Intrinsics matrix for the second camera (3x3).

    Returns
    -------
    NDArray
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
            'max_epipolar_error': threshold,
        }
    )

    return np.array(details['inliers'], dtype=bool)


def estimate_pose(
    mkpts0: NDArray,
    mkpts1: NDArray,
    K0: NDArray,
    K1: NDArray,
    threshold: float = 6.0,
) -> tuple[NDArray, NDArray]:
    """
    Estimates the relative pose and finds inliers.
    Returns the pose and the inlier mask.
    """
    pose, _ = poselib.estimate_relative_pose(
        mkpts0.tolist(),
        mkpts1.tolist(),
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
        ransac_opt={
            'max_iterations': 10000,
            'success_prob': 0.99999,
            'max_epipolar_error': threshold,
        }
    )

    if pose is None:
        return None, None

    return pose.R, pose.t
