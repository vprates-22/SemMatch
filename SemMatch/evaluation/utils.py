import torch
import numpy as np

from numpy.typing import NDArray
from ..evaluation.Scannet1500.utils import map_color_to_depth_point

def intrinsics_to_camera(K: np.ndarray) -> dict:
    """
    Convert a 3x3 intrinsics matrix to a COLMAP-style camera dictionary.

    Parameters
    ----------
    K : np.ndarray
        3x3 camera intrinsics matrix.

    Returns
    -------
    dict
        Dictionary containing COLMAP-style camera parameters:
        {
            'model': 'PINHOLE',
            'width': int,
            'height': int,
            'params': [fx, fy, cx, cy]
        }
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
    Resize a PyTorch image tensor so that the long edge is `max_size`, preserving aspect ratio.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape (C, H, W).
    max_size : int
        Desired size of the long edge.

    Returns
    -------
    resized_image : torch.Tensor
        Resized image tensor of shape (C, new_H, new_W).
    scale : float
        Scaling factor applied to the height.
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

def find_matching_points(
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

    Args:
        x0s: (N,) x-coordinates of the points in camera 0 image space.
        y0s: (N,) y-coordinates of the points in camera 0 image space.
        depth0_img: (H_d, W_d) depth image of camera 0.
        depth1_img: (H_d, W_d) depth image of camera 1.
        K0: (3x3) intrinsic matrix of camera 0.
        K1: (3x3) intrinsic matrix of camera 1.
        T_0to1: (4x4) transformation from camera 0 to camera 1.
        img_shape: (height, width) of camera 1's image.
        max_depth_diff: Maximum allowed difference between projected and measured depth in camera 1.

    Returns:
        projected_points: (N, 2) array of projected (x1, y1) pixel coordinates in camera 1.
                Invalid projections are set to NaN.
        valid: (N,) boolean mask indicating valid projections (inside image + depth consistent).
    """  
    H_d, W_d = depth0_img.shape
    H, W = img_shape

    x0s = np.asarray(mkpts0[:, 0])
    y0s = np.asarray(mkpts0[:, 1])
    N = x0s.shape[0]

    map_color_to_depth_point_vetorized = np.vectorize(map_color_to_depth_point)

    # Mapeia coordenadas coloridas para coordenadas da imagem de profundidade
    x0_depth, y0_depth = map_color_to_depth_point_vetorized(x0s, y0s)
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
    x1_depth, y1_depth = map_color_to_depth_point_vetorized(x1s_rounded, y1s_rounded)
    x1_depth = x1_depth.astype(int)
    y1_depth = y1_depth.astype(int)

    z1s = np.zeros(N, dtype=np.float32)
    z1s[valid] = depth1_img[y1_depth[valid], x1_depth[valid]]

    depth_consistent = np.abs(X_c1_z - z1s) < max_depth_diff
    valid &= (z1s > 0) & depth_consistent

    projected_points = np.full((N, 2), np.nan, dtype=np.float32)
    projected_points[valid] = projected[valid]

    return projected_points, valid