"""
Module: visualization.matches
-----------------------------------

This module provides utilities for visualizing keypoint matches between image pairs, 
including drawing keypoints, visualizing inliers/outliers, and saving annotated 
match images for evaluation and analysis purposes.

Functions:
----------
- plot_matches_parallel(args_plot):
    Plots and saves matches for a single image pair, coloring inliers and outliers 
    differently.

- plot_keypoints(keypoints0, keypoints1, ...):
    Plots keypoints overlaid on the input images, supporting custom colors and sizes.

- plot_matches(mkpts0, mkpts1, ...):
    Draws connecting lines between matched keypoints across two images, optionally 
    using per-match colors and alpha transparency.fu

Usage:
------
These utilities are primarily used in the context of keypoint detection and 
matching pipelines, where visualizing the quality of matches and inlier distributions 
is crucial for qualitative evaluation. Typical usage involves calling 
`plot_matches_parallel` in a loop over image pairs, optionally using multiprocessing.
"""

import os
from typing import Tuple, Union, Optional, List

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from ..helpers import get_inliers
from ..utils.visualization import DEFAULT_COLORS, save, plot_pair


def plot_matches_parallel(args_plot: Tuple[int, dict, Union[np.ndarray, torch.Tensor],
                                           Union[np.ndarray, torch.Tensor], str]) -> bool:
    """
    Plot matched keypoints between two images and save the visualization.

    This function reads the image pair, separates inlier and outlier matches using
    epipolar geometry (via `get_inliers`), visualizes both categories with different 
    colors, and saves the result to the specified output directory.

    Parameters
    ----------
    args_plot : tuple
        A tuple containing the following:
            - pair_idx (int): Index of the image pair.
            - pair (dict): Dictionary with keys 'image0', 'image1', 'K0', 'K1' for paths and intrinsics.
            - mkpts0 (np.ndarray or torch.Tensor): Matched keypoints from image0.
            - mkpts1 (np.ndarray or torch.Tensor): Matched keypoints from image1.
            - out_folder (str): Directory where the visualization image will be saved.

    Returns
    -------
    bool
        True if the image was successfully saved.
    """
    pair_idx, pair, mkpts0, mkpts1, out_folder = args_plot
    out_path = os.path.join(out_folder, f'{pair_idx}.png')

    image0 = cv2.imread(pair['image0'])
    image1 = cv2.imread(pair['image1'])

    if isinstance(mkpts0, torch.Tensor):
        mkpts0 = mkpts0.cpu().numpy()
        mkpts1 = mkpts1.cpu().numpy()

    inliers = get_inliers(mkpts0, mkpts1, pair['K0'], pair['K1'])

    plot_pair(image0, image1)
    plot_matches(mkpts0[~inliers], mkpts1[~inliers], color='r')
    plot_matches(mkpts0[inliers], mkpts1[inliers], color='g')
    save(out_path)

    return True


def plot_keypoints(keypoints0: Optional[Union[np.ndarray, torch.Tensor]] = None,
                   keypoints1: Optional[Union[np.ndarray,
                                              torch.Tensor]] = None,
                   color: Optional[str] = None,
                   kps_size: int = 5,
                   all_colors: Optional[List] = None,
                   **kwargs) -> Tuple[Figure, List[Axes]]:
    """
    Overlay keypoints on one or both images in a side-by-side plot.

    Useful for visualizing detected keypoints in image matching tasks. If a default
    color is not provided, keypoints will be colorized using a rainbow colormap.

    Parameters
    ----------
    keypoints0 : np.ndarray or torch.Tensor, optional
        Keypoints for the first (left) image.
    keypoints1 : np.ndarray or torch.Tensor, optional
        Keypoints for the second (right) image.
    color : str, optional
        Key to a color in DEFAULT_COLORS (e.g., 'g', 'r', 'b'). If not set, uses a rainbow map.
    kps_size : int, optional
        Size of the keypoints (scatter dots). Default is 5.
    all_colors : list, optional
        List of custom colors for individual keypoints.
    **kwargs : dict
        Extra keyword arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plotted images and keypoints.
    ax : list of matplotlib.axes.Axes
        The axes corresponding to each image.
    """
    rainbow = plt.get_cmap('hsv')

    if fig is None or ax is None:
        fig = plt.gcf()
        ax = fig.axes

    for idx, keypoints in enumerate([keypoints0, keypoints1]):
        if keypoints is None:
            continue

        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.detach().cpu().numpy()
        if len(keypoints.shape) == 3:
            keypoints = keypoints.squeeze(0)

        n = len(keypoints)
        colors = all_colors if all_colors is not None else \
            [DEFAULT_COLORS[color]] * n if color in DEFAULT_COLORS else \
            [rainbow(i / n) for i in range(n)]

        ax[idx].scatter(keypoints[:, 0], keypoints[:, 1],
                        s=kps_size, c=colors, **kwargs)


def plot_matches(mkpts0: Union[np.ndarray, torch.Tensor],
                 mkpts1: Union[np.ndarray, torch.Tensor],
                 color: Union[str, List, np.ndarray] = 'b',
                 alphas: Optional[List[float]] = None,
                 **kwargs) -> None:
    """
    Draw connecting lines between matched keypoints on a side-by-side image plot.

    Can be used to differentiate inliers and outliers visually by using different colors.

    Parameters
    ----------
    mkpts0 : np.ndarray or torch.Tensor
        Matched keypoints in the first (left) image.
    mkpts1 : np.ndarray or torch.Tensor
        Corresponding matched keypoints in the second (right) image.
    color : str or list or np.ndarray, optional
        Match line color(s). If a color string (e.g., 'g', 'r', 'b') is passed and exists
        in DEFAULT_COLORS, that color is used. Otherwise, a rainbow colormap is used per line.
    alphas : list of float, optional
        Transparency values (0.0–1.0) for each line. Useful for fading weak matches.
    **kwargs : dict
        Extra arguments passed to `matplotlib.patches.ConnectionPatch`.

    Returns
    -------
    None
    """
    fig = plt.gcf()
    ax = fig.axes

    if isinstance(color, str) and color in DEFAULT_COLORS:
        color = [DEFAULT_COLORS[color]] * len(mkpts0)
    elif not isinstance(color, (list, np.ndarray)):
        color = [color] * len(mkpts0)

    if isinstance(mkpts0, torch.Tensor):
        mkpts0 = mkpts0.detach().cpu().numpy()
        mkpts1 = mkpts1.detach().cpu().numpy()

    for i, (mkp0, mkp1) in enumerate(zip(mkpts0, mkpts1)):
        alpha = kwargs.pop('alpha', alphas[i] if alphas is not None else 1)
        con = ConnectionPatch(
            xyA=mkp0,
            xyB=mkp1,
            coordsA="data",
            coordsB="data",
            axesA=ax[0],
            axesB=ax[1],
            color=color[i],
            linewidth=1.5,
            alpha=alpha,
            **kwargs
        )
        con.set_in_layout(False)
        ax[0].add_artist(con)

    ax[1].set_zorder(-1)
