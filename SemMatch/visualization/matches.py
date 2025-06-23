import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple, Union, Optional, List
from matplotlib.patches import ConnectionPatch
from ..helpers import get_inliers
from .utils import DEFAULT_COLORS, save, plot_pair

def plot_matches_parallel(args_plot: Tuple[int, dict, Union[np.ndarray, torch.Tensor],
                                           Union[np.ndarray, torch.Tensor], str]) -> bool:
    """
    Plot matched keypoints between two images and save the result.

    Parameters
    ----------
    args_plot : tuple
        A tuple containing the following elements:
        pair_idx : int
            Index of the image pair.
        pair : dict
            Dictionary with keys 'image0', 'image1', 'K0', 'K1' pointing to image paths and intrinsics.
        mkpts0 : np.ndarray or torch.Tensor
            Matched keypoints from the first image.
        mkpts1 : np.ndarray or torch.Tensor
            Matched keypoints from the second image.
        out_folder : str
            Output folder to save the visualization.

    Returns
    -------
    bool
        True if the plot was successfully generated and saved.
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
                   keypoints1: Optional[Union[np.ndarray, torch.Tensor]] = None,
                   color: Optional[str] = None,
                   kps_size: int = 5,
                   all_colors: Optional[List] = None,
                   **kwargs) -> Tuple[Figure, List[Axes]]:
    """
    Plot keypoints on one or both images.

    Parameters
    ----------
    keypoints0 : np.ndarray or torch.Tensor, optional
        Keypoints for the first image.
    keypoints1 : np.ndarray or torch.Tensor, optional
        Keypoints for the second image.
    color : str, optional
        Name of default color to use from DEFAULT_COLORS.
    kps_size : int, optional
        Size of the keypoints in the plot. Default is 5.
    all_colors : list, optional
        List of colors for each keypoint.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib `scatter`.
    
    Returns
    -------
    None
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

        ax[idx].scatter(keypoints[:, 0], keypoints[:, 1], s=kps_size, c=colors, **kwargs)

def plot_matches(mkpts0: Union[np.ndarray, torch.Tensor],
                 mkpts1: Union[np.ndarray, torch.Tensor],
                 color: Union[str, List, np.ndarray] = 'b',
                 alphas: Optional[List[float]] = None,
                 **kwargs) -> None:
    """
    Draw lines connecting matched keypoints between two images.

    Parameters
    ----------
    mkpts0 : np.ndarray or torch.Tensor
        Matched keypoints in the first image.
    mkpts1 : np.ndarray or torch.Tensor
        Matched keypoints in the second image.
    color : str or list, optional
        Color name or list of colors for the match lines. Default is 'b' (blue).
    alphas : list of float, optional
        Per-match alpha transparency values.
    **kwargs : dict
        Additional arguments passed to `ConnectionPatch`.

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