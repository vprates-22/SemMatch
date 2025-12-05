"""
Module: semmatch.utils.visualization
------------------------------------

This module provides a suite of functions for visualizing computer vision data,
such as images, keypoints, matches, and segmentation masks, using Matplotlib.
It is designed to facilitate the inspection and debugging of image matching and
segmentation results.

Functions
---------
plot_pair(img0, img1, figsize, title)
    Plots a pair of images side-by-side.
show()
    Displays the current Matplotlib figure.
save(path, **kwargs)
    Saves the current Matplotlib figure to a file.
plot_matches_parallel(args_plot)
    A wrapper for plotting matches in a parallel processing context.
plot_keypoints(mkpts0, mkpts1, color, kps_size, all_colors, **kwargs)
    Plots keypoints on two images.
plot_matches(mkpts0, mkpts1, color, alphas, **kwargs)
    Draws lines connecting matched keypoints between two images.
plot_masks(mask0, mask1, color, alpha, color_it)
    Overlays segmentation masks on a pair of images.
"""
import os
from typing import Any, List, Optional, Tuple, Union

import cv2

import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from numpy.typing import NDArray

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from matplotlib import colors as mcolors

from semmatch.utils.image import to_cv

DEFAULT_COLORS = {
    'g': '#4ade80',
    'r': '#ef4444',
    'b': '#3b82f6',
}


def plot_pair(
    img0: NDArray,
    img1: NDArray,
    figsize: Tuple[int, int] = (20, 10),
    title: Optional[str] = None
) -> Tuple[Figure, List[Axes]]:
    """
    Plots a pair of images side-by-side.

    Parameters
    ----------
    img0 : NDArray
        The first image as a NumPy array.
    img1 : NDArray
        The second image as a NumPy array.
    figsize : Tuple[int, int], optional
        The size of the figure (width, height) in inches. Defaults to (20, 10).
    title : Optional[str], optional
        An optional title for the figure. Defaults to None.

    Returns
    -------
    Tuple[Figure, List[Axes]]
        A tuple containing the Matplotlib Figure object and a list of Axes objects.
    """

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].imshow(to_cv(img0))
    ax[1].imshow(to_cv(img1))

    for a in ax:
        a.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    if title is not None:
        fig.suptitle(title)

    return fig, ax


def show() -> None:
    """
    Displays the current Matplotlib figure.

    This function turns off the axis, adjusts the layout to be tight,
    and then displays the plot.
    """
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def save(path: str, **kwargs: Any) -> None:
    """
    Saves the current Matplotlib figure to a specified path.

    Parameters
    ----------
    path : str
        The file path to save the figure to.
    **kwargs : Any
        Additional keyword arguments to pass to `matplotlib.pyplot.savefig`.
    """
    plt.tight_layout()
    plt.savefig(path, **kwargs)
    plt.close()


def plot_matches_parallel(args_plot: Tuple[int, dict, Union[NDArray, Tensor],
                                           Union[NDArray, Tensor], str]) -> bool:
    """
    Plots matches between two images in parallel processing context.

    This function is designed to be called in a parallel processing environment.
    It reads two images, plots keypoint matches (inliers and outliers) between them,
    and saves the resulting plot to a specified folder.

    Parameters
    ----------
    args_plot : Tuple[int, dict, Union[NDArray, Tensor],
        Union[NDArray, Tensor], str]
        A tuple containing:
        - pair_idx (int): The index of the image pair.
        - pair (dict): A dictionary containing information about the image pair,
        including 'image0' (path to first image), 'image1' (path to second image),
        'K0' (intrinsics for image0), and 'K1' (intrinsics for image1).
        - mkpts0 (Union[NDArray, Tensor]): Matched keypoints from the first image.
        - mkpts1 (Union[NDArray, Tensor]): Matched keypoints from the second image.
        - out_folder (str): The directory to save the output plot.

    Returns
    -------
    bool
        True if the plotting and saving were successful.
    """
    pair_idx, pair, mkpts0, mkpts1, out_folder = args_plot
    out_path = os.path.join(out_folder, f'{pair_idx}.png')

    image0 = cv2.imread(pair['image0'])
    image1 = cv2.imread(pair['image1'])

    if isinstance(mkpts0, Tensor):
        mkpts0 = mkpts0.cpu().numpy()
        mkpts1 = mkpts1.cpu().numpy()

    inliers = get_inliers(
        mkpts0, mkpts1, pair['K0'], pair['K1'])

    plot_pair(image0, image1)
    plot_matches(mkpts0[~inliers], mkpts1[~inliers], color='r')
    plot_matches(mkpts0[inliers], mkpts1[inliers], color='g')
    save(out_path)

    return True


def plot_keypoints(
    mkpts0: Optional[Union[NDArray, Tensor]] = None,
    mkpts1: Optional[Union[NDArray, Tensor]] = None,
    color: Optional[str] = None,
    kps_size: int = 5,
    all_colors: Optional[List] = None,
    **kwargs
) -> None:
    """
    Plots keypoints on the current Matplotlib axes.

    This function takes two sets of keypoints (for two images) and plots them
    on the respective subplots of the current figure. Keypoints can be colored
    uniformly or with a rainbow gradient.

    Parameters
    ----------
    mkpts0 : Optional[Union[NDArray, Tensor]], optional
        Keypoints for the first image. Defaults to None.
    mkpts1 : Optional[Union[NDArray, Tensor]], optional
        Keypoints for the second image. Defaults to None.
    color : Optional[str], optional
        A default color for all keypoints if `all_colors` is not provided.
        Uses predefined colors from `DEFAULT_COLORS` or a single Matplotlib color string.
        Defaults to None.
    kps_size : int, optional
        The size of the keypoint markers. Defaults to 5.
    all_colors : Optional[List], optional
        A list of colors, one for each keypoint. If provided, `color` is ignored.
        Defaults to None.
    **kwargs : Any
        Additional keyword arguments to pass to `matplotlib.pyplot.scatter`.
    """
    rainbow = plt.get_cmap('hsv')

    fig = plt.gcf()
    ax = fig.axes

    for idx, keypoints in enumerate([mkpts0, mkpts1]):
        if keypoints is None:
            continue

        if isinstance(keypoints, Tensor):
            keypoints = keypoints.detach().cpu().numpy()
        if len(keypoints.shape) == 3:
            keypoints = keypoints.squeeze(0)

        n = len(keypoints)
        colors = all_colors if all_colors is not None else \
            [DEFAULT_COLORS[color]] * n if color in DEFAULT_COLORS else \
            [rainbow(i / n) for i in range(n)]

        ax[idx].scatter(keypoints[:, 0], keypoints[:, 1],
                        s=kps_size, c=colors, **kwargs)


def plot_matches(
    mkpts0: Union[NDArray, Tensor],
    mkpts1: Union[NDArray, Tensor],
    color: Union[str, List, NDArray] = 'b',
    alphas: Optional[List[float]] = None,
    **kwargs
) -> None:
    """
    Plots matches between two sets of keypoints on two subplots.

    This function draws lines connecting corresponding keypoints between two images
    displayed in separate subplots. Matches can be colored individually or uniformly.

    Parameters
    ----------
    mkpts0 : Union[NDArray, Tensor]
        Matched keypoints from the first image, shape (N, 2).
    mkpts1 : Union[NDArray, Tensor]
        Matched keypoints from the second image, shape (N, 2).
    color : Union[str, List, NDArray], optional
        The color(s) for the match lines. Can be a single string (e.g., 'b', 'r', 'g'
        using `DEFAULT_COLORS` or any Matplotlib color), a list of colors (one per match),
        or a NumPy array of colors. Defaults to 'b'.
    alphas : Optional[List[float]], optional
        A list of alpha (transparency) values, one for each match. If None,
        a default alpha of 1 is used. Defaults to None.
    **kwargs : Any
        Additional keyword arguments to pass to `matplotlib.patches.ConnectionPatch`.
    """
    fig = plt.gcf()
    ax = fig.axes

    if isinstance(color, str) and color in DEFAULT_COLORS:
        color = [DEFAULT_COLORS[color]] * len(mkpts0)
    elif not isinstance(color, (list, NDArray)):
        color = [color] * len(mkpts0)

    if isinstance(mkpts0, Tensor):
        mkpts0 = mkpts0.detach().cpu().numpy()
        mkpts1 = mkpts1.detach().cpu().numpy()

    for i, (mkp0, mkp1) in enumerate(zip(mkpts0, mkpts1)):
        alpha = kwargs.pop(
            'alpha', alphas[i] if alphas is not None else 1)
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


def plot_masks(
    mask0: Union[NDArray, Tensor],
    mask1: Union[NDArray, Tensor],
    color: str = 'r',
    alpha: float = 0.5,
    color_it: bool = True
) -> None:
    """
    Plots binary masks on the current Matplotlib axes for two images.

    This function takes two binary masks, typically representing segmented objects
    in two different images, and overlays them on the respective subplots of the
    current Matplotlib figure. The masks can be colored and made transparent.

    Parameters
    ----------
    mask0 : Union[NDArray, Tensor]
        Binary mask for the first image. Can be a NumPy array or a PyTorch Tensor.
        Expected shape (H, W) or (1, H, W).
    mask1 : Union[NDArray, Tensor]
        Binary mask for the second image. Can be a NumPy array or a PyTorch Tensor.
        Expected shape (H, W) or (1, H, W).
    color : str, optional
        The color to use for the masks. Can be a key from `DEFAULT_COLORS` ('r', 'g', 'b')
        or any valid Matplotlib color string. Defaults to 'r'.
    alpha : float, optional
        The transparency level for the masks, between 0 (fully transparent) and 1 (fully opaque).
        Defaults to 0.5.
    color_it : bool, optional
        If True, the mask is colored using the specified `color`. If False, the mask
        is plotted as is (assuming it's already a colored image or a grayscale mask
        that Matplotlib can color). Defaults to True.
    """
    # Get the current figure and its axes
    fig = plt.gcf()
    ax = fig.axes

    for i, mask in enumerate((mask0, mask1)):
        if isinstance(mask, Tensor):
            mask = mask.cpu().numpy()

        # Check if the mask contains any foreground pixels
        if not np.any(mask.astype(bool)):
            continue

        # If color_it is True, convert the binary mask to an RGB mask with the specified color
        if color_it:
            mask = mask.astype(bool)
            mask = np.where(mask[..., None], mcolors.to_rgb(
                DEFAULT_COLORS[color]), 1)

        # Overlay the mask on the corresponding subplot
        ax[i].imshow(mask, alpha=alpha)
