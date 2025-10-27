"""
Module: mask_visualization
---------------------------

This module provides utilities for visualizing binary object masks overlaid 
on images, particularly for semantic segmentation, matching, and evaluation tasks.

It assumes integration with existing Matplotlib figures, where two images 
are displayed side-by-side (e.g., using `plot_pair()`), and allows overlaying 
colored masks to highlight regions of interest.

Functions
---------
plot_masks(mask0, mask1, color='r', alpha=0.5, color_it=True)
    Overlays binary masks on a side-by-side image visualization using a 
    semi-transparent colored layer.
    
Notes
-----
- This module expects that a Matplotlib figure with two subplots is already 
  active via `plt.gcf()`.
- Mask inputs can be either NumPy arrays or PyTorch tensors.
- Mask colors are selected from a predefined color palette `DEFAULT_COLORS`.
"""

from typing import Union

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..utils.visualization import DEFAULT_COLORS


def plot_masks(mask0: Union[np.ndarray, torch.Tensor],
               mask1: Union[np.ndarray, torch.Tensor],
               color: str = 'r',
               alpha: float = 0.5,
               color_it: bool = True) -> None:
    """
    Overlay binary masks on top of two side-by-side images in a matplotlib figure.

    This function assumes that a figure with two subplots (side-by-side) already exists 
    and is active via `plt.gcf()`. It will color the given binary masks and overlay them 
    on each image using a specified transparency.

    Parameters
    ----------
    mask0 : np.ndarray or torch.Tensor
        Binary mask for the first (left) image. Shape: (H, W).
    mask1 : np.ndarray or torch.Tensor
        Binary mask for the second (right) image. Shape: (H, W).
    color : str, optional
        Color key (e.g., 'r', 'g', 'b') from DEFAULT_COLORS to apply to the mask overlay.
        Default is 'r' (red).
    alpha : float, optional
        Opacity of the mask overlay in the plot. Value should be between 0 (transparent)
        and 1 (opaque). Default is 0.5.
    color_it : bool, optional
        Whether to colorize the mask using the provided color. If False, overlays the 
        binary mask in grayscale. Default is True.

    Returns
    -------
    None
    """
    fig = plt.gcf()
    ax = fig.axes

    for i, mask in enumerate((mask0, mask1)):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        if not np.any(mask.astype(bool)):
            continue

        if color_it:
            mask = mask.astype(bool)
            mask = np.where(mask[..., None], mcolors.to_rgb(
                DEFAULT_COLORS[color]), 1)

        ax[i].imshow(mask, alpha=alpha)
