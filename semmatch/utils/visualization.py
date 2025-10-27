"""
Module: utils.visualization
----------------------------

This module provides basic visualization utilities using Matplotlib for 
displaying and saving images and image pairs. It is particularly useful 
for visual debugging of computer vision tasks.

Functions:
----------
- plot_pair: Display two images side by side in a single figure.
- show: Render the current Matplotlib figure.
- save: Save the current figure to disk with optional parameters.

Constants:
----------
- DEFAULT_COLORS: Dictionary of default colors used for annotations (e.g., green, red, blue).

Dependencies:
-------------
- matplotlib
- numpy
- to_cv (from internal helpers): Ensures image is in a suitable format for Matplotlib display.

Notes:
------
Images are assumed to be in formats compatible with `to_cv()`, which likely 
handles channel ordering and normalization.
"""

from typing import Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

from ..helpers import to_cv

DEFAULT_COLORS = {
    'g': '#4ade80',
    'r': '#ef4444',
    'b': '#3b82f6',
}


def plot_pair(img0: np.ndarray,
              img1: np.ndarray,
              figsize: Tuple[int, int] = (20, 10),
              title: Optional[str] = None) -> None:
    """
    Plot two images side by side in a single matplotlib figure.

    Parameters
    ----------
    img0 : np.ndarray
        Left image to be plotted (e.g., RGB or grayscale).
    img1 : np.ndarray
        Right image to be plotted.
    figsize : tuple of int, optional
        Size of the matplotlib figure in inches. Default is (20, 10).
    title : str, optional
        Optional title for the figure.

    Returns
    -------
    tuple
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax : list of matplotlib.axes.Axes
            List containing the axes for the two subplots.
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
    Display the current matplotlib figure.

    This function disables axis display and adjusts layout before showing the figure.
    """
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def save(path: str, **kwargs: Any) -> None:
    """
    Save the current matplotlib figure to a file.

    This function saves the current figure using `matplotlib.pyplot.savefig`, 
    applying tight layout for clean output.

    Parameters
    ----------
    path : str
        File path where the figure should be saved.
    **kwargs : dict
        Additional keyword arguments passed to `plt.savefig`, such as `dpi`, 
        `bbox_inches`, or file format options.
    """
    plt.tight_layout()
    plt.savefig(path, **kwargs)
    plt.close()
