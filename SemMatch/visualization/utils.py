import numpy as np
import matplotlib.pyplot as plt

from ..helpers import to_cv
from typing import Tuple, Optional, Any

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
        Left image to be plotted.
    img1 : np.ndarray
        Right image to be plotted.
    figsize : tuple, optional
        Size of the matplotlib figure. Default is (20, 10).
    title : str, optional
        Title for the figure.

    Returns
    -------
    tuple
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax : list of matplotlib.axes.Axes
            List of axes for the left and right images.
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

    The axis is turned off and layout is tightened before displaying.
    """
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def save(path: str, **kwargs: Any) -> None:
    """
    Save the current matplotlib figure to a file.

    Parameters
    ----------
    path : str
        Path where the figure should be saved.
    **kwargs : dict
        Additional keyword arguments passed to `matplotlib.pyplot.savefig`.
    """
    plt.tight_layout()
    plt.savefig(path, **kwargs)
    plt.close()