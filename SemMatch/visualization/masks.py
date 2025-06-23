import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from typing import Union
from .utils import DEFAULT_COLORS

def plot_masks(mask0:Union[np.ndarray, torch.Tensor], 
               mask1:Union[np.ndarray, torch.Tensor], 
               color:str='r', 
               alpha:float=0.5,
               color_it:bool=True) -> None:
    """
    Plot two images side by side with their respective binary masks highlighted.

    The masks are overlaid using a semi-transparent color, preserving the original images.

    Parameters
    ----------
    mask0 : np.ndarray
        Binary mask for the first image, shape (H, W).
    mask1 : np.ndarray
        Binary mask for the second image, shape (H, W).
    color : str, optional
        Color name or list of colors for the match lines. Default is 'r' (blue).
    alpha : float, optional
        Opacity of the overlay. Default is 0.5.
    color_it : bool, optional
        Wether the mask needs to be colored. Default is True.

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
            mask = np.where(mask[..., None], mcolors.to_rgb(DEFAULT_COLORS[color]), 1)
        
        ax[i].imshow(mask, alpha=alpha)