"""
Module: utils.models
--------------------

This module provides utility functions for loading and using pre-trained models,
such as the Segment Anything Model (SAM) for object segmentation and the LPIPS
model for perceptual image similarity.

Functions
---------
load_sam(sam_model, device)
    Loads the Segment Anything Model (SAM) from a specified file.
get_object_mask(sam, image, points, batch_size)
    Predicts binary segmentation masks for a set of points using SAM.
load_lpips(lpips_net, device)
    Initializes the LPIPS model for perceptual similarity evaluation.
get_obj_similarities(lpips, img0, img1, mask0, mask1, device)
    Computes LPIPS similarity between two image regions defined by masks.
"""

from typing import List
import torch
import lpips
import numpy as np

from ultralytics import SAM
from numpy.typing import NDArray
from semmatch.settings import MODEL_DIR_NAME

from semmatch.utils.image import to_tensor
from semmatch.utils.image import crop_square_around_mask

# -------------------------------SAM------------------------------------------------


def load_sam(
        sam_model: str = 'sam2.1_l.pt',
        device: torch.device = None
) -> SAM:
    """
    Loads the Segment Anything Model (SAM) from a specified model file.

    The SAM model is loaded from a local directory defined by the BASE_PATH and MODEL_DIR_NAME.
    The model is then moved to the configured device (GPU or CPU) and set to evaluation mode.

    Parameters
    ----------
    sam_model : str, optional
        The filename of the SAM model to load. Defaults to 'sam2.1_l.pt'.
    device : torch.device, optional
        The device to load the model onto (e.g., 'cuda' or 'cpu'). If None, it defaults to 'cuda' if available, else 'cpu'.
    Returns
    -------
    SAM
        An instance of the SAM model ready for inference.
    """
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_DIR_NAME.mkdir(parents=True, exist_ok=True)

    file_path = MODEL_DIR_NAME / sam_model

    sam = SAM(file_path)
    sam.to(device).eval()

    return sam


def get_object_mask(
    sam: SAM,
    image: NDArray,
    points: List[List[int]],
    batch_size: int = 200
) -> NDArray:
    """
    Predicts binary segmentation masks from a batch of keypoints using the SAM model.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image as a NumPy array (H, W, 3).
    points : list of list of int
        List of 2D points (e.g., [[x, y]]) to prompt SAM.
    batch_size : int, optional
        Number of points per batch. If -1, processes all at once. Default is 200.

    Returns
    -------
    np.ndarray
        Boolean array of shape (N, H, W) with one binary mask per point.
    """
    masks = []

    if batch_size == -1:  # If batch_size is -1, process all points at once
        batch_size = len(points)

    # Divide the points into batches
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]

        # Limpar a memória da GPU antes de cada previsão em batch
        # Clear GPU memory before each batch prediction
        torch.cuda.empty_cache()

        with torch.no_grad():
            # Perform prediction for the batch of points in the same image
            results = sam.predict(image, points=batch_points, verbose=False)
            # Add the masks for each batch of points
            masks.extend(results[0].masks.data.cpu().numpy())

    return np.array(masks, dtype=bool)

# ----------------------------------------------------------------------------------

# -------------------------------LPIPS------------------------------------------------


def load_lpips(
    lpips_net: str,
    device: torch.device = None
) -> lpips.LPIPS:
    """
    Initializes the LPIPS (Learned Perceptual Image Patch Similarity) model.

    Uses the network architecture specified in the configuration (e.g., 'alex', 'vgg').
    The model is moved to the configured device and set up for similarity evaluation.

    Parameters
    ----------
    lpips_net : str
        The name of the LPIPS network architecture to use (e.g., 'alex', 'vgg').
    device : torch.device, optional
        The device to load the model onto (e.g., 'cuda' or 'cpu'). If None, it defaults to 'cuda' if available, else 'cpu'.
    Returns
    -------
    lpips.LPIPS
        An instance of the LPIPS model ready for inference.
    """
    fn = lpips.LPIPS(net=lpips_net, verbose=False)
    return fn.to(device)


def get_obj_similarities(
        lpips,
        img0,
        img1,
        mask0,
        mask1,
        device: torch.device = None
) -> float:
    """
    Computes the LPIPS perceptual similarity between two cropped image regions defined by masks.

    Crops square regions around each mask on the corresponding image, converts them to tensors,
    and computes the LPIPS distance indicating perceptual similarity.

    Parameters
    ----------
    img0 : np.ndarray
        The first image array.
    img1 : np.ndarray
        The second image array.
    mask0 : np.ndarray
        Binary mask defining the object region in the first image.
    mask1 : np.ndarray
        Binary mask defining the object region in the second image.

    Returns
    -------
    float
        The LPIPS perceptual similarity score between the two cropped object regions.
    """
    with torch.no_grad():
        cropped_img0 = to_tensor(crop_square_around_mask(img0, mask0))
        cropped_img1 = to_tensor(crop_square_around_mask(img1, mask1))

        return lpips(
            cropped_img0.to(device),
            cropped_img1.to(device)
        ).item()

# ----------------------------------------------------------------------------------
