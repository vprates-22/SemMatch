import torch
import lpips

from semmatch.helpers import to_tensor
from semmatch.utils.cropping import crop_square_around_mask


def load_lpips(
        lpips_net:str, 
        device: torch.device = None
    ) -> lpips.LPIPS:
    """
    Initializes the LPIPS (Learned Perceptual Image Patch Similarity) model.

    Uses the network architecture specified in the configuration (e.g., 'alex', 'vgg').
    The model is moved to the configured device and set up for similarity evaluation.

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
