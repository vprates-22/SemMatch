import torch
from numpy.typing import NDArray


class UpdateData:
    pass


class BaseMetricUpdateData(UpdateData):
    image0: str
    image1: str
    mkpts0: torch.Tensor
    mkpts1: torch.Tensor
    inliers: NDArray
    mask_hits: NDArray
    lpips_loss: list
    valid_projections: NDArray
