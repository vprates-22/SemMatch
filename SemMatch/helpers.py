import cv2
import h5py
import torch
import torchvision
import numpy as np

import poselib
from .evaluation.utils import intrinsics_to_camera

def load_image(path, gray=False) -> torch.Tensor:
    '''
    Loads an image from a file path and returns it as a torch tensor
    Output shape: (3, H, W) float32 tensor with values in the range [0, 1]
    '''
    image = torchvision.io.read_image(str(path)).float() / 255
    if gray:
        image = torchvision.transforms.functional.rgb_to_grayscale(image)
    return image

def load_depth(path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    image = torch.tensor(image).unsqueeze(0)
    return image

def get_inliers(mkpts0:np.ndarray,
                mkpts1:np.ndarray,
                K0:np.ndarray,
                K1:np.ndarray ) -> np.ndarray:
    _, details = poselib.estimate_relative_pose(
        mkpts0.tolist(),
        mkpts1.tolist(),
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
        ransac_opt={
            'max_iterations': 10000,
            'success_prob': 0.99999,
            'max_epipolar_error': 6.0,
        }
    )

    return np.array(details['inliers']).astype(bool)

def to_cv(torch_image, convert_color=True, batch_idx=0, to_gray=False):
    '''Converts a torch tensor image to a numpy array'''
    if isinstance(torch_image, torch.Tensor):
        if len(torch_image.shape) == 2:
            torch_image = torch_image.unsqueeze(0)
        if len(torch_image.shape) == 4 and torch_image.shape[0] == 1:
            torch_image = torch_image[0]
        if len(torch_image.shape) == 4 and torch_image.shape[0] > 1:
            torch_image = torch_image[batch_idx]
        if len(torch_image.shape) == 3 and torch_image.shape[0] > 1:
            torch_image = torch_image[batch_idx].unsqueeze(0)
            
        if torch_image.max() > 1:
            torch_image = torch_image / torch_image.max()
        
        img = (torch_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    else:
        img = torch_image

    if convert_color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def to_tensor(img_np):
    if isinstance(img_np, torch.Tensor):
        return img_np

    img_np = img_np.astype(np.float32) / 255.0
    img_np = img_np * 2 - 1
    img_np = np.transpose(img_np, (2, 0, 1))
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
    return img_tensor