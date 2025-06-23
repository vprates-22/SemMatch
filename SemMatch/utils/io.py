import cv2
import torch
import torchvision
import numpy as np

from typing import Tuple, Union

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
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) / 1000
    image = torch.tensor(image)

    return image

def reshape(image:torch.Tensor, shape:Union[int, tuple, list]) -> Tuple[torch.Tensor, Union[float, tuple]]:
    h, w = image.shape[-2:]
    new_h, new_w = shape
    scale_h = new_h / h
    scale_w = new_w / w

    return torch.nn.functional.interpolate(image[None], size=shape, mode='bilinear', align_corners=False)[0], (scale_h, scale_w)

def resize_long_edge(image:torch.Tensor, max_size:int) -> Tuple[torch.Tensor, float]:
    h, w = image.shape[-2:]
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    
    scale = new_h / h
    
    return torch.nn.functional.interpolate(image[None], size=(new_h, new_w), mode='bilinear', align_corners=False)[0], scale

def combine_dicts(dict1:dict, dict2:dict) -> dict:
    """
    Combines two dictionaries by preserving key-value pairs with truthy values,
    ensuring that falsy values do not overwrite existing truthy ones.

    Parameters
    ----------
    dict1 : dict
        The first dictionary to combine.
    dict2 : dict
        The second dictionary to combine. If a key exists in both dictionaries,
        and the value in dict2 is truthy, it will take precedence.
        If the value in dict2 is falsy, the truthy value in dict1 will be retained.

    Returns
    -------
    dict
        A dictionary containing the union of keys from both inputs,
        where only truthy values are preserved. In case of key conflicts,
        the truthy value is retained, prioritizing dict2.
    """
    keys = set(dict1) | set(dict2)
    result = {}
    for key in keys:
        v1 = dict1.get(key)
        v2 = dict2.get(key)
        result[key] = v2 if v2 else v1
        if not result[key]:
            result[key] = v2 if key in dict2 else v1
    return {k: v for k, v in result.items()}