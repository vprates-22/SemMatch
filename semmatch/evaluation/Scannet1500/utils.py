"""
Module: depth_mapping_utils
----------------------------

This module provides utility functions for handling depth image data and 
mapping between RGB (color) image coordinates and depth image coordinates.

Constants:
----------
- DEPTH_IMG_WIDTH, DEPTH_IMG_HEIGHT: Expected resolution of the depth images.
- COLOR_IMG_WIDTH, COLOR_IMG_HEIGHT: Expected resolution of the color images.

Functions:
----------
- get_depth_in_meters: Reads a 16-bit PNG depth image and converts it to meters.
- map_color_to_depth_coordinates: Maps (x, y) pixel coordinates from a color image to corresponding depth image coordinates.

Notes:
------
This module assumes a specific resolution and alignment between color and depth 
images. The mapping is a simple linear scale based on known image dimensions.
"""

import cv2
import numpy as np

DEPTH_IMG_HEIGHT = 480
DEPTH_IMG_WIDTH = 640
COLOR_IMG_HEIGHT = 968
COLOR_IMG_WIDTH = 1296


def get_depth_in_meters(image_path: str) -> np.ndarray:
    """

    """
    depth_path = image_path.replace('color', 'depth').replace('.jpg', '.png')

    image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    return image / 1000


def map_color_to_depth_coordinates(x, y) -> tuple[int, int]:
    x_depth = int(DEPTH_IMG_WIDTH * x / COLOR_IMG_WIDTH)
    y_depth = int(DEPTH_IMG_HEIGHT * y / COLOR_IMG_HEIGHT)
    return x_depth, y_depth
