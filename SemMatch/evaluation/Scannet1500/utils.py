import cv2
import numpy as np

DEPTH_IMG_HEIGHT = 480
DEPTH_IMG_WIDTH = 640
COLOR_IMG_HEIGHT = 968
COLOR_IMG_WIDTH = 1296

def get_depth_in_meters(image_path:str) -> np.ndarray:
    """
    
    """
    depth_path = image_path.replace('color', 'depth').replace('.jpg', '.png')

    image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    return image / 1000

def map_color_to_depth_point(x, y) -> tuple[int, int]:
    x_depth = int(DEPTH_IMG_WIDTH * x / COLOR_IMG_WIDTH)
    y_depth = int(DEPTH_IMG_HEIGHT * y / COLOR_IMG_HEIGHT)
    return x_depth, y_depth