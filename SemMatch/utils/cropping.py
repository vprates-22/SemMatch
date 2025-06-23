import cv2
import numpy as np

def crop_square_around_mask(image, mask, output_size=(256, 256)):
    # Ensure mask is boolean
    mask = np.asarray(mask).astype(bool)

    # Get the coordinates of the mask
    y_indices, x_indices = np.where(mask)

    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Mask is empty. No object found.")

    # Bounding box
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Center of the object
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    # Determine square size (max of width/height)
    size = max(x_max - x_min, y_max - y_min)
    half = size // 2 + 1  # +1 to include full object

    # Initial crop coordinates
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    # Pad if crop is out of image bounds
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - image.shape[1])
    pad_bottom = max(0, y2 - image.shape[0])

    # Pad image if needed
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        image = np.pad(image, 
                       ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)) if image.ndim == 3 else 
                       ((pad_top, pad_bottom), (pad_left, pad_right)), 
                       mode='constant', constant_values=255)

    # Adjust crop after padding
    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    # Final square crop
    cropped = image[y1:y2, x1:x2]

    # Resize to fixed shape
    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_LINEAR)

    return resized