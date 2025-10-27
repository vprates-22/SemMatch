from typing import List

import torch
import numpy as np
from numpy.typing import NDArray

from ultralytics import SAM
from ..settings import MODEL_DIR_NAME


def load_sam(
        sam_model: str = 'sam2.1_l.pt',
        device: torch.device = None
) -> SAM:
    """
    Loads the Segment Anything Model (SAM) from a specified model file.

    The SAM model is loaded from a local directory defined by the BASE_PATH and MODEL_DIR_NAME.
    The model is then moved to the configured device (GPU or CPU) and set to evaluation mode.

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

    if batch_size == -1:
        # Se batch_size for -1, processa todos os pontos de uma vez
        batch_size = len(points)

    # Dividir os pontos em lotes (batches)
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]

        # Limpar a memória da GPU antes de cada previsão em batch
        torch.cuda.empty_cache()

        with torch.no_grad():
            # Realiza a previsão para o batch de pontos na mesma imagem
            results = sam.predict(image, points=batch_points, verbose=False)

            # Adicionar as máscaras para cada batch de pontos
            masks.extend(results[0].masks.data.cpu().numpy())

    return np.array(masks, dtype=bool)
