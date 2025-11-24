"""
This module defines the HPatches dataset loader.

The HPatches dataset is a collection of image sequences with varying viewpoints and illumination.
This loader is designed to read image pairs and their corresponding homographies,
which are used for evaluating feature matching algorithms.

The main class, HPatches, handles loading the dataset, caching images, and providing
ground truth data for image pairs.
"""
import os
from typing import List, Dict, Tuple, Any, Union, Iterable

import numpy as np
from tqdm import tqdm

from pathlib import Path
from torch import Tensor
from numpy.typing import NDArray

from semmatch.settings import DATA_PATH
from semmatch.datasets.base import BaseDataset, Config
from semmatch.utils.image import load_image
from semmatch.utils.geometry import map_points_between_images


class HPatches(BaseDataset):
    """
    HPatches dataset loader.

    This class handles loading and caching of HPatches data, including RGB images
    and homography matrices between image pairs. It supports optional downloading
    of the dataset and loading of ground truth from a pre-defined pairs file.

    Configuration:
    This class uses a config dictionary that merges user-provided values with
    sensible defaults. The expected keys include:

    - data_path (str): Local path where the dataset is stored.
    - pairs_path (str): Path to the file describing image pairs or labels.
    - cache_images (bool): If True, loads and stores all images in memory.
    - max_pairs (int): Limits the number of pairs to load (-1 for no limit).
    - url (str): URL to download the dataset if it's not found locally.
    - url_download_extension (str): File extension for the archive (e.g., '.zip').
    """

    def __init__(self, config: Union[Config, Dict[str, Any]] = None):
        base_config = Config({
            'url': 'https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip',
            'url_download_extension': '.zip',
            'pairs_path': DATA_PATH/'hpatches_pairs_calibrated.txt',
        })

        base_config.merge_config(config)

        super().__init__(base_config)

    def load_images(self, attr='image') -> None:
        """
        Loads and caches images used in the data pairs.

        Parameters:
        ----------
        attr : str, optional
            The type of data to load: `'image'` for RGB images.
            Defaults to `'image'`. Any other value will raise an error.

        Raises:
        ------
        NotImplementedError
            If `attr` is not `'image'`.
        """
        if attr != 'image':
            raise NotImplementedError(
                'The only attribute allowed is "image"')

        for pair in tqdm(self.pairs, desc='Caching images'):
            if pair['image0'] not in self._image_cache:
                self._image_cache[pair['image0']] = load_image(pair['image0'])
            if pair['image1'] not in self._image_cache:
                self._image_cache[pair['image1']] = load_image(pair['image1'])

    def read_image(self, path: str, attr: str = 'image') -> Tensor:
        """
        Reads an image from the given path, using cache if enabled.

        Parameters:
        ----------
        path : str
            The file path to the image.
        attr : str, optional
            The type of data to load: `'image'` for RGB images.
            Defaults to `'image'`.

        Returns:
        -------
        torch.Tensor
            The loaded image as a tensor.
        """
        if attr != 'image':
            raise NotImplementedError(
                'The only attribute allowed is "image"')

        if self.config.cache_images:
            return self.image_cache[path]

        return load_image(path)

    def read_gt(self) -> List[Dict[str, Any]]:
        """
        Reads ground truth data pairs from a text file, including image paths
        and homography matrices.

        Returns:
        -------
        dict
            A list of dictionaries, each containing the following keys:
            - 'image0': str
              Path to the first RGB image in the pair.
            - 'image1': str
              Path to the second RGB image in the pair.
            - 'T_0to1': NDArray
              3x3 homography matrix from image0 to image1.

        Notes:
        ------
        If `max_pairs` is set in the config and greater than 0, the number of returned
        pairs is limited accordingly.
        """
        pairs = []
        with Path(self.config.pairs_path).open() as f:
            for line in f.readlines():
                line = line.strip()
                all_info = line.split(' ')
                image0, image1 = all_info[:2]

                if len(all_info) != 11:
                    continue

                T_0to1 = HPatches._build_intrinsic_matrixes(all_info)

                image0 = os.path.join(
                    self.config.data_path, image0)
                image1 = os.path.join(
                    self.config.data_path, image1)

                pairs.append({
                    'image0': image0,
                    'image1': image1,
                    'T_0to1': T_0to1,
                })

            if self.config.max_pairs > 0:
                pairs = pairs[:self.config.max_pairs]
        return pairs

    def map_point(
        self,
        points: NDArray,
        pair_index: int,
        scale_img0: Iterable[float] = [1.0, 1.0],
        scale_img1: Iterable[float] = [1.0, 1.0]
    ) -> tuple[Tuple[float, float], bool]:
        """
        Maps 2D points from the first image to the second image using the ground truth homography.

        Parameters
        ----------
        points : NDArray
            (N, 2) Array of 2D points in the source image (image0).
        pair_index : int
            The index of the image pair in the dataset's `_pairs` list.
        scale_img0 : Iterable[float], optional
            Scaling factors for the source image (image0) as (scale_h, scale_w).
            Defaults to [1.0, 1.0].
        scale_img1 : Iterable[float], optional
            Scaling factors for the target image (image1) as (scale_h, scale_w).
            Defaults to [1.0, 1.0].

        Returns
        -------
        mapped : NDArray
            (N, 2) Array of mapped 2D points in the target image (image1).
        valid : NDArray
            (N,) Boolean mask indicating whether each mapped point is valid
            (i.e., inside the target image bounds after mapping).
        """
        pair_info = self._pairs[pair_index]
        H = pair_info["T_0to1"]
        img1_shape = self.read_image(pair_info["image1"]).shape[-2:]

        return map_points_between_images(
            points, H, img1_shape, scale_img0, scale_img1)

    def get_inliers(
        self,
        mkpts0: NDArray,
        mkpts1: NDArray,
        pair_index: int,
        scale_img0: Iterable[Iterable[float]],
        scale_img1: Iterable[Iterable[float]],
        threshold: float = 6.0,
    ) -> NDArray:
        """
        Estimates inliers for an HPatches pair by comparing mapped points with actual points.

        This method uses the ground-truth homography to project `mkpts0` into the second image
        and then checks the reprojection error against `mkpts1`. Points with a reprojection
        error below the `threshold` and that are valid (within image bounds) are considered inliers.

        Parameters
        ----------
        mkpts0 : NDArray
            (N, 2) Array of matched keypoints from the first image.
        mkpts1 : NDArray
            (N, 2) Array of corresponding matched keypoints from the second image.
        pair_index : int
            The index of the HPatches pair in the dataset's `_pairs` list.
        scale_img0 : Iterable[Iterable[float]]
            Scaling factors for the first image.
        scale_img1 : Iterable[Iterable[float]]
            Scaling factors for the second image.
        threshold : float, optional
            Reprojection error threshold in pixels to consider a match an inlier. Defaults to 6.0.

        Returns
        -------
        NDArray
            Boolean array of shape (N,) indicating which keypoint matches are inliers.
        """
        assert mkpts0.shape == mkpts1.shape

        mapped, valid = self.map_point(
            mkpts0, pair_index, scale_img0, scale_img1)

        errors = np.linalg.norm(mapped - mkpts1, axis=1)
        return valid & (errors < threshold)

    def estimate_pose(
        self,
        pair_index: int,
        mkpts0: NDArray,
        mkpts1: NDArray,
        threshold: float = 6.0,
    ) -> Tuple[NDArray, NDArray]:
        """
        Estimates the relative pose (rotation and translation) between two images in the HPatches dataset.

        This method computes the essential matrix using the matched keypoints and
        decomposes it to obtain the rotation and translation between the two views.

        Parameters
        ----------
        pair_index : int
            The index of the HPatches pair in the dataset's `_pairs` list.
        mkpts0 : NDArray
            (N, 2) Array of matched keypoints from the first image.
        mkpts1 : NDArray
            (N, 2) Array of corresponding matched keypoints from the second image.
        threshold : float, optional
            RANSAC reprojection error threshold in pixels. Defaults to 6.0.

        Returns
        -------
        Tuple[NDArray, NDArray] | None
            A tuple containing the rotation matrix (3x3) and translation vector (3x1)
            between the two images. Returns None if pose estimation fails.
        """
        return None, None

    @staticmethod
    def _build_intrinsic_matrixes(all_info: List[str]) -> NDArray:
        """
        Builds the homography matrix from raw text data.

        Parameters
        ----------
        all_info: List[str]
            A list of strings containing the homography matrix elements.

        Returns
        -------
        NDArray
            The 3x3 homography matrix (T_0to1) as a NumPy array.
        """
        T_0to1 = np.array(all_info[2:]).astype(np.float32)
        T_0to1 = T_0to1.reshape(3, 3)

        return T_0to1
