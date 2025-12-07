"""
Module: semmatch.datasets.hpatches
------------------

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
from semmatch.utils.image import load_image
from semmatch.datasets.base import BaseDataset, Config
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
        if attr != 'image':
            raise NotImplementedError(
                'The only attribute allowed is "image"')

        for pair in tqdm(self.pairs, desc='Caching images'):
            if pair['image0'] not in self._image_cache:
                self._image_cache[pair['image0']] = load_image(pair['image0'])
            if pair['image1'] not in self._image_cache:
                self._image_cache[pair['image1']] = load_image(pair['image1'])

    def read_image(self, path: str, attr: str = 'image') -> Tensor:
        if attr != 'image':
            raise NotImplementedError(
                'The only attribute allowed is "image"')

        if self.config.cache_images:
            return self.image_cache[path]

        return load_image(path)

    def read_gt(self) -> List[Dict[str, Any]]:
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
        scale_img0: Iterable[Iterable[float]] = [1.0, 1.0],
        scale_img1: Iterable[Iterable[float]] = [1.0, 1.0],
        threshold: float = 6.0,
    ) -> NDArray:
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
    ) -> Tuple[None, None]:
        return None, None

    @staticmethod
    def _build_intrinsic_matrixes(all_info: List[str]) -> NDArray:
        """
        Builds the homography matrix from raw text data.

        Parameters
        ----------
        all_info: List[str]
            A list of strings containing the homography matrix elements (9 elements).

        Returns
        -------
        NDArray
            The 3x3 homography matrix (T_0to1) as a NumPy array.
        """
        T_0to1 = np.array(all_info[2:]).astype(np.float32)
        T_0to1 = T_0to1.reshape(3, 3)

        return T_0to1
