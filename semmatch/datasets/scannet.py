"""
Module: scannet
---------------

This module implements the `Scannet` class for loading and processing the ScanNet dataset.
It extends the `BaseDatasetLoader` class and provides functionality for reading images,
depth maps, and calibrated ground truth data (intrinsics and poses).

Classes:
    Scannet (BaseDatasetLoader):
        A dataset loader for the ScanNet dataset. Provides methods for loading and caching
        image and depth data, as well as parsing ground truth transformation and intrinsic matrices.
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

from semmatch.utils.image import load_image, load_depth
from semmatch.utils.geometry import get_inliers_ransac, project_points_between_cameras, estimate_pose


class Scannet(BaseDataset):
    """
    ScanNet dataset loader.

    This class handles loading and caching of ScanNet data, including RGB images,
    depth maps, camera intrinsics, and relative poses between image pairs.
    It supports downloading from Google Drive and parsing from a calibrated pairs file.

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
            'url': 'https://drive.google.com/uc?id=1wtl-mNicxGlXZ-UQJxFnKuWPvvssQBwd',
            'url_download_extension': '.tar',
            'pairs_path': DATA_PATH/'scannet1500_pairs_calibrated.txt',
        })

        base_config.merge_config(config)

        super().__init__(base_config)

        self._depth_cache = {}
        if self.config.cache_images:
            self.load_images('depth')

    def load_images(self, attr: str = 'image') -> None:
        """
        Loads and caches images or depth maps used in the data pairs.

        Parameters:
        ----------
        attr : str, optional
            The type of data to load: `'image'` for RGB images or `'depth'` for depth maps.
            Defaults to `'image'`. Any other value will raise an error.

        Raises:
        ------
        NotImplementedError
            If `attr` is not `'image'` or `'depth'`.
        """
        if attr not in ['image', 'depth']:
            raise NotImplementedError(
                'The only attributes allowed are "image" and "depth"')

        attr_name = f'_{attr}_cache'
        cache = getattr(self, attr_name)
        load_func = load_image if attr == 'image' else load_depth

        for pair in tqdm(self.pairs, desc=f'Caching {attr}s'):
            if pair[f'{attr}0'] not in cache:
                cache[pair[f'{attr}0']] = load_func(pair[f'{attr}0'])
            if pair[f'{attr}1'] not in cache:
                cache[pair[f'{attr}1']] = load_func(pair[f'{attr}1'])

    def read_image(self, path: str, attr: str = 'image') -> Tensor:
        """
        Reads an image or depth map from the given path, using cache if enabled.

        Parameters:
        ----------
        path : str
            The file path to the image or depth map.
        attr : str, optional
            The type of data to load: `'image'` for RGB images or `'depth'` for depth maps.
            Defaults to `'image'`.

        Returns:
        -------
        torch.Tensor
            The loaded image or depth map as a tensor.
        """
        if self.config.cache_images:
            return self.image_cache[path]

        load_func = load_image if attr == 'image' else load_depth
        return load_func(path)

    def read_gt(self) -> List[Dict[str, Any]]:
        """
        Reads ground truth data pairs from a text file, including image paths, depth paths,
        camera intrinsics, and relative poses.

        Returns:
        -------
        dict
            A list of dictionaries, each containing the following keys:

            - 'image0' : str
              Path to the first RGB image in the pair.
            - 'image1' : str
              Path to the second RGB image in the pair.
            - 'depth0' : str
              Path to the depth map corresponding to `image0`.
            - 'depth1' : str
              Path to the depth map corresponding to `image1`.
            - 'K0' : np.ndarray
              3x3 intrinsic matrix for the first camera.
            - 'K1' : np.ndarray
              3x3 intrinsic matrix for the second camera.
            - 'T_0to1' : np.ndarray
              4x4 transformation matrix representing the pose from image0 to image1.

        Notes:
        ------
        Only lines with exactly 32 elements are considered valid. If `max_pairs` is set
        in the config and greater than 0, the number of returned pairs is limited accordingly.
        """
        pairs = []
        with Path(self.config.pairs_path).open() as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').replace('  ', '')
                all_info = line.split(' ')
                image0 = all_info[0]
                image1 = all_info[1]

                if len(all_info) != 38:
                    continue

                K0, K1, T_0to1 = Scannet.build_intrinsic_matrixes(all_info)

                image0 = os.path.join(self.config.data_path, image0)
                image1 = os.path.join(self.config.data_path, image1)
                depth0 = image0.replace('color', 'depth').replace('jpg', 'png')
                depth1 = image1.replace('color', 'depth').replace('jpg', 'png')

                pairs.append({
                    'image0': image0,
                    'image1': image1,
                    'depth0': depth0,
                    'depth1': depth1,
                    'K0': K0,
                    'K1': K1,
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
        return project_points_between_cameras(
            points,
            self.read_image(self.pairs[pair_index]['depth0'], 'depth'),
            self.read_image(self.pairs[pair_index]['depth1'], 'depth'),
            self.pairs[pair_index]['K0'],
            self.pairs[pair_index]['K1'],
            self.pairs[pair_index]['T_0to1'],
            self.read_image(self.pairs[pair_index]['image0']).shape[-2:],
            self.read_image(self.pairs[pair_index]['image1']).shape[-2:],
            scale_img0,
            scale_img1,
            max_depth_diff=1
        )

    def get_inliers(
        self,
        mkpts0: NDArray,
        mkpts1: NDArray,
        pair_index: int,
        scale_img0: Iterable[Iterable[float]] = [1.0, 1.0],
        scale_img1: Iterable[Iterable[float]] = [1.0, 1.0],
        threshold: float = 6.0,
    ) -> NDArray: return get_inliers_ransac(
        mkpts0,
        mkpts1,
        self.pairs[pair_index]['K0'],
        self.pairs[pair_index]['K1'],
        threshold
    )

    def estimate_pose(
        self,
        pair_index: int,
        mkpts0: NDArray,
        mkpts1: NDArray,
        threshold: float = 6.0,
    ) -> Tuple[NDArray, NDArray]:
        return estimate_pose(
            mkpts0,
            mkpts1,
            self.pairs[pair_index]['K0'],
            self.pairs[pair_index]['K1'],
            threshold
        )

    @staticmethod
    def build_intrinsic_matrixes(all_info: List[str]) -> Tuple[NDArray, NDArray, NDArray]:
        K0 = np.array(all_info[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(all_info[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(all_info[22:38]).astype(float).reshape(4, 4)

        return K0, K1, T_0to1

    @property
    def depth_cache(self) -> Dict[str, Tensor]:
        return self._depth_cache
