"""
Module: megadepth
------------------

This module implements the `MegaDepth` class for loading and processing MegaDepth dataset.
It extends the `BaseDatasetLoader` class and provides methods for reading image data, depth data,
and ground truth information.

Classes:
    MegaDepth (BaseDatasetLoader):
        A class that loads the MegaDepth dataset. It provides functionality to
        load and cache images and depth data, as well as read ground truth data
        including camera intrinsics and extrinsics.
"""

import os
from typing import List, Dict, Tuple, Any

import numpy as np
from tqdm import tqdm

from torch import Tensor
from numpy.typing import NDArray

from semmatch.settings import DATA_PATH
from semmatch.loaders.base_dataset_loader import BaseDatasetLoader
from semmatch.utils.io import load_image, load_depth, combine_dicts


class MegaDepth(BaseDatasetLoader):
    """
    MegaDepth dataset loader.

    This class handles loading and caching of MegaDepth data, including RGB images,
    depth maps, camera intrinsics, and relative poses between image pairs. It supports
    optional downloading of the dataset and loading of ground truth from a pre-defined
    pairs file.

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

    def __init__(self, config: Dict[str, Any] = None):
        base_config = {
            'url': 'https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip',
            'url_download_extension': '.zip',
            'pairs_path': DATA_PATH/'megadepth1500_pairs_calibrated.txt',
        }
        config = combine_dicts(base_config, config or {})

        super().__init__(config)

        self._depth_cache = {}
        if self.config['cache_images']:
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
        if self.config['cache_images']:
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
        with self.config['pairs_path'].open() as f:
            for line in f.readlines():
                line = line.replace('\n', '').replace('  ', '')
                all_info = line.split(' ')
                image0 = all_info[0]
                image1 = all_info[1]

                if len(all_info) != 32:
                    continue

                K0, K1, T_0to1 = MegaDepth.build_intrinsic_matrixes(all_info)

                depth0 = os.path.join(
                    self.config['data_path'], 'depths', image0).replace('.jpg', '.h5')
                depth1 = os.path.join(
                    self.config['data_path'], 'depths', image1).replace('.jpg', '.h5')
                image0 = os.path.join(
                    self.config['data_path'], 'images', image0)
                image1 = os.path.join(
                    self.config['data_path'], 'images', image1)

                pairs.append({
                    'image0': image0,
                    'image1': image1,
                    'depth0': depth0,
                    'depth1': depth1,
                    'K0': K0,
                    'K1': K1,
                    'T_0to1': T_0to1,
                })

            if self.config['max_pairs'] > 0:
                pairs = pairs[:self.config['max_pairs']]
        return pairs

    def map_point(self, point, pair_index, scale_img0 = 1, scale_img1 = 1):
        pass

    @staticmethod
    def build_intrinsic_matrixes(all_info: List[str]) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Builds the intrinsic matrices and relative transformation from raw text data.

        Parameters:
        ----------
        all_info : List[str]
            A list of 32 strings representing the pair info, intrinsics, and pose.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Intrinsics K0, K1 (3x3) and relative transformation T_0to1 (4x4).
        """
        K0 = np.array(all_info[2:11]).astype(float).reshape(3, 3)
        K1 = np.array(all_info[11:20]).astype(float).reshape(3, 3)

        pose_elems = all_info[20:32]
        R, t = pose_elems[:9], pose_elems[9:12]
        R = np.array([float(x) for x in R]).reshape(
            3, 3).astype(np.float32)
        t = np.array([float(x) for x in t]).astype(np.float32)

        T_0to1 = np.eye(4)
        T_0to1[:3, :3] = R
        T_0to1[:3, 3] = t

        return K0, K1, T_0to1

    @property
    def depth_cache(self) -> Dict[str, Tensor]:
        return self._depth_cache
