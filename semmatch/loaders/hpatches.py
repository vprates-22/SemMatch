import os
from typing import List, Dict, Tuple, Any

import numpy as np
from tqdm import tqdm

from torch import Tensor
from numpy.typing import NDArray

from semmatch.settings import DATA_PATH
from semmatch.loaders.base_dataset_loader import BaseDatasetLoader
from semmatch.utils.io import load_image, combine_dicts
from semmatch.utils.evaluation import apply_homography_to_point, rescale_homography


class HPatches(BaseDatasetLoader):
    def __init__(self, config: Dict[str, Any] = None):
        base_config = {
            'url': 'https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip',
            'url_download_extension': '.zip',
            'pairs_path': DATA_PATH/'hpatches_pairs_calibrated.txt',
        }
        config = combine_dicts(base_config, config or {})

        super().__init__(config)

    def load_images(self, attr='image') -> None:
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
        if attr != 'image':
            raise NotImplementedError(
                'The only attribute allowed are "image"')

        for pair in tqdm(self.pairs, desc='Caching images'):
            if pair['image0'] not in self._image_cache:
                self._image_cache[pair['image0']] = load_image(pair['image0'])
            if pair['image1'] not in self._image_cache:
                self._image_cache[pair['image1']] = load_image(pair['image1'])

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
        if attr != 'image':
            raise NotImplementedError(
                'The only attribute allowed are "image"')

        if self.config['cache_images']:
            return self.image_cache[path]

        return load_image(path)

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
                line = line.strip()
                all_info = line.split(' ')
                image0, image1 = all_info[:2]

                if len(all_info) != 11:
                    continue

                T_0to1 = HPatches.build_intrinsic_matrixes(all_info)

                image0 = os.path.join(
                    self.config['data_path'], image0)
                image1 = os.path.join(
                    self.config['data_path'], image1)

                pairs.append({
                    'image0': image0,
                    'image1': image1,
                    'T_0to1': T_0to1,
                })

            if self.config['max_pairs'] > 0:
                pairs = pairs[:self.config['max_pairs']]
        return pairs

    def map_point(
        self,
        point: Tuple[float, float],
        pair_index: int,
        scale_img0: float = 1.0,
        scale_img1: float = 1.0
    ) -> tuple[Tuple[float, float], bool]:
        """
        Maps a single 2D point from image 0 to image 1 using the ground-truth homography
        from the HPatches dataset and verifies whether the mapped point lies within the
        bounds of image 1.

        Parameters
        ----------
        point : Tuple[float, float]
            (x, y) coordinates in image 0.
        pair_index : int
            Index of the HPatches image pair to use.
        scale_img0 : float, optional
            Scaling factor applied to image 0 before mapping (default is 1.0).
        scale_img1 : float, optional
            Scaling factor applied to image 1 before mapping (default is 1.0).

        Returns
        -------
        mapped_point : Tuple[float, float]
            The (x, y) coordinates of the mapped point in image 1 space. If invalid, returns (np.nan, np.nan).
        valid : bool
            True if the mapped point lies inside image 1 bounds, False otherwise.

        Notes
        -----
        - This function uses the homography `T_0to1` provided by HPatches.
        - If scaling is applied, the homography is rescaled accordingly using `rescale_homography`.
        - Image 1 bounds are inferred from the stored pair information.
        """
        x, y = point
        pair_info = self._pairs[pair_index]

        # --- Retrieve homography and image shape ---
        T_0to1 = pair_info["T_0to1"]
        img1_shape = self.read_image(pair_info["image1"]).shape[-2:]  # (H, W)

        # --- Apply scaling if needed ---
        if scale_img0 != 1.0 or scale_img1 != 1.0:
            T_0to1 = rescale_homography(T_0to1, scale_img0, scale_img1)
            img1_shape = (int(img1_shape[0] * scale_img1), int(img1_shape[1] * scale_img1))

        # --- Apply homography ---
        x1, y1 = apply_homography_to_point(x, y, T_0to1)

        # --- Check image bounds ---
        H, W = img1_shape
        inside = (0 <= x1 < W) and (0 <= y1 < H)

        # --- Return result ---
        if not inside:
            return (np.nan, np.nan), False

        return (float(x1), float(y1)), True


    def get_inliers(
        self,
        mkpts0: np.ndarray,
        mkpts1: np.ndarray,
        pair_index: int,
        threshold: float = 6.0,
        scale_img0: float = 1.0,
        scale_img1: float = 1.0,
    ) -> np.ndarray:
        """
        Estimates inliers for an HPatches pair using the ground-truth homography.

        Args:
            mkpts0: Nx2 array of points from image 0.
            mkpts1: Nx2 array of corresponding points from image 1.
            pair_index: index of the HPatches pair to use.
            hpatches_dataset: object providing `map_point(point, pair_index, ...)`.
            threshold: reprojection error threshold in pixels to consider inliers.
            scale_img0: optional rescaling factor for image 0.
            scale_img1: optional rescaling factor for image 1.

        Returns:
            inliers: boolean mask of shape (N,), True for inliers.
        """
        assert mkpts0.shape == mkpts1.shape
        N = mkpts0.shape[0]

        mapped_points = np.zeros_like(mkpts0, dtype=float)
        for i, pt0 in enumerate(mkpts0):
            mapped_points[i] = self.map_point(
                tuple(pt0),
                pair_index,
                scale_img0=scale_img0,
                scale_img1=scale_img1
            )

        # Compute Euclidean reprojection error
        errors = np.linalg.norm(mapped_points - mkpts1, axis=1)
        inliers = errors < threshold
        return inliers

    @staticmethod
    def build_intrinsic_matrixes(all_info: List[str]) -> NDArray:
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
        T_0to1 = np.array(all_info[2:]).astype(np.float32)
        T_0to1 = T_0to1.reshape(3, 3)

        return T_0to1
