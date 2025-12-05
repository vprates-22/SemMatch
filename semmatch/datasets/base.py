"""
Module: semmatch.datasets.base
------------------------------

This module defines the abstract base class `BaseDataset`, which serves as a
standardized interface for loading and interacting with various image-based datasets.
It provides common functionalities such as configuration management, dataset
downloading, image caching, and abstract methods that subclasses must implement
to handle dataset-specific logic.

Classes
-------
BaseDataset
    An abstract base class that defines the core methods and properties required
    for any dataset loader within the `semmatch` framework.
"""

import sys
from pathlib import Path
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple, Any, Union, Iterable

import gdown
import torch

from numpy.typing import NDArray
from semmatch.configs.dataset_config import Config, DatasetConfig
from semmatch.utils.download import download_from_url, extract_archive


class BaseDataset(metaclass=ABCMeta):
    """
    Abstract base class for dataset loaders.

    This class provides a standardized interface and shared functionality for
    loading datasets, particularly those involving image pairs and optional downloading
    from remote sources. It is intended to be subclassed with specific implementations
    of the abstract methods to define dataset-specific logic.

    Parameters
    ----------
    config : Union[Config, Dict[str, Any]], optional
        Configuration settings for the dataset. This can be a `Config` object or a
        dictionary. These settings are merged with default values provided by `DatasetConfig`.

    Attributes
    ----------
    config : DatasetConfig
        The configuration object for the dataset, managing settings like `data_path`,
        `pairs_path`, `cache_images`, `max_pairs`, `url`, and `url_download_extension`.
    _pairs : List[Dict[str, Any]]
        A list of dictionaries, where each dictionary represents an image pair and its
        associated ground truth information. This is populated by `read_gt()`.
    _image_cache : Dict[str, Any]
        A dictionary used to cache loaded images if `config.cache_images` is True.

    Notes
    -----
    Subclasses must implement the following abstract methods:
    - `load_images`: To load and optionally cache images or other attributes.
    - `read_image`: To read a single image or attribute from a given path.
    - `read_gt`: To read and parse the dataset's ground truth information.
    - `map_point`: To map 2D points from one image to another using ground truth.
    - `get_inliers`: To determine inliers among matched keypoints based on ground truth.
    - `estimate_pose`: To estimate the relative pose between two images.
    """

    def __init__(self, config: Union[Config, Dict[str, Any]] = None):
        self.config = DatasetConfig(config)

        if not Path(self.config.data_path).exists():
            if self.config.url:
                self.download_dataset()
            else:
                print(
                    'Neither a valid "data_path" nor a "url" was provided to download the dataset.', file=sys.stderr)
                sys.exit(1)

        self._pairs = self.read_gt()

        self._image_cache = {}
        if self.config.cache_images:
            self.load_images()

    @abstractmethod
    def load_images(self, attr: str = 'image') -> None:
        """
        Load and optionally cache images or other attributes (e.g., depth maps).

        Parameters
        ----------
        attr : str, optional
            The name of the attribute to load (e.g., 'image', 'depth').
            Defaults to 'image'.
        """
        raise NotImplementedError

    @abstractmethod
    def read_image(self, path, attr: str = 'image') -> torch.Tensor:
        """
        Read an image or other attribute from a given path.

        Parameters
        ----------
        path : str
            The file path to the image or data to be read.
        attr : str, optional
            The type of data to load (e.g., 'image', 'depth'). Defaults to 'image'.

        Returns
        -------
        torch.Tensor
            The loaded image or data as a PyTorch tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def read_gt(self) -> List[Dict[str, Any]]:
        """
        Read ground truth data for the dataset.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, each representing a data pair with its
            associated ground truth information.

        Notes
        -----
            The structure of the dictionaries in the returned list will depend on
            the specific dataset and the type of ground truth it provides.
        """
        raise NotImplementedError

    @abstractmethod
    def map_point(
        self,
        points: NDArray,
        pair_index: int,
        scale_img0: Iterable[float] = [1.0, 1.0],
        scale_img1: Iterable[float] = [1.0, 1.0]
    ) -> tuple[Tuple[float, float], bool]:
        """ 
        Map 2D points from the first image to the second image
        using ground truth information (e.g., homography, camera pose).

        Subclasses must implement this method to define how points are transformed
        between image coordinate systems based on the dataset's ground truth.

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
        tuple[Tuple[float, float], bool]
            A tuple containing:
            - mapped : (N, 2) Array of mapped 2D points in the target image (image1).
            - valid : (N,) Boolean mask indicating whether each mapped point is valid
            (e.g., inside the target image bounds after mapping).
        """
        raise NotImplementedError

    @abstractmethod
    def get_inliers(
        self,
        mkpts0: NDArray,
        mkpts1: NDArray,
        pair_index: int,
        scale_img0: Iterable[Iterable[float]] = [1.0, 1.0],
        scale_img1: Iterable[Iterable[float]] = [1.0, 1.0],
        threshold: float = 6.0,
    ) -> NDArray:
        """
        Determine inliers among matched keypoints based on ground truth.

        Parameters
        ----------
        mkpts0 : NDArray
            (N, 2) Array of matched keypoints from the first image.
        mkpts1 : NDArray
            (N, 2) Array of corresponding matched keypoints from the second image.
        pair_index : int
            The index of the image pair in the dataset's `_pairs` list.
        scale_img0 : Iterable[Iterable[float]], optional
            Scaling factors for the first image (e.g., from resizing).
            Defaults to [1.0, 1.0].
        scale_img1 : Iterable[Iterable[float]], optional
            Scaling factors for the second image (e.g., from resizing).
            Defaults to [1.0, 1.0].
        threshold : float, optional
            The threshold for considering a match an inlier (e.g., reprojection error).
            Defaults to 6.0.

        Returns
        -------
        NDArray
            Boolean array of shape (N,) indicating which keypoint matches are inliers.
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_pose(
        self,
        pair_index: int,
        mkpts0: NDArray,
        mkpts1: NDArray,
        threshold: float = 6.0,
    ) -> Tuple[NDArray, NDArray]:
        """ 
        Estimates the relative pose (rotation and translation) between two images.

        Parameters
        ----------
        pair_index : int
            The index of the image pair in the dataset's `_pairs` list.
        mkpts0 : NDArray
            (N, 2) Array of matched keypoints from the first image.
        mkpts1 : NDArray
            (N, 2) Array of corresponding matched keypoints from the second image.
        threshold : float, optional
            RANSAC reprojection error threshold in pixels. Defaults to 6.0.

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing the rotation matrix (3x3) and translation vector (3x1)
            between the two images. Returns None if pose estimation fails.
        """
        raise NotImplementedError

    def download_dataset(self, chunk_size: int = 1024) -> None:
        """
        Downloads a dataset archive from a URL, displays download progress,
        extracts its contents, and removes the archive file after extraction.

        Parameters:
        ----------
        chunk_size : int, optional 
            Size (in bytes) of each chunk to read during download.
            Defaults to 1024.
        """
        url = self.config.url
        archive_ext = self.config.url_download_extension or '.zip'

        output_dir = Path(self.config.data_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Definir nome temporário do arquivo
        archive_path = output_dir / f"temp_dataset{archive_ext}"

        try:
            if 'drive.google' in url:
                gdown.download(url, str(archive_path), quiet=False)

            else:
                download_from_url(url, archive_path, chunk_size)

            extract_archive(
                archive_path,
                output_dir,
                remove_after_extraction=True
            )

        except Exception as err:
            print(f"Failed to download or extract dataset: {err}")
            if archive_path.exists():
                archive_path.unlink()
            output_dir.rmdir()

    @property
    def pairs(self) -> List[Dict[str, Any]]:
        return self._pairs

    @property
    def image_cache(self) -> Dict[str, Any]:
        return self._image_cache
