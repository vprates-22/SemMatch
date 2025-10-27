"""
Module: datasets.base_dataset_loader
----------------------------

This module defines the abstract base class `BaseDatasetLoader` for dataset loading.
It establishes a interface for reading images and optionally downloading and 
caching datasets.

Classes:
    BaseDatasetLoader (ABC): 
        An abstract base class that defines methods and configuration needed to load
                    a dataset. Subclasses must implement the `load_images`, 
                    `read_image`, and `read_gt` methods.
"""

import sys
from pathlib import Path
from abc import ABCMeta, abstractmethod
from typing import Dict, Any, List, Tuple

import gdown
import torch

from semmatch.utils.io import combine_dicts
from semmatch.utils.loaders import download_from_url, extract_archive

DATASET_CONFIG_KEYS = [
    'data_path', 'pairs_path', 'cache_images',
    'max_pairs', 'url', 'url_download_extension',
]


class BaseDatasetLoader(metaclass=ABCMeta):
    """
    Abstract base class for dataset loaders.

    This class provides a standardized interface and shared functionality for 
    loading datasets, particularly those involving image pairs and optional 
    downloading from remote sources. It is intended to be subclassed with specific 
    implementations of the abstract methods: `load_images`, `read_image`, and `read_gt`.

    Configuration:
        This class uses a config dictionary that merges user-provided values with
        sensible defaults. The expected keys include:

        - data_path (str): Local path where the dataset is stored.
        - pairs_path (str): Path to the file describing image pairs or labels.
        - cache_images (bool): If True, loads and stores all images in memory.
        - max_pairs (int): Limits the number of pairs to load (-1 for no limit).
        - url (str): URL to download the dataset if it's not found locally.
        - url_download_extension (str): File extension for the archive (e.g., '.zip').

    Important:
        - Subclasses must implement the abstract methods to define dataset-specific logic.
        - The `download_dataset()` method supports both direct URLs and Google Drive links.
    """
    default_config = {
        'data_path': '',
        'pairs_path': '',
        'cache_images': False,
        'max_pairs': -1,
        'url': '',
        'url_download_extension': ''
    }

    def __init__(self, config: Dict[str, Any] = None):
        self.config = combine_dicts(self.default_config, config or {})

        if not Path(self.config['data_path']).exists():
            if self.config.get('url'):
                self.download_dataset()
            else:
                print(
                    'Neither a valid "data_path" nor a "url" was provided to download the dataset.', file=sys.stderr)
                sys.exit(1)

        self._pairs = self.read_gt()

        self._image_cache = {}
        if self.config['cache_images']:
            self.load_images()

    @abstractmethod
    def load_images(self, attr: str = 'image') -> None:
        raise NotImplementedError

    @abstractmethod
    def read_image(self, path, attr: str = 'image') -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def read_gt(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def map_point(self, point: Tuple[int, int], pair_index: int, scale_img0: float = 1.0, scale_img1: float = 1.0) -> Tuple[int, int]:
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

        Raises:
        ------
        Exception
            If the download or extraction process fails. In such cases,
            the temporary archive and output directory are cleaned up.
        """
        url = self.config['url']
        archive_ext = self.config['url_download_extension'] or '.zip'

        output_dir = Path(self.config['data_path'])
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

        except Exception as e:
            print(f"Failed to download or extract dataset: {e}")
            if archive_path.exists():
                archive_path.unlink()
            output_dir.rmdir()

    @property
    def pairs(self) -> List[Dict[str, Any]]:
        return self._pairs

    @property
    def image_cache(self) -> Dict[str, Any]:
        return self._image_cache
