"""
Module: semmatch.configs.dataset_config
---------------------------------------

This module defines the `DatasetConfig` class, which extends the base `Config`
class to provide specific configuration settings for dataset handling within
the SemMatch project.

Classes:
    DatasetConfig: Configuration settings for datasets, including paths, caching,
                   and download information.
"""

from semmatch.configs.base import Config


class DatasetConfig(Config):
    """
    Configuration settings for datasets within the SemMatch project.

    This class extends the base `Config` class to define specific parameters
    related to dataset handling, such as file paths, caching behavior,
    and download information.

    Attributes
    ----------
    _config : dict
        A class-level dictionary defining the default configuration parameters:
        - 'data_path' (str): The local path where the dataset files are stored.
                             Defaults to an empty string.
        - 'pairs_path' (str): The path to a file containing information about
                              image pairs (e.g., ground truth, file paths).
                              Defaults to an empty string.
        - 'cache_images' (bool): If True, images will be loaded into memory
                                 and cached for faster access. Defaults to False.
        - 'max_pairs' (int): The maximum number of image pairs to load from
                             the dataset. A value of -1 means no limit.
                             Defaults to -1.
        - 'url' (str): The URL from which the dataset can be downloaded.
                       Defaults to an empty string.
        - 'url_download_extension' (str): The file extension of the dataset
                                          archive (e.g., '.zip', '.tar').
                                          Defaults to an empty string.
    """
    _config = {
        'data_path': '',
        'pairs_path': '',
        'cache_images': False,
        'max_pairs': -1,
        'url': '',
        'url_download_extension': '',
    }
