"""
Module: semmatch.configs.result_routes_config
---------------------------------------------

This module defines the `ResultRoutesConfig` class, which extends the base `Config`
class to provide specific configuration settings for handling result files within
the SemMatch project.

Classes:
 ResultRoutesConfig: Configuration settings for result file paths and naming conventions.
"""
from semmatch.configs.base import Config


class ResultRoutesConfig(Config):
    """
    Configuration settings for result file paths and naming conventions.

    This class extends the base `Config` class and centralizes configuration
    values related to how result files are named and where they are stored.

    Attributes
    ----------
    _config : dict
        A class-level dictionary defining the default configuration parameters:
        - `arr_name` (str): The JSON array or top-level key name that contains
            match entries within result files. Defaults to `'all_matches'`.
        - `results_file_path` (str): Default filename (or relative path) used
            to store aggregated results. Defaults to `'results.json'`.
    """
    _config = {
        'arr_name': 'all_matches',
        'results_file_path': 'results.json',
    }
