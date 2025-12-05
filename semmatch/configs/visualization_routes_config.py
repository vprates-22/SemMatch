"""
Module: semmatch.configs.visualization_routes_config
----------------------------------------------------

This module defines the `VisualizationRoutesConfig` class, which extends the base `Config`
class to provide specific configuration settings for visualization-related file paths
and model parameters within the SemMatch project.

Classes:
 VisualizationRoutesConfig: Configuration settings for visualization file paths and models.
"""
from semmatch.configs.base import Config


class VisualizationRoutesConfig(Config):
    """
    Configuration settings for visualization file paths and model parameters.

    This class extends the base `Config` class and groups configuration
    parameters used by visualization tooling (e.g. where to read matches from,
    which SAM model to use, and batching parameters for mask generation).

    Attributes
    ----------
    _config : dict
        A class-level dictionary defining the default configuration parameters:
        - `arr_name` (str): The JSON array/top-level key name containing match entries. Defaults to `'all_matches'`.
        - `matches_file_path` (str): Default path to the matches file used for visualization.
        - `sam_model` (str): Filename of the SAM model to use for mask generation. Defaults to `'sam2.1_l.pt'`.
        - `batch_size` (int): Batch size for model inference. Defaults to `200`.

    Notes
    -----
    These defaults are suitable for local workflows; override them via a
    higher-level configuration when integrating visualizations into pipelines.
    """
    _config = {
        'arr_name': 'all_matches',
        'matches_file_path': '',
        'sam_model': 'sam2.1_l.pt',
        'batch_size': 200
    }
