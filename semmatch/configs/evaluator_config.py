"""
Module: semmatch.configs.evaluator_config
-----------------------------------------

This module defines the `EvaluatorConfig` class, which extends the base `Config`
class to provide specific configuration settings for the evaluation pipeline
within the SemMatch project.

Classes:
 EvaluatorConfig: Configuration settings for the evaluation process,
 including metrics, dataset, and various thresholds.
"""

from semmatch.configs.base import Config
from semmatch.configs.dataset_config import DatasetConfig

from semmatch.settings import RESULTS_PATH, MATCHES_PATH, VISUALIZATIONS_PATH


class EvaluatorConfig(Config):
    """
    Configuration settings for the evaluation process within the SemMatch project.

    This class extends the base `Config` class to provide configuration options
    for running evaluations. It centralizes choices such as which metrics to
    compute, dataset selection and its configuration, multiprocessing options,
    and file-system paths used to store results, matches and visualizations.

    Attributes
    ----------
    _config : dict
        A class-level dictionary defining the default configuration parameters:
        - `metrics` (list): List of metric class references to run. Defaults to an empty list.
        - `dataset` (str): Identifier of the dataset to use (e.g. 'scannet').
            Defaults to `'scannet'`.
        - `n_workers` (int): Number of worker processes for parallel evaluation.
            Defaults to `8`.
        - `metrics_config` (dict): Sub-configuration for metric-related options,
            including:
                - `mask_model` (str): Masking backend identifier (e.g. 'sam').
                - `mask_batch_size` (int): Batch size for mask inference.
                - `sam_model` (str): SAM model filename to use.
                - `lpips_net` (str): Network backbone for LPIPS ('alex'|'vgg').
                - `inlier_threshold` (float): Pixel threshold used when deciding inliers.
                - `projection_threshold` (float): Pixel threshold for projection comparison.
                - `ransac_thresholds` (list[float]): List of RANSAC thresholds to evaluate.
                - `pose_thresholds` (list[int]): Pose error thresholds (in degrees) to evaluate.
        - `dataset_config` (dict): Base dataset configuration (copied from `DatasetConfig`).
        - `resize` (None|tuple[int,int]): Optional image resize target (width, height).
        - `results_path` (str): Directory path where aggregated results are saved.
        - `matches_path` (str): Directory path where per-pair match files are saved.
        - `visualization_path` (str): Directory path where visualizations are saved.
    """
    _config = {
        'dataset': 'scannet',
        'n_workers': 8,
        'metrics_config': {
            'mask_model': 'sam',
            'mask_batch_size': 200,
            'sam_model': 'sam2.1_l.pt',
            'lpips_net': 'alex',
            'inlier_threshold': 6.0,
            'projection_threshold': 6.0,
            'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            'pose_thresholds': [5, 10, 20],
        },
        'dataset_config': DatasetConfig._config.copy(),
        'resize': None,
        'results_path': RESULTS_PATH,
        'matches_path': MATCHES_PATH,
        'visualization_path': VISUALIZATIONS_PATH,
    }
