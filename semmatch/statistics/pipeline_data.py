"""
Module: semmatch.statistics.pipeline_data
-----------------------------------------

This module defines the data structures used throughout the data analysis pipeline,
categorized into Raw Input, Generated Data, and Analysis Results.
"""

from dataclasses import dataclass

import numpy as np
from typing import Union
from numpy.typing import NDArray

from semmatch.datasets.base import BaseDataset

# --- Camada 1: Input ---


@dataclass(frozen=True)
class RawDataInput:
    """
    Raw input data for a single iteration of the analysis pipeline.

    This dataclass holds all the initial, unprocessed data required for
    a given pair of images.

    Attributes
    ----------
    image0_path : str
        File path to the first image.
    image1_path : str
        File path to the second image.
    image0 : NDArray
        The first image as a NumPy array.
    image1 : NDArray
        The second image as a NumPy array.
    mkpts0 : NDArray
        Keypoints detected in the first image.
    mkpts1 : NDArray
        Keypoints detected in the second image.
    dataset : BaseDataset
        The dataset object from which this data originated.
    pair_index : int
        The index of the current image pair within the dataset.
    """
    image0_path: str
    image1_path: str
    image0: NDArray
    image1: NDArray
    mkpts0: NDArray
    mkpts1: NDArray
    dataset: BaseDataset
    pair_index: int

# --- Camada 2: Dados Gerados ---


class GeneratedData:
    """
    Base class for all generated data structures.

    This class serves as a marker interface for data that is derived
    or processed from `RawDataInput` by `DataGenerator` components.
    """
    pass


@dataclass(frozen=True)
class KeypointData(GeneratedData):
    """
    Generated data containing keypoints for two images.

    Attributes
    ----------
    mkpts0 : NDArray
        Keypoints from the first image.
    mkpts1 : NDArray
        Keypoints from the second image.
    """
    mkpts0: NDArray
    mkpts1: NDArray


@dataclass(frozen=True)
class InlierData(GeneratedData):
    """
    Generated data containing a boolean mask for inliers and the threshold used.

    Attributes
    ----------
    inliers : NDArray[np.bool_]
        A boolean array indicating which matches are considered inliers.
    threshold : float
        The threshold value used to determine inliers.
    """
    inliers: NDArray[np.bool_]
    threshold: float


@dataclass(frozen=True)
class ProjectionData(GeneratedData):
    """
    Generated data containing projected points, their validity, and keypoints from the second image.

    Attributes
    ----------
    projections : NDArray[np.float32]
        The 2D coordinates of keypoints from the first image projected onto the second image plane.
    valid : NDArray[np.bool_]
        A boolean array indicating the validity of each projection (e.g., within image bounds, depth-consistent).
    mkpts1 : NDArray[np.float32]
        Keypoints from the second image, used for comparison with projections.
    """
    projections: NDArray[np.float32]
    valid: NDArray[np.bool_]
    mkpts1: NDArray[np.float32]


@dataclass(frozen=True)
class PoseData(GeneratedData):
    """
    Generated data containing estimated and ground truth camera poses.

    Attributes
    ----------
    threshold : float
        The threshold used for pose estimation (e.g., RANSAC threshold).
    R_est : NDArray[np.float32]
        Estimated rotation matrix (3x3).
    t_est : NDArray[np.float32]
        Estimated translation vector (3,).
    R_gt : NDArray[np.float32]
        Ground truth rotation matrix (3x3).
    t_gt : NDArray[np.float32]
        Ground truth translation vector (3,).
    """
    threshold: float

    R_est: NDArray[np.float32]
    t_est: NDArray[np.float32]
    R_gt: NDArray[np.float32]
    t_gt: NDArray[np.float32]


@dataclass(frozen=True)
class MaskData(GeneratedData):
    """
    Generated data containing keypoints and their corresponding segmentation masks.

    Attributes
    ----------
    mkpts0 : NDArray
        Keypoints from the first image.
    mkpts1 : NDArray
        Keypoints from the second image.
    masks0 : NDArray
        Segmentation masks corresponding to `mkpts0` in the first image.
    masks1 : NDArray
        Segmentation masks corresponding to `mkpts1` in the second image.
    """
    mkpts0: NDArray
    mkpts1: NDArray
    masks0: NDArray
    masks1: NDArray

# --- Camada 3: Resultados da Análise ---


@dataclass(frozen=True)
class AnalysisResult:
    """
    Base class for all analysis results.

    This class serves as a marker interface for data that is the output
    of an `DataAnalyzer` component.

    Attributes
    ----------
    key : Union[str, int, float], optional
        A unique identifier for this specific analysis result,
        allowing multiple results from a single analyzer. Defaults to "default".
    """
    key: Union[str, int, float] = "default"


@dataclass(frozen=True)
class MatchResult(AnalysisResult):
    """
    Analysis result containing counts for true positives, false positives,
    false negatives, and true negatives, typically from a matching task.

    Attributes
    ----------
    true_positives : int, optional
        Number of true positive matches.
    false_positives : int, optional
        Number of false positive matches.
    false_negatives : int, optional
        Number of false negative matches.
    true_negatives : int, optional
        Number of true negative matches.
    threshold : float, optional
        The threshold used to determine the classification (e.g., inlier threshold).
    """
    true_positives: int = None
    false_positives: int = None
    false_negatives: int = None
    true_negatives: int = None

    threshold: float = None


@dataclass(frozen=True)
class ErrorResult(AnalysisResult):
    """
    Analysis result containing a list of error values.

    Attributes
    ----------
    errors : list[float], optional
        A list of individual error measurements.
    """
    errors: list[float] = None


@dataclass(frozen=True)
class PoseErrorResult(AnalysisResult):
    """
    Analysis result containing rotation and translation errors for a pose estimation.

    Attributes
    ----------
    rotation_error : float, optional
        The rotation error (e.g., in degrees).
    translation_error : float, optional
        The translation error (e.g., in meters or pixels).
    threshold : float, optional
        The threshold used during pose estimation (e.g., RANSAC threshold).
    """
    rotation_error: float = None
    translation_error: float = None

    threshold: float = None
