"""
Module: semmatch.statistics_new.data_generators.geometry
--------------------------------------------------------

This module provides data generators for geometric analysis, specifically for
determining inliers among matched keypoints and projecting points between images.
"""
import numpy as np
from collections.abc import Iterable

from numpy.typing import NDArray

from semmatch.configs.base import Config
from semmatch.statistics.data_generators.base import DataGenerator
from semmatch.statistics.pipeline_data import RawDataInput, InlierData, ProjectionData, PoseData


class InlierGenerator(DataGenerator):
    """
    Generates inlier masks for matched keypoints based on various geometric thresholds.

    This generator uses the `get_inliers` method from the dataset to determine
    which keypoint matches are geometrically consistent for a given threshold.

    Parameters
    ----------
    config : Config, optional
        Configuration object for the generator.
        Expected keys:
        - 'inliers_thresholds' (list of float): A list of thresholds to use
          for inlier determination. Defaults to [0.5, 1.0, ..., 6.0].

    """

    def __init__(self, config=None):
        default_config = Config({
            'inlier_threshold': [6.0],
        })
        super().__init__(default_config.merge_config(config))

    def generate(self, raw_input: RawDataInput) -> list[InlierData]:
        threshold = self._config.inlier_threshold
        if not isinstance(threshold, Iterable):
            threshold = [threshold]

        data = []
        for t in threshold:
            data.append(InlierData(
                threshold=threshold,
                inliers=raw_input.dataset.get_inliers(
                    raw_input.mkpts0,
                    raw_input.mkpts1,
                    raw_input.pair_index,
                    threshold=t,
                )
            ))
        return data


class ProjectionGenerator(DataGenerator):
    """
    Generates projected points and their validity masks by mapping keypoints
    from one image to another using the ground truth transformation.

    This generator uses the `map_point` method from the dataset to project
    keypoints from `image0` to `image1` and determines which projections
    are valid (e.g., within image bounds, depth-consistent).

    Parameters
    ----------
    config : Config, optional
        Configuration object for the generator. Currently, no specific
        configuration parameters are defined for this generator.

    Returns
    -------
    ProjectionData
        A dataclass containing:
        - projections (NDArray[np.float32]): The projected 2D points in `image1`.
        - valid (NDArray[np.bool_]): A boolean mask indicating the validity
          of each projection.
    """

    def __init__(self, config: Config = None):
        super().__init__(config)

    def generate(self, raw_input: RawDataInput) -> list[ProjectionData]:
        projections, valid = raw_input.dataset.map_point(
            raw_input.mkpts0,
            raw_input.pair_index,
        )
        return [ProjectionData(
            projections=projections,
            valid=valid,
            mkpts1=raw_input.mkpts1,
        )]


class PoseEstimationGenerator(DataGenerator):
    """
    Generates estimated camera poses based on matched keypoints.

    This generator uses the `estimate_pose` method from the dataset to compute
    the relative pose (rotation and translation) between two images given
    matched keypoints and a threshold.

    Parameters
    ----------
    config : Config, optional
        Configuration object for the generator.
        Expected keys:
        - 'pose_threshold' (float): Threshold for pose estimation. Defaults to 6.0.

    Returns
    -------
    list[PoseEstimationData]
        A list of dataclasses containing estimated poses.
    """

    def __init__(self, config: Config = None):
        default_config = Config({
            'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        })
        super().__init__(default_config.merge_config(config))

    def generate(self, raw_input: RawDataInput) -> list[PoseData]:
        ransac_threshold = self._config.pose_thresholds
        if not isinstance(ransac_threshold, Iterable):
            ransac_threshold = [ransac_threshold]

        dataset = raw_input.dataset
        pair_index = raw_input.pair_index

        data = []
        for threshold in ransac_threshold:
            R_est, t_est = dataset.estimate_pose(
                pair_index=pair_index,
                mkpts0=raw_input.mkpts0,
                mkpts1=raw_input.mkpts1,
                threshold=threshold,
            )
            T_0to1 = dataset.pairs[pair_index].get(
                'T_0to1', None)

            R_gt, t_gt = None, None
            if T_0to1:
                R_gt = T_0to1[:3, :3]
                t_gt = T_0to1[:3, 3]

            data.append(PoseData(
                threshold=threshold,
                R_est=R_est,
                t_est=t_est,
                R_gt=R_gt,
                t_gt=t_gt,
            ))

        return data
