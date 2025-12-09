"""
Module: semmatch.statistics.metrics.pose
----------------------------------------

This module defines metrics specifically designed for evaluating pose estimation
results within the SemMatch statistics pipeline. It includes metrics for
calculating average rotation and translation errors, as well as Area Under the Curve (AUC)
for pose estimation accuracy across various thresholds.
"""

import numpy as np

from semmatch.configs.base import Config
from semmatch.statistics.metrics import BaseMetric
from semmatch.statistics.pipeline_data import PoseErrorResult


class PoseMetric(BaseMetric):
    """
    Base class for pose-related metrics.

    This class serves as a common ancestor for metrics that process
    `PoseErrorResult` objects.
    """

    _allowed_result_type = PoseErrorResult


class RotationError(PoseMetric):
    """
    Calculates the average rotation error.

    This metric accumulates individual rotation errors from `PoseErrorResult`
    objects and computes their mean.

    Attributes
    ----------
    _allowed_result_type : Type[PoseErrorResult]
        Specifies that this metric processes `PoseErrorResult` objects.
    _raw_results : list[float]
        A list to store all individual rotation errors accumulated across `update` calls.
    """

    def update(self, data: PoseErrorResult) -> None:
        self._raw_results.append(data.rotation_error)

    def compute(self) -> None:
        self._result = np.mean(self._raw_results)


class TranslationError(PoseMetric):
    """
    Calculates the average translation error.

    This metric accumulates individual translation errors from `PoseErrorResult`
    objects and computes their mean.

    Attributes
    ----------
    _allowed_result_type : Type[PoseErrorResult]
        Specifies that this metric processes `PoseErrorResult` objects.
    _raw_results : list[float]
        A list to store all individual translation errors accumulated across `update` calls.
    """

    def update(self, data: PoseErrorResult) -> None:
        self._raw_results.append(data.translation_error)

    def compute(self) -> None:
        self._result = np.mean(self._raw_results)


class AUCBase(PoseMetric):
    """
    Base class for Area Under the Curve (AUC) metrics for pose estimation.

    This class provides the common infrastructure for calculating AUC based on
    pose errors (rotation, translation, or combined). It stores raw errors
    grouped by the threshold used during pose estimation and computes AUC
    for a set of predefined `pose_thresholds`.

    Attributes
    ----------
    _allowed_result_type : Type[PoseErrorResult]
        Specifies that this metric processes `PoseErrorResult` objects.
    _raw_results : dict
        A dictionary where keys are pose estimation thresholds and values are
        lists of individual errors (rotation, translation, or combined)
        accumulated across `update` calls for that threshold.
    _result : dict
        A dictionary storing the computed AUC values for each `pose_thresholds`
    defined in the configuration, keyed by a descriptive string.

    Parameters
    ----------
    config : Config, optional
        Configuration object for the metric. Expected key: 'pose_thresholds'.
    """

    def __init__(self, config=None):
        default_config = Config({
            'pose_thresholds': [5, 10, 20]
        }).merge_config(config)
        super().__init__(default_config)
        self._result: dict = {}

    def compute(self) -> None:
        pose_thresholds = self._config['pose_thresholds']
        final_results = {}

        errors = np.array(self._raw_results, dtype=float)
        aucs = self._pose_auc(errors, pose_thresholds)
        final_results = {
            f'pose_thresholds:{t}': a * 100
            for t, a in zip(pose_thresholds, aucs)
        }

        self._result = final_results

    def reset(self):
        self._result = []
        self._raw_results = {}

    @staticmethod
    def _pose_auc(errors: np.ndarray, thresholds: list[float]) -> list[float]:
        """
        Calculates the Area Under the Curve (AUC) for a given set of errors
        and thresholds.

        The AUC is computed by sorting the errors, calculating recall values,
        and then integrating the precision-recall curve up to each threshold.
        The result is normalized by the threshold value.

        Parameters
        ----------
        errors : np.ndarray
            A 1D NumPy array of individual errors (e.g., rotation errors,
            translation errors).
        thresholds : list[float]
            A list of threshold values at which to compute the AUC.

        Returns
        -------
        list[float]
            A list of AUC values, one for each threshold, normalized by the
            corresponding threshold. Returns 0.0 for all thresholds if `errors`
            is empty.

        Notes

        The AUC is computed by sorting the errors, calculating recall values,
        and then integrating the precision-recall curve up to each threshold.
        The result is normalized by the threshold value.

        Parameters
        ----------
        errors : np.ndarray
            A 1D NumPy array of individual errors (e.g., rotation errors,
            translation errors).
        thresholds : list[float]
            A list of threshold values at which to compute the AUC.

        Returns
        -------
        list[float]
            A list of AUC values, one for each threshold, normalized by the
            corresponding threshold. Returns 0.0 for all thresholds if `errors`
        """

        errors = np.asarray(errors)
        if len(errors) == 0:
            return [0.0] * len(thresholds)

        N = len(errors)

        # Ordenação dos erros
        errors_sorted = np.sort(errors)
        recalls = (np.arange(1, N + 1)) / N  # fração de keypoints corretos

        # Base da curva (0,0) até todos os erros
        base_e = np.r_[0.0, errors_sorted]
        base_r = np.r_[0.0, recalls]

        aucs = []
        for t in thresholds:
            if t == 0:
                aucs.append(0.0)
                continue

            # índice onde t deve entrar na curva
            idx = np.searchsorted(base_e, t, side='right')

            # construir a curva até t
            if idx == 0:
                e = np.array([0.0, t])
                r = np.array([0.0, 0.0])
            else:
                e = np.r_[base_e[:idx], t]
                r = np.r_[base_r[:idx], base_r[idx - 1]]

            # calcular AUC normalizado
            auc = np.trapz(r, x=e) / t
            aucs.append(float(auc))

        return aucs


class AUCRotation(AUCBase):
    """
    Computes the Area Under the Curve (AUC) for rotation errors.

    This metric inherits from `AUCBase` and specifically updates its
    internal state with rotation errors from `PoseErrorResult` objects.
    """

    def update(self, data: PoseErrorResult) -> None:
        self._raw_results.append(float(data.rotation_error))


class AUCTranslation(AUCBase):
    """
    Computes the Area Under the Curve (AUC) for translation errors.

    This metric inherits from `AUCBase` and specifically updates its
    internal state with translation errors from `PoseErrorResult` objects.
    """

    def update(self, data: PoseErrorResult) -> None:
        self._raw_results.append(float(data.translation_error))


class AUC(AUCBase):
    """
    Computes the Area Under the Curve (AUC) for the combined pose error.

    The combined pose error is defined as the maximum of the rotation error
    and the translation error for a given `PoseErrorResult`. This class
    inherits from `AUCBase` and updates its internal state with these
    combined errors.

    This class keeps the original name `AUC` but now computes only the combined
    pose error AUC. Use `AUCRotation` and `AUCTranslation` for per-component AUCs.
    """

    def update(self, data: PoseErrorResult) -> None:
        combined = max(float(data.rotation_error),
                       float(data.translation_error))
        self._raw_results.append(combined)
