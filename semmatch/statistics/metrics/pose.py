"""
Module: semmatch.statistics.metrics.pose
----------------------------------------

This module defines metrics specifically designed for evaluating pose estimation
results within the SemMatch statistics pipeline. It includes metrics for
calculating average rotation and translation errors, as well as Area Under the Curve (AUC)
for pose estimation accuracy across various thresholds.
"""

import numpy as np
from collections import defaultdict

from semmatch.configs.base import Config
from semmatch.statistics.metrics import BaseMetric
from semmatch.statistics.pipeline_data import PoseErrorResult


class RotationError(BaseMetric):
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

    _allowed_result_type = PoseErrorResult

    def update(self, data: PoseErrorResult) -> None:
        self._raw_results.append(data.rotation_error)

    def compute(self) -> None:
        self._result = np.mean(self._raw_results)


class TranslationError(BaseMetric):
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

    _allowed_result_type = PoseErrorResult

    def update(self, data: PoseErrorResult) -> None:
        self._raw_results.append(data.translation_error)

    def compute(self) -> None:
        self._result = np.mean(self._raw_results)


class AUC(BaseMetric):
    """
    Calculates the Area Under the Curve (AUC) for pose estimation errors.

    This metric evaluates the accuracy of pose estimation by computing AUC
    for rotation errors, translation errors, and a combined pose error
    (maximum of rotation and translation errors) across a set of predefined
    thresholds.

    The `_raw_results` attribute stores errors grouped by the `threshold`
    from the `PoseErrorResult` objects. The `_result` attribute stores
    the computed AUC values for each threshold and error type.

    Attributes
    ----------
    _allowed_result_type : Type[PoseErrorResult]
        Specifies that this metric processes `PoseErrorResult` objects.
    _raw_results : dict
        A dictionary to store accumulated rotation and translation errors,
        keyed by the `threshold` from `PoseErrorResult`.
        Each value is a list of dictionaries, where each dictionary contains
        'rotation_errors' and 'translation_errors'.
    _result : dict
        A dictionary to store the final computed AUC values, keyed by the
        `threshold` from `PoseErrorResult`. Each value is a dictionary
        containing 'rotation_auc', 'translation_auc', and 'combined_auc'.

    Parameters
    ----------
    config : Config, optional
        Configuration settings for the metric.
        Expected keys:
        - 'pose_thresholds' (list[float]): A list of thresholds (in degrees/units)
          at which to evaluate the AUC. Defaults to [5, 10, 20].
    """

    _allowed_result_type = PoseErrorResult

    def __init__(self, config=None):
        default_config = Config(
            {'pose_thresholds': [5, 10, 20]}
        ).merge_config(config)

        super().__init__(default_config)
        self._raw_results: dict = defaultdict(list)
        self._result: dict = {}

    def update(self, data: PoseErrorResult) -> None:
        threshold = data.threshold

        self._raw_results[threshold].append({
            'rotation_error': data.rotation_error,
            'translation_error': data.translation_error,
        })

    def compute(self) -> None:
        pose_thresholds = self._config['pose_thresholds']
        final_results = {}

        for thresh_key, entries in self._raw_results.items():
            # Collect all rotation and translation errors for this threshold
            rot_errs = [
                err['rotation_error']
                for err in entries 
            ]
            trans_errs = [
                err['translation_error']
                for err in entries 
            ]
            rot_errs = np.array(rot_errs, dtype=float)
            trans_errs = np.array(trans_errs, dtype=float)

            # Combined pose metric (max(err_t, err_R))
            combined = np.maximum(rot_errs, trans_errs)

            # AUC separately and combined
            auc_rot = self._pose_auc(rot_errs, pose_thresholds)
            auc_trans = self._pose_auc(trans_errs, pose_thresholds)
            auc_comb = self._pose_auc(combined, pose_thresholds)
            auc_rot = {t: a * 100 for t, a in zip(pose_thresholds, auc_rot)}
            auc_trans = {t: a * 100 for t,
                         a in zip(pose_thresholds, auc_trans)}
            auc_comb = {t: a * 100 for t, a in zip(pose_thresholds, auc_comb)}

            final_results[thresh_key] = {
                "rotation_auc": auc_rot,
                "translation_auc": auc_trans,
                "combined_auc": auc_comb,
            }

        # Store result in the metric system
        self._result = final_results

    def reset(self):
        self._raw_results = defaultdict(list)
        self._result = {}

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
