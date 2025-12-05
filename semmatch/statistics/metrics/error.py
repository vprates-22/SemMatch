"""
Module: semmatch.statistics.metrics.error
-----------------------------------------

This module defines metrics specifically designed for evaluating error-related
analysis results within the SemMatch statistics pipeline.
"""

import numpy as np

from semmatch.statistics.metrics.base import BaseMetric
from semmatch.statistics.pipeline_data import ErrorResult


class ErrorMetrics(BaseMetric):
    """
    Base class for error-related metrics.

    This class serves as a common base for metrics that process `ErrorResult`
    objects. It sets the `_allowed_result_type` to `ErrorResult`, ensuring
    that subclasses are compatible with error analysis data.
    """
    _allowed_result_type = ErrorResult


class ReprojectionAverageError(ErrorMetrics):
    """
    Calculates the average reprojection error.

    This metric accumulates all individual errors from `ErrorResult` objects
    and computes their mean. It also stores the mean error for each update
    call in `_raw_results`.

    Attributes
    ----------
    _all_errors : list[float]
        A list to store all individual errors accumulated across `update` calls.

    Parameters
    ----------
    config : Config, optional
        Configuration settings for the metric. Defaults to None.
    """

    def __init__(self, config=None):
        super().__init__(config)

        self._all_errors: list[float] = []

    def update(self, data):
        self._raw_results.append(np.mean(data.errors)
                                 if data.errors else 0.0)
        self._all_errors.extend(data.errors)

    def compute(self) -> None:
        self._result = np.mean(self._all_errors) if self._all_errors else 0.0

    def reset(self) -> None:
        super().reset()
        self._all_errors = []
