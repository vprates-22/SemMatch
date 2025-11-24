import numpy as np

from semmatch.statistics.pipeline_data import ErrorResult
from semmatch.statistics.metrics.base import BaseMetric, RAW_RESULT_DEFAULT_KEY, RESULT_DEFAULT_KEY


class ErrorMetrics(BaseMetric):
    _allowed_result_type = ErrorResult


class ReprojectionAverageError(ErrorMetrics):
    def __init__(self, config=None):
        super().__init__(config)

        self._all_errors: list[float] = []

    def update(self, data):
        if isinstance(data, ErrorResult):
            data = [data]

        for error_data in data:
            raw_results = self._raw_results.get(RAW_RESULT_DEFAULT_KEY, [])
            raw_results.append(np.mean(error_data.errors)
                               if error_data.errors else 0.0)

            self._all_errors.extend(error_data.errors)

    def compute(self) -> None:
        self._result[RESULT_DEFAULT_KEY] = np.mean(
            self._all_errors) if self._all_errors else 0.0

    def reset(self) -> None:
        super().reset()
        self._all_errors = []
