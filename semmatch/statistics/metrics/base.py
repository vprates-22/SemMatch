from collections import defaultdict

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Any, Type, Union

from semmatch.configs.base import Config
from semmatch.statistics.pipeline_data import AnalysisResult

RAW_RESULT_DEFAULT_KEY = "metric_raw_results"
RESULT_DEFAULT_KEY = "metric_raw_results"


class BaseMetric(metaclass=ABCMeta):
    _allowed_result_type: Union[Type[AnalysisResult],
                                list[Type[AnalysisResult]]] = None

    _result: Dict = defaultdict(float)
    _raw_results: Dict = defaultdict(list)

    def __init__(self, config: Config = None):
        if not self._allowed_result_type:
            raise NotImplementedError(
                f"Metric '{self.__class__.__name__}' must define '_allowed_result_type'."
            )

        self._config = config

    @abstractmethod
    def update(self, data: list[AnalysisResult]) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> None:
        raise NotImplementedError

    def reset(self):
        self._result = 0
        self._raw_results = []

    def get_last_raw_result(self) -> Any:
        return self._raw_results[-1] if self._raw_results else None

    def get_raw_results(self) -> List:
        return self._raw_results

    def get_result(self) -> float:
        return self._result

    @classmethod
    def get_allowed_result_types(cls) -> list[type]:
        return cls._allowed_result_type if isinstance(cls._allowed_result_type, list) else [cls._allowed_result_type]
