from typing import Dict, List, Any
from abc import ABCMeta, abstractmethod

from semmatch.statistics.base_update_data import UpdateData


class BaseMetric(metaclass=ABCMeta):
    _dependencies: List[type["BaseMetric"]] = []
    _dependency_objects: Dict[type["BaseMetric"], "BaseMetric"] = {}

    _result: float = 0
    _raw_results: List = []

    def __init__(self, metric_objects: Dict[type["BaseMetric"], "BaseMetric"]):
        for dependency in self._dependencies:
            self._dependency_objects[dependency] = metric_objects[dependency]

    @abstractmethod
    def update(self, data: UpdateData) -> None:
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
    def get_dependencies(cls) -> List:
        return cls._dependencies
