from abc import ABCMeta, abstractmethod

from typing import Iterable

from semmatch.configs.base import Config
from semmatch.statistics.pipeline_data import AnalysisResult
from semmatch.statistics.data_generators.base import DataGenerator, GeneratedData


class DataAnalyzer(metaclass=ABCMeta):
    _data_generator_depedencies: list[DataGenerator] = []
    _data_result_type: type[AnalysisResult] = None

    def __init__(self, config: Config = None):
        base_config = Config({})
        self._config = base_config.merge_config(config)

    @abstractmethod
    def analyze(self, generated_data: dict[type[DataGenerator], list[GeneratedData]]) -> Iterable[AnalysisResult]:
        raise NotImplementedError

    @classmethod
    def get_dependencies(cls):
        return cls._data_generator_depedencies
