"""
Module: semmatch.statistics.analyzers.base
------------------------------------------

This module defines the abstract base class for all data analyzers within the SemMatch
statistics pipeline.
"""
from abc import ABCMeta, abstractmethod

from typing import Iterable

from semmatch.configs.base import Config
from semmatch.statistics.pipeline_data import AnalysisResult
from semmatch.statistics.data_generators.base import DataGenerator, GeneratedData


class DataAnalyzer(metaclass=ABCMeta):
    """
    Abstract base class for all data analyzers in the SemMatch statistics pipeline.

    DataAnalyzers are responsible for processing `GeneratedData` from `DataGenerator`s
    and producing `AnalysisResult`s. Each analyzer must declare its dependencies
    on `DataGenerator` types and the type of `AnalysisResult` it produces.

    Attributes
    ----------
    _data_generator_depedencies : list[DataGenerator]
        A list of `DataGenerator` classes that this analyzer depends on.
        These generators will be run before this analyzer.
    _data_result_type : type[AnalysisResult]
        The specific `AnalysisResult` subclass that this analyzer is designed
        to produce. This is used for validation within the `PipelineOrchestrator`.

    Parameters
    ----------
    config : Config, optional
        Configuration settings for the analyzer. Defaults to None, which
        results in an empty configuration.
    """
    _data_generator_depedencies: list[DataGenerator] = []
    _data_result_type: type[AnalysisResult] = None

    def __init__(self, config: Config = None):
        if not self._data_result_type:
            raise NotImplementedError(
                f"DataAnalyzer '{self.__class__.__name__}' must define '_data_result_type'."
            )

        base_config = Config({})
        self._config = base_config.merge_config(config)

    @abstractmethod
    def analyze(self, generated_data: dict[type[DataGenerator], list[GeneratedData]]) -> Iterable[AnalysisResult]:
        """
        Perform data analysis.

        This method takes a dictionary of generated data (keyed by `DataGenerator` type)
        and processes it to produce analysis results.

        Parameters
        ----------
        generated_data : dict[type[DataGenerator], list[GeneratedData]]
            A dictionary where keys are `DataGenerator` types and values are lists
            of `GeneratedData` objects produced by those generators.

        Returns
        -------
        Iterable[AnalysisResult]
            An iterable (e.g., a list) of `AnalysisResult` objects.
        """
        raise NotImplementedError

    @classmethod
    def get_dependencies(cls):
        """
        Returns the list of `DataGenerator` types that this analyzer depends on.
        """
        return cls._data_generator_depedencies

    @classmethod
    def get_result_type(cls):
        """
        Returns the `AnalysisResult` type that this analyzer produces.
        """
        return cls._data_result_type
