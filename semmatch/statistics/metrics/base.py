"""
Module: semmatch.statistics.metrics.base
----------------------------------------

This module defines the abstract base class for all metrics within the SemMatch
statistics pipeline. It establishes the common interface and fundamental
properties that all concrete metric implementations must adhere to.
"""
from abc import ABCMeta, abstractmethod
from typing import List, Type, Union

from semmatch.configs.base import Config
from semmatch.statistics.pipeline_data import AnalysisResult

RAW_RESULT_DEFAULT_KEY = "metric_raw_results"
RESULT_DEFAULT_KEY = "metric_raw_results"


class BaseMetric(metaclass=ABCMeta):
    """
    Abstract base class for all metrics in the SemMatch statistics pipeline.

    This class defines the common interface and fundamental properties that all
    concrete metric implementations must adhere to. It ensures that all metrics
    can be integrated into the `PipelineOrchestrator` and provide consistent
    methods for updating, computing, and retrieving results.

    Attributes
    ----------
    _allowed_result_type : Union[Type[AnalysisResult], List[Type[AnalysisResult]]]
        A class variable that must be defined by subclasses. It specifies the
        `AnalysisResult` type(s) that this metric is designed to process.
        This is used for validation during pipeline initialization.
    _result : float
        The final computed scalar result of the metric. Initialized to 0.0.
    _raw_results : Union[list, dict]
        A container to store raw data or intermediate results accumulated
        during the `update` calls. Can be a list or a dictionary depending
        on the metric's needs. Initialized to an empty list.

    Parameters
    ----------
    config : Config, optional
        Configuration settings for the metric. Defaults to None.
    """
    _allowed_result_type: Union[Type[AnalysisResult],
                                list[Type[AnalysisResult]]] = None

    _result: float = 0.0
    _raw_results: Union[list, dict] = []

    def __init__(self, config: Config = None):
        """
        Initializes the BaseMetric instance.

        Performs a check to ensure `_allowed_result_type` is defined by the subclass.
        """
        if not self._allowed_result_type:
            raise NotImplementedError(
                f"Metric '{self.__class__.__name__}' must define '_allowed_result_type'."
            )

        self._config = config

    @abstractmethod
    def update(self, data: AnalysisResult) -> None:
        """
        Update the metric's internal state with new data.

        This method is called iteratively for each `AnalysisResult` produced
        by an analyzer. Subclasses must implement this to accumulate necessary
        information for later computation.

        Parameters
        ----------
        data : AnalysisResult
            The analysis result object to process. The type of this object
            must be compatible with `_allowed_result_type`.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> None:
        """
        Compute the final metric result.

        This method is called after all `update` calls have been completed.
        Subclasses must implement this to perform the final calculation
        and store the result in `_result`.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the internal state of the metric.

        This method clears any accumulated raw results and resets the computed
        result to its initial state. It is typically called to prepare the
        metric for a new evaluation run.
        """
        self._result = 0.0
        self._raw_results = []

    def get_raw_results(self) -> List:
        """
        Retrieves the accumulated raw results.

        Returns
        -------
        List
            A list or dictionary containing the raw data or intermediate results
            collected during `update` calls.
        """
        return self._raw_results

    def get_result(self) -> float:
        """
        Retrieves the final computed scalar result of the metric.

        Returns
        -------
        float
            The scalar value representing the computed metric.
        """
        return self._result

    @classmethod
    def get_allowed_result_types(cls) -> list[type]:
        """
        Returns a list of allowed `AnalysisResult` types for this metric.

        This method handles both single and list definitions of `_allowed_result_type`.
        """
        return cls._allowed_result_type if isinstance(cls._allowed_result_type, list) else [cls._allowed_result_type]
