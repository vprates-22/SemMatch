"""
Module: semmatch.statistics.data_generators.base
------------------------------------------------

This module defines the base classes for data generators within the SemMatch
statistics pipeline. Data generators are responsible for processing raw input
data into a more structured or derived format suitable for analysis.
"""
from abc import ABCMeta, abstractmethod

from semmatch.configs.base import Config
from semmatch.statistics.pipeline_data import RawDataInput, GeneratedData, KeypointData


class DataGenerator(metaclass=ABCMeta):
    """
    Base class for all data generators.

    Data generators take `RawDataInput` and produce `GeneratedData` objects,
    which are then consumed by `DataAnalyzer` instances.

    Parameters
    ----------
    config : Config, optional
        Configuration object for the generator. Defaults to an empty `Config`
        object if not provided. Subclasses should define their own
        `_default_config` and merge the provided config with it.
    """
    _default_config = Config({})

    def __init__(self, config: Config = None):
        self._config = self._default_config.merge_config(config)

    @abstractmethod
    def generate(self, raw_data: RawDataInput) -> list[GeneratedData]:
        """
        Abstract method to generate derived data from raw input.

        Parameters
        ----------
        raw_data : RawDataInput
            The raw input data for the current iteration.

        Returns
        -------
        list[GeneratedData]
            A list of generated data objects.
        """
        raise NotImplementedError


class KeyPointGenerator(DataGenerator):
    """
    A simple data generator that extracts keypoint data directly from `RawDataInput`.

    This generator is useful when keypoints (`mkpts0`, `mkpts1`) are needed
    as `GeneratedData` for analyzers, without any further processing.

    Parameters
    ----------
    config : Config, optional
        Configuration object for the generator. This generator does not
        require any specific configuration.
    """

    def __init__(self, config=None):
        default_config = Config({})
        super().__init__(default_config.merge_config(config))

    def generate(self, raw_data: RawDataInput) -> list[KeypointData]:
        return [
            KeypointData(
                # mkpts0: NDArray
                #     Keypoints from the first image.
                # mkpts1: NDArray
                #     Keypoints from the second image.
                mkpts0=raw_data.mkpts0,
                mkpts1=raw_data.mkpts1
            )
        ]
