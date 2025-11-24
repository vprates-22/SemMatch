from abc import ABCMeta, abstractmethod

from semmatch.configs.base import Config
from semmatch.statistics.pipeline_data import RawDataInput, GeneratedData, KeypointData


class DataGenerator(metaclass=ABCMeta):
    """
    Classe base para geradores de dados.
    """
    _default_config = Config({})

    def __init__(self, config: Config | None = None):
        self._config = self._default_config.merge_config(config)

    @abstractmethod
    def generate(self, raw_data: RawDataInput) -> list[GeneratedData]:
        raise NotImplementedError


class KeyPointGenerator(DataGenerator):
    def __init__(self, config=None):
        default_config = Config({})
        super().__init__(default_config.merge_config(config))

    def generate(self, raw_data: RawDataInput) -> list[KeypointData]:
        return [
            KeypointData(
                mkpts0=raw_data.mkpts0,
                mkpts1=raw_data.mkpts1
            )
        ]
