import json
from abc import ABCMeta
from types import MappingProxyType
from typing import Union, Dict, Any
from semmatch.utils.misc import combine_dicts


class Config(metaclass=ABCMeta):
    _config: Dict[str, Any] = {}

    def __init__(self, config: Union["Config", Dict[str, Any]] = None):
        object.__setattr__(self, "_locked", False)

        if isinstance(config, Config):
            config = dict(config.config)

        combined = combine_dicts(self._config, config)
        object.__setattr__(self, "_config_data", combined)

        for key, value in combined.items():
            if type(value) == dict:
                value = Config(value)
            try:
                object.__setattr__(self, key, value)
            except TypeError:
                object.__setattr__(self, f"_{key}", value)

        object.__setattr__(self, "_locked", True)

    @property
    def config(self) -> MappingProxyType:
        return MappingProxyType(self._config_data)

    def __getattr__(self, attr):
        return None

    def __repr__(self):
        return json.dumps(self._config_data, indent=4, default=str)

    def __setattr__(self, key: str, value: Any):
        if getattr(self, "_locked", False):
            raise AttributeError(
                f"Cannot modify '{key}' directly. Use set_config().")
        super().__setattr__(key, value)

    # Dict-like compatibility
    def __getitem__(self, key):
        return self._config_data[key]

    def __iter__(self):
        return iter(self._config_data)

    def __len__(self):
        return len(self._config_data)

    def __contains__(self, key):
        return key in self._config_data

    def keys(self):
        return self._config_data.keys()

    def values(self):
        return self._config_data.values()

    def items(self):
        return self._config_data.items()

    def get(self, key, default=None):
        return self._config_data.get(key, default)

    # Controlled mutation
    def set_config(self, key: str, value: Any):
        new_dict = dict(self._config_data)
        new_dict[key] = value

        object.__setattr__(self, "_locked", False)
        object.__setattr__(self, "_config_data", new_dict)
        object.__setattr__(self, key, value)
        object.__setattr__(self, "_locked", True)

    @classmethod
    def set_global_config(cls, key: str, value: Any):
        cls._config[key] = value

    def merge_config(self, extra: Union[Dict[str, Any], "Config", None]) -> "Config":
        """
        Merges an extra dictionary or a Config object into the current config.
        Updates both _config_data and instance attributes.
        """
        if not extra:
            return self

        if isinstance(extra, Config):
            extra = extra._config_data
        elif not isinstance(extra, dict):
            raise TypeError(
                "merge_config() requires a dict or a Config instance.")

        # Merge the dictionaries
        merged = combine_dicts(self._config_data, extra)

        # Unlock to update attributes
        object.__setattr__(self, "_locked", False)

        # Update the internal _config_data
        object.__setattr__(self, "_config_data", merged)

        # Update the instance attributes
        for key, value in merged.items():
            object.__setattr__(self, key, value)

        # Lock the config back
        object.__setattr__(self, "_locked", True)

        return self
