"""
Module: semmatch.configs.base
-----------------------------

This module defines the `Config` base class, which provides a flexible and
extensible way to manage configuration settings within the SemMatch project.
It supports hierarchical configuration, merging of settings, and dictionary-like
access to configuration parameters.

Classes:
    Config: A base class for managing configuration settings.
"""

import json
from abc import ABCMeta
from types import MappingProxyType
from typing import Union, Dict, Any
from semmatch.utils.misc import combine_dicts


class Config(metaclass=ABCMeta):
    """
    A base class for managing configuration settings.

    This class provides a flexible and extensible way to manage configuration
    settings within the SemMatch project. It supports hierarchical configuration,
    merging of settings, and dictionary-like access to configuration parameters.

    Parameters
    ----------
    config : Union["Config", Dict[str, Any]], optional
        Initial configuration settings. Can be another `Config` object or a
        dictionary. If None, an empty dictionary is used.

    Attributes
    ----------
    _config : Dict[str, Any]
        A class-level dictionary storing default or global configuration settings.
    _config_data : Dict[str, Any]
        An instance-level dictionary storing the merged configuration data.
    _locked : bool
        A flag to prevent direct modification of attributes after initialization.

    """
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
        """
        Returns a read-only proxy of the internal configuration dictionary.

        Returns
        -------
        MappingProxyType
            A read-only view of the configuration data.
        """
        return MappingProxyType(self._config_data)

    def __getattr__(self, attr):
        """
        Provides attribute-style access to configuration parameters.

        If an attribute is not found, it returns None instead of raising
        an AttributeError.

        Parameters
        ----------
        attr : str
            The name of the attribute to retrieve.

        Returns
        -------
        Any or None
            The value of the configuration parameter, or None if not found.
        """
        return None

    def __repr__(self):
        """
        Returns a string representation of the configuration, formatted as JSON.

        Returns
        -------
        str
            A JSON string representation of the configuration data.
        """
        return json.dumps(self._config_data, indent=4, default=str)

    def __setattr__(self, key: str, value: Any):
        """
        Prevents direct modification of attributes after the object is locked.

        Raises
        ------
        AttributeError
            If an attempt is made to modify an attribute after the object is locked.
        """
        if getattr(self, "_locked", False):
            raise AttributeError(
                f"Cannot modify '{key}' directly. Use set_config().")
        super().__setattr__(key, value)

    def __getitem__(self, key):
        """
        Provides dictionary-style access to configuration parameters.

        Parameters
        ----------
        key : str
            The key of the configuration parameter to retrieve.

        Returns
        -------
        Any
            The value associated with the given key.
        """
        return self._config_data[key]

    def __iter__(self):
        """
        Returns an iterator over the keys of the configuration dictionary.
        """
        return iter(self._config_data)

    def __len__(self):
        """
        Returns the number of items in the configuration dictionary.

        Returns
        -------
        int
            The number of configuration parameters.
        """
        return len(self._config_data)

    def __contains__(self, key):
        """
        Checks if a key exists in the configuration dictionary.

        Parameters
        ----------
        key : str
            The key to check.

        Returns
        -------
        bool
            True if the key exists, False otherwise.
        """
        return key in self._config_data

    def keys(self):
        return self._config_data.keys()

    def values(self):
        return self._config_data.values()

    def items(self):
        return self._config_data.items()

    def get(self, key, default=None):
        """
        Retrieves the value for a given key, with an optional default.

        Parameters
        ----------
        key : str
            The key to retrieve.
        default : Any, optional
            The default value to return if the key is not found. Defaults to None.

        Returns
        -------
        Any
            The value associated with the key, or the default value if the key is not found.
        """
        return self._config_data.get(key, default)

    def set_config(self, key: str, value: Any):
        """
        Sets or updates a configuration parameter.

        This method allows controlled mutation of configuration parameters by
        temporarily unlocking the object, updating the internal dictionary and
        attribute, and then re-locking it.

        Parameters
        ----------
        key : str
            The key of the configuration parameter to set.
        value : Any
            The value to assign to the parameter.
        """
        new_dict = dict(self._config_data)
        new_dict[key] = value

        object.__setattr__(self, "_locked", False)
        object.__setattr__(self, "_config_data", new_dict)
        object.__setattr__(self, key, value)
        object.__setattr__(self, "_locked", True)

    @classmethod
    def set_global_config(cls, key: str, value: Any) -> None:
        """
        Sets a global configuration parameter for all instances of this Config class.

        This method modifies the class-level `_config` dictionary, affecting
        default values for all new `Config` instances.

        Parameters
        ----------
        key : str
            The key of the global configuration parameter to set.
        value : Any
            The value to assign to the global parameter.
        """
        cls._config[key] = value

    def merge_config(self, extra: Union[Dict[str, Any], "Config", None]) -> "Config":
        """
        Merges an extra dictionary or a Config object into the current config.
        Updates both _config_data and instance attributes.

        Parameters
        ----------
        extra : Union[Dict[str, Any], "Config", None]
            A dictionary or another `Config` object whose settings will be merged
            into the current configuration. Values from `extra` will override
            existing values.

        Returns
        -------
        Config
            The current `Config` instance, with merged settings.

        Raises
        ------
        TypeError
            If `extra` is not a dictionary or a `Config` instance.
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
