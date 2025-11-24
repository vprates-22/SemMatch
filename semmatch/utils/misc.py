"""
Module: utils.misc
------------------

This module provides miscellaneous utility functions that do not fit into other specific categories.
It includes functions for dictionary manipulation.
"""

def combine_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Combine two dictionaries, preserving only truthy values and preferring values from dict2.

    If both dictionaries contain a key:
    - Use the value from dict2 if it is truthy.
    - Otherwise, fall back to the value from dict1.

    Parameters
    ----------
    dict1 : dict
        The base dictionary.
    dict2 : dict
        The overriding dictionary.

    Returns
    -------
    dict
        The combined dictionary with only truthy values preserved.
    """
    # Initialize dict1 and dict2 as empty dictionaries if they are None
    dict1 = dict1 or {}
    dict2 = dict2 or {}

    # Get all unique keys from both dictionaries
    keys = set(dict1) | set(dict2)
    result = {}
    for key in keys:
        # Get values for the current key from both dictionaries
        v1 = dict1.get(key)
        v2 = dict2.get(key)
        # Prefer v2 if it's truthy, otherwise use v1
        result[key] = v2 if v2 else v1
        # If the combined value is still falsy, and v2 was not the reason (i.e., key was only in dict1),
        # then explicitly use v1 if it exists, otherwise v2 (which would be falsy or None)
        if not result[key]:
            result[key] = v2 if key in dict2 else v1
    return result
