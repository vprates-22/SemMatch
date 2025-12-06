"""
Module: semmatch.datasets
-------------------------

This module is responsible for handling the operations related to datasets.
It includes functionality for downloading them, and loading and reading images, 
as well as managing related data.
"""

import inspect
import importlib.util
from .base import BaseDataset


def get_dataset(name: str) -> type[BaseDataset]:
    """
    Retrieve the class responsible for handling a specific dataset by its name.

    Parameters
    ----------
    name : str
        The name of the dataset for which the responsible class will be retrieved.

    Returns
    -------
    type
        The class (subclass of `BaseDataset`) responsible for extracting and processing the specified dataset.
    """
    paths = [name, f'{__name__}.{name}']

    for path in paths:
        spec = importlib.util.find_spec(path)
        if spec:
            mod = __import__(path, fromlist=[""])

            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, BaseDataset) and obj is not BaseDataset:
                    return obj

    raise RuntimeError(
        f'Dataset {name} not found in any of [{" ".join(paths)}]')
