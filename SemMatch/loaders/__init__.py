"""
Module Name: dataset
----------------------------

This module is responsible for handling the operations related to datasets.
It includes functionality for downloading them, and loading and reading images, 
as well as managing related data.
"""

import inspect
import importlib.util
from .base_dataset_loader import BaseDatasetLoader


def get_loader(name: str) -> object:
    """
    Retrieve the class responsible for handling a specific dataset.

    Parameters
    ----------
    name : str
        The name of the dataset for which the responsible class will be retrieved.

    Returns
    -------
    type
        The class responsible for extracting and processing the specified dataset.
    """
    paths = [name, f'{__name__}.{name}']

    for path in paths:
        spec = importlib.util.find_spec(path)
        if spec:
            mod = __import__(path, fromlist=[""])

            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, BaseDatasetLoader) and obj is not BaseDatasetLoader:
                    return obj

    raise RuntimeError(
        f'Dataset {name} not found in any of [{" ".join(paths)}]')
