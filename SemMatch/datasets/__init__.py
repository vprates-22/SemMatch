import inspect
import importlib.util

def get_dataset(name):
    from .base_dataset_loader import BaseDatasetLoader

    paths = [name, f'{__name__}.{name}']

    for path in paths:
        spec = importlib.util.find_spec(path)
        if spec:
            mod = __import__(path, fromlist=[""])

            for nome, obj in inspect.getmembers(mod, inspect.isclass):
                if issubclass(obj, BaseDatasetLoader) and obj is not BaseDatasetLoader:
                    return obj

    raise RuntimeError(f'Dataset {name} not found in any of [{" ".join(paths)}]')