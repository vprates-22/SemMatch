from semmatch.configs.base import Config


class DatasetConfig(Config):
    _config = {
        'data_path': '',
        'pairs_path': '',
        'cache_images': False,
        'max_pairs': -1,
        'url': '',
        'url_download_extension': '',
    }
