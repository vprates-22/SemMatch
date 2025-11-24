from semmatch.configs.base import Config


class VisualizationRoutesConfig(Config):
    _config = {
        'arr_name': 'all_matches',
        'matches_file_path': '',
        'sam_model': 'sam2.1_l.pt',
        'batch_size': 200
    }
