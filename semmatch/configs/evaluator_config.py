from semmatch.configs.base import Config
from semmatch.configs.dataset_config import DatasetConfig


class EvaluatorConfig(Config):
    _config = {
        'metrics': [],
        'dataset': 'scannet',
        'n_workers': 8,
        'metrics_config': {
            'mask_model': 'sam',
            'mask_batch_size': 200,
            'sam_model': 'sam2.1_l.pt',
            'lpips_net': 'alex',
            'inlier_threshold': 6.0,
            'projection_threshold': 6.0,
            'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            'pose_thresholds': [5, 10, 20],
        },
        'dataset_config': DatasetConfig._config.copy(),
        'resize': None,
        'results_path': RESULTS_PATH,
        'matches_path': MATCHES_PATH,
        'visualization_path': VISUALIZATIONS_PATH,
        # 'report': None,
        # 'detector_only': False,
    }
