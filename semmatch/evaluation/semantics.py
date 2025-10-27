"""
Module: semantics
----------------------

This module implements the `SemanticEval` class for evaluating semantic segmentation models
and performing matching tasks between image pairs. It integrates multiple tools and models
such as SAM (Segment Anything Model) and LPIPS (Learned Perceptual Image Patch Similarity)
to generate object segmentation masks, compute similarity metrics, and extract matches from
image pairs.

Classes:
    SemanticEval:
        A class that encapsulates the process of evaluating semantic segmentation on image pairs.
        It handles image preprocessing, mask prediction, matching, and result extraction. It
        also computes LPIPS loss for object similarity between corresponding image regions.
"""
import multiprocessing as mp
from typing import List
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor

from semmatch.utils.evaluation import project_points_between_cameras

from semmatch.statistics.orchestrator import MetricsOrchestrator

from semmatch.loaders import get_loader
from semmatch.helpers import to_cv, get_inliers
from semmatch.utils.sam import load_sam, get_object_mask
from semmatch.utils.lpips import load_lpips, get_obj_similarities
from semmatch.utils.io import combine_dicts, resize_long_edge
from semmatch.loaders.base_dataset_loader import DATASET_CONFIG_KEYS
from semmatch.settings import BASE_PATH, RESULTS_PATH, MATCHES_PATH, VISUALIZATIONS_PATH

MODEL_DIR_NAME = 'models'


class SemanticEval():
    """
    Module: semantics
    -----------------

    This module implements the `SemanticEval` class for evaluating semantic matching
    and segmentation quality across image pairs using learned similarity and segmentation models.
    It is designed for tasks that require understanding the semantic consistency between
    corresponding regions in different images, particularly in 3D scene datasets.

    Core Features:
    --------------
    - Integrates SAM (Segment Anything Model) for generating object-level masks from points.
    - Uses LPIPS (Learned Perceptual Image Patch Similarity) to compute perceptual similarity
    between cropped object regions.
    - Supports evaluation of correspondence and inlier quality using pose and depth data.
    - Supports optional image resizing and multiprocessing.
    - Caches results to disk and optionally reloads from saved files.

    Classes:
    --------
    - SemanticEval:
        A configurable evaluation class that supports object segmentation, LPIPS-based
        region comparison, geometric inlier estimation, and results saving.

    Configuration Keys:
    -------------------
    - dataset (str): Name of the dataset loader to use.
    - pairs_path (str): Path to image pairs metadata.
    - cache_images (bool): Whether to cache all images in memory.
    - pose_thresholds, ransac_thresholds (List[float]): Thresholds for geometric evaluation.
    - n_workers (int): Number of worker processes for matching.
    - sam_model (str): File name of the SAM model to use.
    - lpips_net (str): Backbone for LPIPS (e.g., 'alex', 'vgg').

    Saved Outputs:
    --------------
    - Matches: Saved as a compressed `.npz` file.
    - Evaluation Summary: Saved as a `.json` file with hit/miss ratios and LPIPS statistics.

    """
    default_config = {
        'metrics': [],
        'report': None,
        'dataset': 'scannet',
        'data_path': '',
        'pairs_path': '',
        'pose_estimator': 'poselib',
        'cache_images': False,
        'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        'pose_thresholds': [5, 10, 20],
        'max_pairs': -1,
        'results_path': RESULTS_PATH,
        'matches_path': MATCHES_PATH,
        'visualization_path': VISUALIZATIONS_PATH,
        'n_workers': 8,
        'resize': None,
        'detector_only': False,
        'sam_model': 'sam2.1_l.pt',
        'lpips_net': 'alex',
        'url': '',
        'url_download_extension': ''
    }

    def __init__(self, config: dict = {}):
        self.config = combine_dicts(self.default_config, config)

        if not self.config['metrics']:
            raise Exception("Missing `metrics` param")
        self.metric_orchestrator = MetricsOrchestrator(self.config['metrics'])

        if not self.config['report']:
            raise Exception("Missing `report` param")
        self.report = self.config['report'](self.metric_orchestrator)

        if self.config['n_workers'] == -1:
            self.config['n_workers'] = mp.cpu_count()

        loaders_config = {k: v for k,
                          v in self.config.items() if k in DATASET_CONFIG_KEYS}
        self.dataset = get_loader(self.config['dataset'])(loaders_config)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.sam = load_sam(self.config['sam_model'], self.device)
        self.lpips = load_lpips(self.config['lpips_net'], self.device)

    def generate_matches(self, matcher_fn):
        """
        Generates matches and related evaluation metrics for all image pairs in the dataset.

        For each pair:
        - Reads and optionally resizes images.
        - Extracts keypoint matches using the provided matcher function.
        - Computes inliers via geometric consistency.
        - Projects keypoints between views and validates via depth maps.
        - Computes object segmentation masks and mask hit statistics.
        - Computes LPIPS similarity scores for matched objects.

        Parameters
        ----------
        matcher_fn : callable
            A function that takes two images and returns matched keypoints in both.

        Yields
        ------
        dict
            Dictionary containing:
            - 'image0', 'image1': Image file paths.
            - 'mkpts0', 'mkpts1': Matched keypoints arrays.
            - 'inliers': Boolean array indicating inlier matches.
            - 'mask_hits': Boolean array indicating mask-based hit validity.
            - 'lpips_loss': List of LPIPS similarity scores per matched object.
        """
        for i, pair in tqdm(enumerate(self.dataset.pairs), desc='Extraindo correspondências'):
            image0 = self.dataset.read_image(pair['image0'])
            image1 = self.dataset.read_image(pair['image1'])

            scale_img0 = scale_img1 = 1.0
            if self.config['resize'] is not None:
                image0, scale_img0 = resize_long_edge(image0, self.config['resize'])
                image1, scale_img1 = resize_long_edge(image1, self.config['resize'])

            mkpts0, mkpts1 = matcher_fn(image0, image1)
            if isinstance(mkpts0, torch.Tensor):
                mkpts0 = mkpts0.cpu().numpy()
                mkpts1 = mkpts1.cpu().numpy()

            if self.config['resize'] is not None:
                mkpts0 = mkpts0 / scale_img0
                mkpts1 = mkpts1 / scale_img1

            inliers = self.dataset.get_inliers(mkpts0, mkpts1, i)

            image0 = to_cv(image0)
            image1 = to_cv(image1)

            reprojected_pts = []
            valid_flags = []
            for pt0 in mkpts0:
                mapped_pt, valid = self.dataset.map_point(
                    tuple(pt0),
                    pair_index=i,
                    scale_img0=scale_img0,
                    scale_img1=scale_img1
                )
                reprojected_pts.append(mapped_pt)
                valid_flags.append(valid)

            real_mkpts_0_on_1 = np.array(reprojected_pts, dtype=np.float32)
            valid_projections = np.array(valid_flags, dtype=bool)

            masks0 = get_object_mask(self.sam, image0, mkpts0)
            masks1 = get_object_mask(self.sam, image1, mkpts1)

            mask_hits = np.array([
                mask[int(y), int(x)] if valid and not np.isnan(x) and not np.isnan(y) else False
                for (x, y), valid, mask in zip(real_mkpts_0_on_1, valid_projections, masks1)
            ], dtype=bool)

            lpips_similarity = [
                get_obj_similarities(self.lpips, image0, image1, mask0, mask1, self.device)
                for mask0, mask1 in zip(masks0, masks1)
            ]


            yield {
                'image0': pair['image0'],
                'image1': pair['image1'],
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
                'inliers': inliers,
                'mask_hits': mask_hits,
                'lpips_loss': lpips_similarity
            }

    def extract_and_save_matches(self, matcher_fn, name='', force=False):
        """
        Extracts matches using the specified matcher function and saves the results and statistics to disk.

        If matches already exist and `force` is False, loads and returns the cached results.
        Otherwise, processes all pairs, computes metrics, saves match data (.npz), and summary statistics (.json).

        Parameters
        ----------
        matcher_fn : callable
            Matching function to generate keypoints and matches between image pairs.
        name : str, optional
            Optional name for saved files. Defaults to the matcher function name.
        force : bool, optional
            If True, forces recomputation and overwriting of existing results.

        Returns
        -------
        Tuple[pathlib.Path, pathlib.Path]
            Paths to the saved matches file (.npz) and results summary file (.json).
        """
        if name == '':
            name = matcher_fn.__name__

        matches_path = Path(self.config['matches_path'])

        fname = matches_path / f'{name}_matches.npz'

        if not matches_path.exists():
            matches_path.mkdir(parents=True, exist_ok=True)

        results_path = Path(self.config['results_path'])

        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        rname = results_path / f'{name}_results.json'

        # Se o arquivo já existir e 'force' não for True, retorne o arquivo existente
        if not force and fname.exists():
            return np.load(fname, allow_pickle=True)['all_matches']

        matches = []

        for match_data in self.generate_matches(matcher_fn):
            matches.append(match_data)

        np.savez_compressed(fname, all_matches=matches)

        return fname, rname
