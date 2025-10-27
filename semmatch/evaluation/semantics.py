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
import json
import multiprocessing as mp
from typing import List
from pathlib import Path

import lpips
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor
from ultralytics import SAM

from semmatch.utils.evaluation import project_points_between_cameras

from semmatch.loaders import get_loader
from semmatch.helpers import to_cv, get_inliers, to_tensor
from semmatch.utils.cropping import crop_square_around_mask
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

        if self.config['n_workers'] == -1:
            self.config['n_workers'] = mp.cpu_count()

        dataset_config = {k: v for k,
                          v in self.config.items() if k in DATASET_CONFIG_KEYS}
        self.dataset = get_loader(self.config['dataset'])(dataset_config)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.sam = self.load_sam()
        self.lpips = self.load_lpips()

    def load_sam(self):
        """
        Loads the Segment Anything Model (SAM) from a specified model file.

        The SAM model is loaded from a local directory defined by the BASE_PATH and MODEL_DIR_NAME.
        The model is then moved to the configured device (GPU or CPU) and set to evaluation mode.

        Returns
        -------
        SAM
            An instance of the SAM model ready for inference.
        """

        dir_path = BASE_PATH / MODEL_DIR_NAME
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / self.config['sam_model']

        sam = SAM(file_path)
        sam.to(self.device).eval()

        return sam

    def load_lpips(self):
        """
        Initializes the LPIPS (Learned Perceptual Image Patch Similarity) model.

        Uses the network architecture specified in the configuration (e.g., 'alex', 'vgg').
        The model is moved to the configured device and set up for similarity evaluation.

        Returns
        -------
        lpips.LPIPS
            An instance of the LPIPS model ready for inference.
        """
        fn = lpips.LPIPS(net=self.config['lpips_net'], verbose=False)
        return fn.to(self.device)

    def get_object_mask(self,
                        image: List[List[List[int]]],
                        points: List[List[List[int]]],
                        batch_size: int = 200) -> Tensor:
        """
        Predicts segmentation masks for an image given a list of prompt points in batches.

        This method uses SAM to generate binary object masks corresponding to the provided points.
        Supports batch prediction to efficiently handle large numbers of prompt points.

        Parameters
        ----------
        image : List[List[List[int]]]
            The input image as a 3D list (height x width x RGB channels).
        points : List[List[int]]
            A list of 2D coordinates [x, y] representing prompt points for segmentation.
        batch_size : int, optional
            Number of points to process per batch. If -1, processes all points in one batch.

        Returns
        -------
        np.ndarray (bool)
            Array of binary masks generated for the prompted points.
        """
        masks = []

        if batch_size == -1:
            # Se batch_size for -1, processa todos os pontos de uma vez
            batch_size = len(points)

        # Dividir os pontos em lotes (batches)
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]

            # Limpar a memória da GPU antes de cada previsão em batch
            torch.cuda.empty_cache()

            with torch.no_grad():
                # Realiza a previsão para o batch de pontos na mesma imagem
                results = self.sam.predict(
                    image, points=batch_points, verbose=False)

                # Adicionar as máscaras para cada batch de pontos
                masks.extend(results[0].masks.data.cpu().numpy())

        return np.array(masks, dtype=bool)

    def get_obj_similarities(self,
                             img0,
                             img1,
                             mask0,
                             mask1
                             ) -> float:
        """
        Computes the LPIPS perceptual similarity between two cropped image regions defined by masks.

        Crops square regions around each mask on the corresponding image, converts them to tensors,
        and computes the LPIPS distance indicating perceptual similarity.

        Parameters
        ----------
        img0 : np.ndarray
            The first image array.
        img1 : np.ndarray
            The second image array.
        mask0 : np.ndarray
            Binary mask defining the object region in the first image.
        mask1 : np.ndarray
            Binary mask defining the object region in the second image.

        Returns
        -------
        float
            The LPIPS perceptual similarity score between the two cropped object regions.
        """
        with torch.no_grad():
            cropped_img0 = to_tensor(crop_square_around_mask(img0, mask0))
            cropped_img1 = to_tensor(crop_square_around_mask(img1, mask1))

            return self.lpips(
                cropped_img0.to(self.device),
                cropped_img1.to(self.device)
            ).item()

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
        for pair in tqdm(self.dataset.pairs, desc='Extraindo correspondências'):
            image0 = self.dataset.read_image(pair['image0'])
            image1 = self.dataset.read_image(pair['image1'])

            if self.config['resize'] is not None:
                image0, scale = resize_long_edge(image0, self.config['resize'])
                image1, scale = resize_long_edge(image1, self.config['resize'])

            mkpts0, mkpts1 = matcher_fn(image0, image1)
            if isinstance(mkpts0, torch.Tensor):
                mkpts0 = mkpts0.cpu().numpy()
                mkpts1 = mkpts1.cpu().numpy()

            if self.config['resize'] is not None:
                mkpts0 = mkpts0 / scale
                mkpts1 = mkpts1 / scale

            inliers = get_inliers(mkpts0, mkpts1, pair['K0'], pair['K1'])

            image0 = to_cv(image0)
            image1 = to_cv(image1)
            depth0 = self.dataset.read_image(pair['depth0'], 'depth')
            depth1 = self.dataset.read_image(pair['depth1'], 'depth')

            real_mkpts_0_on_1, valid_projections =\
                project_points_between_cameras(
                    mkpts0,
                    depth0, depth1,
                    pair['K0'], pair['K1'],
                    pair['T_0to1'],
                    image0.shape[:2]
                )

            masks0 = self.get_object_mask(image0, mkpts0)
            masks1 = self.get_object_mask(image1, mkpts1)

            mask_hits = np.array([mask[int(y), int(x)] if valid else np.False_
                                  for (x, y), valid, mask in
                                  zip(real_mkpts_0_on_1, valid_projections, masks1)], dtype=bool)

            lpips_similarity = [
                self.get_obj_similarities(image0, image1, mask0, mask1)
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

        lpips_loss = []
        total_points = 0
        points_hits = 0
        masks_hits = 0
        masks_misses = 0
        misses = 0

        for match_data in self.generate_matches(matcher_fn):
            total_points += len(match_data['mkpts0'])
            points_hits += sum(match_data['inliers'])
            masks_hits += sum(match_data['mask_hits'])
            misses += sum(~match_data['inliers'])
            masks_misses += sum(~(match_data['mask_hits']
                                [match_data['inliers']]))

            lpips_loss.extend(match_data['lpips_loss'])
            matches.append(match_data)

        result = {
            'points_hit_ratio': points_hits / total_points,  # Percentual de acertos em pontos
            # Percentual de erros em pontos
            'points_miss_ratio': 1 - points_hits / total_points,
            'mask_hit_ratio': masks_hits / total_points,  # Percentual de acertos em máscaras
            # Percentual de erros em máscaras
            'mask_miss_ratio': 1 - masks_hits / total_points,
            # Percentual de erros em máscaras por total de misses
            'mask_miss_ratio_per_miss': masks_misses / misses,
            # Percentual de erros em pontos por total de misses
            'points_miss_ratio_per_miss': 1 - masks_misses / misses,
            'average_lpips_loss': np.mean(lpips_loss),  # Perda média LPIPS
        }

        np.savez_compressed(fname, all_matches=matches)
        json.dump(result, rname.open('w'), indent=2)

        return fname, rname
