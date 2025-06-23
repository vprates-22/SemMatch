import os
import cv2
import h5py
import json
import lpips
import torch
import pickle
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from typing import List
from pathlib import Path
from torch import Tensor
from ultralytics import SAM

from .utils import find_matching_points

from ..datasets import get_dataset
from ..helpers import to_cv, get_inliers, to_tensor
from ..utils.cropping import crop_square_around_mask
from ..utils.io import combine_dicts, resize_long_edge
from ..datasets.base_dataset_loader import DATASET_CONFIG_KEYS
from ..settings import BASE_PATH, RESULTS_PATH, MATCHES_PATH, VISUALIZATIONS_PATH

MODEL_DIR_NAME = 'models'

class SemanticEval():
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

    def __init__(self, config:dict = {}):
        self.config = combine_dicts(self.default_config, config)

        if self.config['n_workers'] == -1:
            self.config['n_workers'] = mp.cpu_count()

        dataset_config = {k: v for k, v in self.config.items() if k in DATASET_CONFIG_KEYS}
        self.dataset = get_dataset(self.config['dataset'])(dataset_config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sam = self.load_sam()
        self.lpips = self.load_lpips()

    def load_sam(self):
        """

        """
        dir_path = BASE_PATH / MODEL_DIR_NAME
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / self.config['sam_model']

        sam = SAM(file_path)
        sam.to(self.device).eval()

        return sam

    def load_lpips(self):
        """
        
        """
        fn = lpips.LPIPS(net=self.config['lpips_net'], verbose=False)
        return fn.to(self.device)

    def get_object_mask(self, image:List[List[List[int]]], points:List[List[List[int]]], batch_size:int = 200) -> Tensor:
        """
        Realiza a previsão em batches para uma única imagem, com diferentes pontos de prompt.

        Parameters:
        ----------
        image : List[List[List[int]]]
            A imagem representada como uma lista 3D (altura x largura x canais RGB).
        points : List[List[int]]
            Lista de 2D coordenadas [x, y] representando os pontos de prompt para a segmentação.
        batch_size : int
            O tamanho do batch. Se `-1`, usa o número total de pontos.

        Returns:
        -------
        masks : List[np.ndarray]
            Lista de máscaras binárias para os objetos segmentados na imagem.
        """
        masks = []

        if batch_size == -1:
            batch_size = len(points)  # Se batch_size for -1, processa todos os pontos de uma vez

        # Dividir os pontos em lotes (batches)
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]

            # Limpar a memória da GPU antes de cada previsão em batch
            torch.cuda.empty_cache()

            with torch.no_grad():
                # Realiza a previsão para o batch de pontos na mesma imagem
                results = self.sam.predict(image, points=batch_points, verbose=False)
                
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
        
        """
        with torch.no_grad():
            cropped_img0 = to_tensor(crop_square_around_mask(img0, mask0))
            cropped_img1 = to_tensor(crop_square_around_mask(img1, mask1))

            return self.lpips(
                cropped_img0.to(self.device), 
                cropped_img1.to(self.device)
            ).item()

    def generate_matches(self, matcher_fn):
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
                find_matching_points(
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
            masks_misses += sum(~(match_data['mask_hits'][match_data['inliers']]))

            lpips_loss.extend(match_data['lpips_loss'])
            matches.append(match_data)

        result = {
            'points_hit_ratio': points_hits / total_points,  # Percentual de acertos em pontos
            'points_miss_ratio': 1 - points_hits / total_points,  # Percentual de erros em pontos
            'mask_hit_ratio': masks_hits / total_points,  # Percentual de acertos em máscaras
            'mask_miss_ratio': 1 - masks_hits / total_points,  # Percentual de erros em máscaras
            'mask_miss_ratio_per_miss': masks_misses / misses,  # Percentual de erros em máscaras por total de misses
            'points_miss_ratio_per_miss': 1 - masks_misses / misses,  # Percentual de erros em pontos por total de misses
            'average_lpips_loss': np.mean(lpips_loss),  # Perda média LPIPS
        }

        np.savez_compressed(fname, all_matches=matches)
        json.dump(result, rname.open('w'), indent=2)

        return fname, rname
