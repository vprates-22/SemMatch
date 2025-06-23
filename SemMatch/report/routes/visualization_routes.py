import os
import io
import cv2
import torch
import matplotlib
import numpy as np
import matplotlib.colors as mcolors

from ultralytics import SAM

from flask import Blueprint, Response, request, send_file
from ...visualization.utils import plot_pair, save
from ...visualization.masks import plot_masks
from ...visualization.matches import plot_matches

from ...settings import MATCHES_PATH, BASE_PATH
from ...utils.io import combine_dicts
from ...visualization.utils import DEFAULT_COLORS

matplotlib.use('Agg')

MODEL_DIR_NAME = 'models'

class VisualizationRoutes:
    default_config = {
        'arr_name': 'all_matches',
        'matches_file_path': '',
        'sam_model': 'sam2.1_l.pt',
    }

    def __init__(self, config:dict):
        """
        Initializes the visualization routes with the given configuration and loads the data.

        Parameters
        ----------
        config : dict
            Configuration dictionary. Expected keys:
            - 'arr_name': str, name of the array in the .npz file. Defaults to 'all_matches'.
            - 'matches_path': str, path to directory containing the matches file.
            - 'matches_file': str, filename of the .npz file containing match data.
        """
        self.config = combine_dicts(self.default_config, config)

        if not self.config['matches_file_path']:
            raise Exception('Missing "matches_file_path"')

        self.data = self._load_data()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam = self._load_sam()

        self.blueprint = Blueprint('visualization', __name__)

        self.last_pair = -1
        self.cache = {}

        self._routes()

    def _load_sam(self):
        """

        """
        dir_path = BASE_PATH / MODEL_DIR_NAME
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / self.config['sam_model']

        sam = SAM(file_path)
        sam.to(self.device).eval()

        return sam

    def _get_object_mask(self, image, points, batch_size = 200) -> np.ndarray:
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

    def _load_data(self):
        """
        Loads the match data from the specified NumPy file.

        Returns
        -------
        np.ndarray
            The match data loaded from the file.
        """        
        return np.load(self.config['matches_file_path'], allow_pickle=True)[self.config['arr_name']]

    def _routes(self):
        """
        Defines the Flask route for visualizations:
        GET /visualization/<int:pair>/?type={point2point|mask2mask|mask2point}&set={all|hits|misses}&point-match=<int>

        Parameters
        ----------
        pair : int
            Index of the pair to visualize.

        Returns
        -------
        flask.Response
            A PNG image as an HTTP response or error message.
        """
        @self.blueprint.route('/visualization/<int:pair>/', methods=['GET'])
        def plot(pair:int):
            plot_type = request.args.get("type", "point2point")
            match_set = request.args.get("set", "all")
            point_match = request.args.get("point-match", "0")

            if pair >= len(self.data):
                return Response("Pair index out of range", status=400)
            
            if self.last_pair != pair:
                self.cache = {}

            data = self.data[pair]

            if plot_type == "mask2mask":
                return self._mask_to_mask(data, point_match)
            elif plot_type == "point2point":
                return self._point_to_point(data, match_set)
            elif plot_type == "mask2point":
                return self._mask_to_point(data, match_set, point_match)
            else:
                return Response(f"Unknown type: {plot_type}", status=400)

    def _prepare_image_response(self) -> Response:
        """
        Prepares a PNG image response from the current matplotlib plot.

        Returns
        -------
        Response
            A Flask Response containing a PNG image.
        """        
        buffer = io.BytesIO()
        save(buffer, format='png')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='image/png')

    def _point_to_point(self, data:dict, match_set:str) -> Response:
        """
        Visualizes point-to-point matches between two images.

        Parameters
        ----------
        data : dict
            Dictionary containing image paths and match data.
        match_set : str
            One of ['all', 'hits', 'misses'] indicating which matches to show.

        Returns
        -------
        Response
            PNG image response of the visualization.
        """
        img0 = cv2.imread(data['image0'])
        img1 = cv2.imread(data['image1'])
        mkpts0, mkpts1 = data['mkpts0'], data['mkpts1']
        inliers = np.array(data['inliers'])

        plot_pair(img0, img1)

        if match_set == 'all':
            plot_matches(mkpts0[~inliers], mkpts1[~inliers], color='r')
            plot_matches(mkpts0[inliers], mkpts1[inliers], color='g')
        elif match_set == 'misses':
            plot_matches(mkpts0[~inliers], mkpts1[~inliers], color='r')
        elif match_set == 'hits':
            plot_matches(mkpts0[inliers], mkpts1[inliers], color='g')

        return self._prepare_image_response()

    def _mask_to_mask(self, data:dict, point_match:str) -> Response:
        """
        Visualizes the masks of a given matched point between two images.

        Parameters
        ----------
        data : dict
            Dictionary containing image paths, masks, and inliers.
        point_match : str
            Index (as string) of the point whose masks are to be visualized.

        Returns
        -------
        Response
            PNG image response of the visualization.
        """
        point_match = int(point_match)

        img0 = cv2.imread(data['image0'])
        img1 = cv2.imread(data['image1'])

        inliers = np.array(data['inliers'])

        mkpts0 = data['mkpts0'][point_match]
        mkpts1 = data['mkpts1'][point_match]

        if self.cache.get(point_match):
            mask0 = self.cache[point_match][0]
            mask1 = self.cache[point_match][1]

        else:
            mask0 = self._get_object_mask(img0, mkpts0[None])[0]
            mask1 = self._get_object_mask(img1, mkpts1[None])[0]

            self.cache[point_match] = {
                0: mask0,
                1: mask1
            }

        plot_pair(img0, img1, title=f"LPIPS Loss: {data['lpips_loss'][point_match]}")

        color = 'r'
        if inliers[point_match]:
            color = 'g'

        plot_masks(mask0, mask1, color=color)

        return self._prepare_image_response()
    
    def _mask_to_point(self, data: dict, match_set: str, point_match: str) -> Response:
        """
        Visualiza como uma máscara na primeira imagem se mapeia para múltiplas regiões de pontos e máscaras
        na segunda imagem, opcionalmente sobrepondo acertos (hits) e erros (misses).

        Parâmetros
        ----------
        data : dict
            Dicionário contendo caminhos de imagens, máscaras, correspondências e inliers.
        match_set : str
            Um dos ['all', 'hits', 'misses'] para filtrar a visualização.
        point_match : str
            Índice (como string) do ponto na primeira imagem a ser usado como base para a máscara.

        Retorna
        -------
        Response
            Resposta PNG da visualização.
        """
        point_match = int(point_match)

        # Carregamento das imagens
        img0 = cv2.imread(data['image0'])
        img1 = cv2.imread(data['image1'])

        mkpts0, mkpts1 = data['mkpts0'], data['mkpts1']
        inliers = np.array(data['inliers'])

        # Cache de máscaras
        if self.cache.get(point_match):
            mask0 = self.cache[point_match][0]
        else:
            mask0 = self._get_object_mask(img0, mkpts0[point_match][None])[0]
            mask1 = self._get_object_mask(img1, mkpts1[point_match][None])[0]
            self.cache[point_match] = {0: mask0, 1: mask1}

        # Índices dos pontos que estão dentro da máscara
        indexes = [i for i, (x, y) in enumerate(mkpts0.astype(int)) if mask0[y, x]]

        # Máscaras adicionais
        indexes_to_search = [index for index in indexes if index not in self.cache]
        if indexes_to_search:
            pts0_to_search = [mkpts0[index] for index in indexes_to_search]
            pts1_to_search = [mkpts1[index] for index in indexes_to_search]

            masks0 = self._get_object_mask(img0, pts0_to_search)
            masks1 = self._get_object_mask(img1, pts1_to_search)

            for index, mask0, mask1 in zip(indexes_to_search, masks0, masks1):
                self.cache[index] = {0: mask0, 1: mask1}

        # Obtenção das máscaras e inliers para os pontos relevantes
        masks1 = np.array([self.cache[index][1] for index in indexes])
        inliers = np.array(inliers[indexes])

        # Filtrando os pontos relevantes
        mkpts0 = mkpts0[indexes]
        mkpts1 = mkpts1[indexes]

        # Garantindo que as máscaras não estão vazias
        if masks1.size > 0:
            # Criando as máscaras de erros (misses) e acertos (hits) para a imagem 1
            masks1_misses = np.logical_or.reduce(masks1[~inliers]) if (~inliers).any() else np.zeros_like(masks1[0])
            masks1_hits = np.logical_or.reduce(masks1[inliers]) if inliers.any() else np.zeros_like(masks1[0])

            # Colorindo as máscaras
            mask1_misses_colored = np.where(masks1_misses[..., None], mcolors.to_rgb(DEFAULT_COLORS['r']), 0)
            mask1_hits_colored = np.where(masks1_hits[..., None], mcolors.to_rgb(DEFAULT_COLORS['g']), 0)

            mask0_colored = np.where(mask0[..., None], mcolors.to_rgb(DEFAULT_COLORS['b']), 0)
            mask1_colored = np.where(mask1_hits_colored != 0, mask1_hits_colored, mask1_misses_colored)
        else:
            # Caso a máscara esteja vazia, crie um array de zeros
            mask0_colored = np.zeros_like(img0)
            mask1_colored = np.zeros_like(img1)

        # Plotando as imagens e máscaras
        plot_pair(img0, img1)

        # Filtrando e visualizando com base no `match_set`
        if match_set == 'all':
            plot_masks(mask0_colored, mask1_colored, color_it=False)
            plot_matches(mkpts0[~inliers], mkpts1[~inliers], color='r')
            plot_matches(mkpts0[inliers], mkpts1[inliers], color='g')
        elif match_set == 'misses':
            plot_masks(mask0_colored, mask1_misses_colored, color_it=False)
            plot_matches(mkpts0[~inliers], mkpts1[~inliers], color='r')
        elif match_set == 'hits':
            plot_masks(mask0_colored, mask1_hits_colored, color_it=False)
            plot_matches(mkpts0[inliers], mkpts1[inliers], color='g')

        # Preparando a resposta
        return self._prepare_image_response()
