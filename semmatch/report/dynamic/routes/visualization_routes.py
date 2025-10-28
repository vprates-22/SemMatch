"""
Module: report.routes.visualization_routes
----------------------------

This module provides a Flask blueprint that exposes visualizations of keypoint matches
and segmentation masks between image pairs. It supports:

- Point-to-point match visualization with inlier classification.
- Visualization of SAM-predicted segmentation masks.
- Cross-image segmentation propagation (from a point in one image to regions in the second).

The visualizations are accessible via HTTP routes (see `_routes`), and are served as PNG images.
"""

import io
from collections.abc import Iterable
from typing import Dict, Any, Tuple, Iterable, Union, List

import cv2
import numpy as np
import matplotlib
import matplotlib.colors as mcolors

from numpy.typing import NDArray
from flask import Blueprint, Response, request, send_file
from semmatch.visualization.masks import plot_masks
from semmatch.visualization.matches import plot_matches

from semmatch.utils.io import combine_dicts
from semmatch.utils.sam import load_sam, get_object_mask
from semmatch.utils.visualization import plot_pair, save, DEFAULT_COLORS

matplotlib.use('Agg')


class VisualizationRoutes:
    """
    Flask route handler for visualizing keypoint correspondences and segmentation masks.

    This class wraps functionality to:
    - Load and cache keypoint match data (.npz format).
    - Use SAM to segment regions around matched keypoints.
    - Generate visualizations combining matches and masks.
    - Serve visualizations over HTTP via a Flask blueprint.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Keys include:
        - 'arr_name' : str
            Name of the array in the .npz file (default: 'all_matches').
        - 'matches_file_path' : str
            Path to the .npz file containing match data.
        - 'sam_model' : str
            Path to SAM model weights (default: 'sam2.1_l.pt').
    """
    default_config = {
        'arr_name': 'all_matches',
        'matches_file_path': '',
        'sam_model': 'sam2.1_l.pt',
        'batch_size': 200
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = combine_dicts(self.default_config, config)

        if not self.config['matches_file_path']:
            raise Exception('Missing "matches_file_path"')

        self.data = self._load_data()
        self.sam = load_sam(self.config['sam_model'])
        self.blueprint = Blueprint('visualization', __name__)

        self.last_pair = -1
        self.cache = {}

        self._routes()

    def _load_data(self):
        return np.load(self.config['matches_file_path'], allow_pickle=True)[self.config['arr_name']]

    def _routes(self):
        """
        Registers the main `/visualization/<int:pair>/` Flask route.

        Available query parameters:
        ----------------------------
        - type : str
            Type of visualization. One of:
                - 'point2point' (default)
                - 'mask2mask'
                - 'mask2point'
        - set : str
            One of ['all', 'hits', 'misses']. Filters inliers/outliers.
        - point-match : int
            Index of the point used for mask-based visualizations.

        Returns
        -------
        flask.Response
            PNG image response with the rendered visualization.
        """
        @self.blueprint.route('/visualization/<int:pair>/', methods=['GET'])
        def plot(pair: int):
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

            if plot_type == "point2point":
                return self._point_to_point(data, match_set)

            if plot_type == "mask2point":
                return self._mask_to_point(data, match_set, point_match)

            return Response(f"Unknown type: {plot_type}", status=400)

    def _prepare_image_response(self) -> Response:
        """
        Converts the current matplotlib figure to a PNG image for HTTP response.

        Returns
        -------
        flask.Response
            Image served as 'image/png'.
        """
        buffer = io.BytesIO()
        save(buffer, format='png')
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png')

    def _load_images_and_keypoints(self, data: Dict[str, Any], point_match: Union[int, None] = None):
        """

        """
        img0 = cv2.imread(data['image0'])
        img1 = cv2.imread(data['image1'])
        inliers = np.array(data['inliers'])

        if point_match is not None:
            point_match = int(point_match)
            mkpts0 = np.array(data['mkpts0'][point_match])
            mkpts1 = np.array(data['mkpts1'][point_match])
        else:
            mkpts0 = np.array(data['mkpts0'])
            mkpts1 = np.array(data['mkpts1'])

        return img0, img1, mkpts0, mkpts1, inliers

    def _cache_masks(
        self,
        points_match: Iterable[int],
        imgs: Tuple[NDArray, NDArray],
        mkpts: Tuple[NDArray, NDArray]
    ) -> None:
        img0, img1 = imgs
        mkpts0, mkpts1 = mkpts

        not_cached = [i for i in points_match if not i in self.cache]

        if not not_cached:
            return

        if len(points_match) != len(mkpts0) or len(points_match) != len(mkpts1):
            raise Exception("...")

        masks0 = get_object_mask(
            self.sam, img0, mkpts0, self.config['batch_size'])
        masks1 = get_object_mask(
            self.sam, img1, mkpts1, self.config['batch_size'])

        for idx, mask0, mask1 in zip(not_cached, masks0, masks1):
            self.cache[idx] = {0: mask0, 1: mask1}

    def _get_masks_for_indices(
        self,
        indices: List[int],
        imgs: Tuple[NDArray, NDArray],
        mkpts: Tuple[NDArray, NDArray]
    ) -> List[Tuple[NDArray, NDArray]]:
        self._cache_masks(indices, imgs, mkpts)
        return [(self.cache[i][0], self.cache[i][1]) for i in indices]

    def _filter_points_inside_mask(self, mkpts: NDArray, mask: NDArray) -> List[int]:
        return [i for i, (x, y) in enumerate(mkpts.astype(int)) if mask[y, x]]

    def _color_mask(self, mask: NDArray, color: str) -> NDArray:
        rgb = mcolors.to_rgb(DEFAULT_COLORS[color])
        return np.where(mask[..., None], rgb, 0)

    def _plot_matches_by_inliers(self, mkpts0, mkpts1, inliers, match_set):
        if match_set in ('all', 'hits'):
            plot_matches(mkpts0[inliers], mkpts1[inliers], color='g')
        if match_set in ('all', 'misses'):
            plot_matches(mkpts0[~inliers], mkpts1[~inliers], color='r')

    def _point_to_point(self, data: Dict[str, Any], match_set: str) -> Response:
        """
        Renders a point-to-point keypoint match visualization between two images.

        Parameters
        ----------
        data : dict
            Dictionary with image paths and keypoint data.
        match_set : str
            One of ['all', 'hits', 'misses'].

        Returns
        -------
        flask.Response
            PNG image with the keypoint matches visualized.
        """
        img0, img1, mkpts0, mkpts1, inliers =\
            self._load_images_and_keypoints(data)

        plot_pair(img0, img1)

        self._plot_matches_by_inliers(mkpts0, mkpts1, inliers, match_set)

        return self._prepare_image_response()

    def _mask_to_mask(self, data: Dict[str, Any], point_match: str) -> Response:
        """
        Visualizes SAM-generated masks for a specific matched keypoint pair.

        Parameters
        ----------
        data : dict
            Contains 'image0', 'image1', 'mkpts0', 'mkpts1', and 'inliers'.
        point_match : str
            Index (as string) of the match to visualize.

        Returns
        -------
        flask.Response
            PNG image with both masks overlaid.
        """
        point_match = int(point_match)
        
        img0, img1, mkpts0, mkpts1, inliers =\
            self._load_images_and_keypoints(data, point_match)

        mask0, mask1 = self._get_masks_for_indices(
            [point_match], (img0, img1), (mkpts0[None], mkpts1[None])
        )[0]

        plot_pair(
            img0, img1, title=f"LPIPS Loss: {data['lpips_loss'][point_match]}")

        color = 'r' if inliers[point_match] else 'g'

        plot_masks(mask0, mask1, color=color)

        return self._prepare_image_response()

    def _mask_to_point(self, data: Dict[str, Any], match_set: str, point_match: str) -> Response:
        """
        Visualizes how a mask in image 0 maps to multiple keypoints and masks in image 1.

        This is useful for understanding how segmentation corresponds across matched points.

        Parameters
        ----------
        data : dict
            Dictionary with image paths, matches, and inliers.
        match_set : str
            One of ['all', 'hits', 'misses'].
        point_match : str
            Index (as string) of the point in image 0 to use for the initial mask.

        Returns
        -------
        flask.Response
            PNG visualization of the propagated mask and matched keypoints.
        """
        point_match = int(point_match)

        # Carregamento das imagens
        img0, img1, mkpts0, mkpts1, inliers =\
            self._load_images_and_keypoints(data)

        mask0, _ = self._get_masks_for_indices(
            [point_match], (img0, img1), (mkpts0[None], mkpts1[None]))[0]

        inside_indices = self._filter_points_inside_mask(mkpts0, mask0)
        if not inside_indices:
            return Response("No points inside the selected mask.", status=400)

        pts0 = mkpts0[inside_indices]
        pts1 = mkpts1[inside_indices]
        inliers_subset = inliers[inside_indices]

        masks = self._get_masks_for_indices(
            inside_indices, (img0, img1), (pts0, pts1))

        # Obtenção das máscaras e inliers para os pontos relevantes
        masks1 = np.array([m[1] for m in masks])

        mask0_colored = self._color_mask(mask0, 'b')

        mask1_hits = np.logical_or.reduce(
            masks1[inliers_subset]) if inliers_subset.any() else np.zeros_like(mask0)
        mask1_misses = np.logical_or.reduce(masks1[~inliers_subset]) if (
            ~inliers_subset).any() else np.zeros_like(mask0)

        mask1_colored = np.where(
            mask1_hits[..., None] != 0,
            self._color_mask(mask1_hits, 'g'),
            self._color_mask(mask1_misses, 'r')
        )

        # Plotando as imagens e máscaras
        plot_pair(img0, img1)

        # Filtrando e visualizando com base no `match_set`
        if match_set == 'hits':
            plot_masks(mask0_colored, self._color_mask(
                mask1_hits, 'g'), color_it=False)
        elif match_set == 'misses':
            plot_masks(mask0_colored, self._color_mask(
                mask1_misses, 'r'), color_it=False)
        else:
            plot_masks(mask0_colored, mask1_colored, color_it=False)

        self._plot_matches_by_inliers(pts0, pts1, inliers_subset, match_set)
        return self._prepare_image_response()
