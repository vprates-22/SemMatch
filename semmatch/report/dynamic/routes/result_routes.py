"""
Module: report.routes.result_routes

This module defines the ResultRoutes class which sets up Flask routes for displaying
experimental results related to image matching or related computer vision tasks.

The module handles:
- Loading overall experiment results from a JSON file.
- Loading individual experiment/match data from a NumPy .npz file.
- Providing Flask routes to serve web pages for:
    - A summary view of general experiment metrics.
    - A list view of individual experiment results.
    - A detailed view of results for a specific experiment.

The ResultRoutes class is designed to be integrated within a larger Flask application
and provides a Flask Blueprint for easy registration of its routes.
"""

import json
import numpy as np
from typing import Dict, Any, Union

from flask import Blueprint, render_template

from semmatch.configs.result_routes_config import Config, ResultRoutesConfig


class ResultRoutes:
    """
    Flask routes for displaying experimental results and individual match data.

    This class loads overall results from a JSON file and individual match data
    from a NumPy `.npz` file, then serves web pages via Flask blueprints
    to display summary metrics and detailed per-experiment results.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Keys:
        - 'arr_name' : str
            Name of the array in the `.npz` file containing individual results.
            Defaults to 'all_matches'.
        - 'results_file_path' : str
            Path to the JSON file containing general results summary.
        - 'matches_file_path' : str
            Path to the `.npz` file containing individual match data.

    Attributes
    ----------
    blueprint : flask.Blueprint
        Flask blueprint object with routes registered.
    general_results : dict
        Parsed JSON dictionary with overall experiment results.
    individual_data : np.ndarray
        Loaded NumPy array with individual experiment data.
    """

    def __init__(self, config: Union[Config, Dict[str, Any]] = None):
        self.config = ResultRoutesConfig(config)

        if not self.config.results_file_path:
            raise Exception('Missing "results_file_path"')

        self._load_data()

        # Cria o blueprint para o grupo de rotas
        self.blueprint = Blueprint('results', __name__)
        self._routes()

    def _load_data(self) -> None:
        """
        Load the overall results from JSON and individual data from `.npz`.

        Loads and sets:
        - self.general_results (dict)
        - self.individual_data (np.ndarray)
        """
        results_file = self.config.results_file_path
        with open(results_file, 'r') as f:
            self.general_results = json.load(f)

        self.individual_data = np.load(self.config.matches_file_path, allow_pickle=True)[
            self.config.arr_name]

    def _routes(self):
        """
        Define Flask routes:

        - `/` : Shows summary page with general metrics.
        - `/individual_results` : Lists all individual experiment results.
        - `/details/<int:experiment_id>` : Shows detailed results for one experiment.
        """

        @self.blueprint.route('/')
        def index() -> str:
            """
            Render the homepage showing overall experiment metrics.

            Returns
            -------
            Response
                Rendered HTML template with general metrics.
            """
            return render_template('index.html', metrics=self.general_results)

        @self.blueprint.route('/individual_results')
        def individual_results() -> str:
            """
            Render a page listing all individual experiment results.

            Returns
            -------
            Response
                Rendered HTML template with list of individual results.
            """
            # Aqui criamos uma lista de opções a partir dos resultados
            individual_list = self.individual_data
            return render_template('individual_results.html', results=individual_list)

        @self.blueprint.route('/details/<int:experiment_id>')
        def experiment_result(experiment_id: int) -> str:
            """
            Render detailed results for a specific experiment.

            Parameters
            ----------
            experiment_id : int
                Index of the experiment to display.

            Returns
            -------
            Response
                Rendered HTML template with detailed experiment info.
            """
            # Filtra o experimento com base no ID
            experiment = self.individual_data[experiment_id]
            return render_template('details.html', experiment_id=experiment_id, experiment=experiment)
