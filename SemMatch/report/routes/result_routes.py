import os
import json
import numpy as np
from flask import Blueprint, render_template

class ResultRoutes:
    default_config = {
        'arr_name': 'all_matches',
        'results_file_path': 'results.json',
    }

    def __init__(self, config: dict):
        """
        Initializes the results routes with the given configuration and loads the data.

        Parameters
        ----------
        config : dict
            Configuration dictionary. Expected keys:
            - 'arr_name': str, name of the array in the .npz file. Defaults to 'all_results'.
            - 'results_path': str, path to directory containing the results file.
            - 'results_file': str, filename of the .npz file containing result data.
        """
        self.config = {**self.default_config, **config}

        if not self.config['results_file_path']:
            raise Exception('Missing "results_file_path"')

        self._load_data()

        # Cria o blueprint para o grupo de rotas
        self.blueprint = Blueprint('results', __name__)
        self._routes()

    def _load_data(self):
        """
        Loads the match data from the specified JSON file.

        Returns
        -------
        dict
            The match data loaded from the file.
        """
        results_file = self.config['results_file_path']
        with open(results_file, 'r') as f:
            self.general_results = json.load(f)

        self.individual_data = np.load(self.config['matches_file_path'], allow_pickle=True)[self.config['arr_name']]

    def _routes(self):
        """
        Define as rotas Flask para as visualizações.
        """

        @self.blueprint.route('/')
        def index():
            """
            Página principal que mostra os resultados gerais do experimento
            e oferece botões para navegar para o cenário atual ou resultados individuais.
            """
            return render_template('index.html', metrics=self.general_results)

        @self.blueprint.route('/individual_results')
        def individual_results():
            """
            Exibe os resultados individuais como uma lista de opções.
            """
            # Aqui criamos uma lista de opções a partir dos resultados
            individual_list = self.individual_data
            return render_template('individual_results.html', results=individual_list)

        @self.blueprint.route('/details/<int:experiment_id>')
        def experiment_result(experiment_id:int):
            """
            Exibe os resultados de um experimento específico.
            """
            # Filtra o experimento com base no ID
            experiment = self.individual_data[experiment_id]
            return render_template('details.html', experiment_id=experiment_id, experiment=experiment)