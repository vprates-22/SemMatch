import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from torch import Tensor
from pathlib import Path
from typing import Any, Dict, Union, List

from semmatch.datasets import get_dataset
from semmatch.utils.image import resize_long_edge
from semmatch.utils.validation import ValidatedClass
from semmatch.settings import RESULTS_PATH, MATCHES_PATH
from semmatch.statistics.pipeline_data import RawDataInput
from semmatch.configs.evaluator_config import Config, EvaluatorConfig
from semmatch.statistics.orchestrator import PipelineOrchestrator, AnalysisPlan


class Evaluator(ValidatedClass):

    _validation_rules = {
        'dataset': {'required': True, 'type': str},
        'n_workers': {'required': False, 'type': int, 'default': 1},
        'metrics_config': {'required': False, 'type': [dict, Config], 'default': {}},
        'dataset_config': {'required': False, 'type': [dict, Config], 'default': {}},
        'resize': {'required': False, 'type': tuple},
        # 'report': {'required': True, 'type': type},
        'results_path': {'required': False, 'type': str, 'default': RESULTS_PATH},
        'matches_path': {'required': False, 'type': str, 'default': MATCHES_PATH}
    }

    def __init__(self, plan: List[AnalysisPlan] = None, config: Union[Config, Dict[str, Any]] = None):
        super().__init__(EvaluatorConfig(config))

        self.metric_orchestrator = PipelineOrchestrator(
            plan, self.config)

        # self.report = self.config.report(self.metric_orchestrator, 0)

        if self.config.n_workers == -1:
            self.config.n_workers = mp.cpu_count()

        self.dataset = get_dataset(self.config.dataset)(
            self.config.dataset_config)

    def extract_matches(self, matcher_fn: callable, name: str) -> None:
        if name == '':
            name = matcher_fn.__name__

        matches_path = Path(self.config.matches_path)
        fname = matches_path / f'{name}_matches.npz'

        matches = []

        for pair in tqdm(self.dataset.pairs, desc='Extracting matches', total=len(self.dataset.pairs)):
            image0 = self.dataset.read_image(pair['image0'])
            image1 = self.dataset.read_image(pair['image1'])

            scale_img0 = scale_img1 = 1.0
            if self.config.resize is not None:
                image0, scale_img0 = resize_long_edge(
                    image0, self.config.resize)
                image1, scale_img1 = resize_long_edge(
                    image1, self.config.resize)

            mkpts0, mkpts1 = matcher_fn(image0, image1)
            if isinstance(mkpts0, Tensor):
                mkpts0 = mkpts0.cpu().numpy()
                mkpts1 = mkpts1.cpu().numpy()

            if self.config.resize is not None:
                mkpts0 = mkpts0 / scale_img0
                mkpts1 = mkpts1 / scale_img1

            matches.append({
                'image0': pair['image0'],
                'image1': pair['image1'],
                'mkpts0': mkpts0,
                'mkpts1': mkpts1
            })

        np.savez_compressed(fname, matches=matches)

        return matches

    def run(self, matcher_fn: callable, name: str = '') -> None:
        self.metric_orchestrator.reset()

        matches = self.extract_matches(matcher_fn, name)

        for idx, match in enumerate(matches):
            mkpts0 = match['mkpts0']
            mkpts1 = match['mkpts1']

            img0_path = match['image0']
            img1_path = match['image1']

            img0 = self.dataset.read_image(img0_path)
            img1 = self.dataset.read_image(img1_path)

            self.metric_orchestrator.execute(RawDataInput(
                image0_path=img0_path,
                image1_path=img1_path,
                image0=img0,
                image1=img1,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                dataset=self.dataset,
                pair_index=idx
            ))

        self.metric_orchestrator.summarize()
