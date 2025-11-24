import numpy as np
import pandas as pd

from typing import Union
from abc import abstractmethod, ABCMeta

from semmatch.statistics.orchestrator import MetricsOrchestrator
from semmatch.report.static.modes import ReportMode


class BaseStaticReport(metaclass=ABCMeta):
    def __init__(self, metric_orchestrator: MetricsOrchestrator, mode: Union[int, ReportMode] = ReportMode.SHOW_SUMMARY_ONLY):
        self.metric_orchestrator = metric_orchestrator
        self.mode = ReportMode(mode)

    @abstractmethod
    def generate_report(self):
        raise NotImplementedError

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Creates a summary table combining descriptive statistics (like pd.describe)
        and aggregated values for each metric.
        """
        metric_data = self.metric_orchestrator.get_results()

        # Filtra apenas métricas válidas
        metrics = {
            k: v for k, v in metric_data.items()
            if isinstance(v, dict) and 'pairs_results' in v
        }

        # Garantir listas do mesmo tamanho
        n_pairs = len(metric_data.get('pairs', []))
        df_pairs = pd.DataFrame({
            k: (v['pairs_results'][:n_pairs] if isinstance(v['pairs_results'], list) else [np.nan] * n_pairs)
            for k, v in metrics.items()
        })

        # Substituir inf/-inf por NaN
        df_pairs = df_pairs.replace([np.inf, -np.inf], np.nan)

        # Estatísticas descritivas
        df_describe = df_pairs.describe().T.drop(columns=['count'], errors='ignore')

        # Linha de valores agregados
        df_value = pd.DataFrame({k: [v.get('aggregated', np.nan)] for k, v in metrics.items()},
                                index=['value'])

        # Combinar
        df_summary = pd.concat([df_value, df_describe.T])
        df_summary = df_summary.replace([np.inf, -np.inf], np.nan).round(3)

        return df_summary


    def generate_pairs_table(self) -> pd.DataFrame:
        """
        Creates a dataframe with one row per pair, and metrics as columns.
        """
        metric_data = self.metric_orchestrator.get_results()

        metrics = {
            k: v for k, v in metric_data.items()
            if isinstance(v, dict) and 'pairs_results' in v
        }

        pairs_list = metric_data.get('pairs', [])
        n_pairs = len(pairs_list)

        # Criar DataFrame com métricas, alinhando corretamente pelo número de pares
        df_metrics = pd.DataFrame({
            k: (v['pairs_results'][:n_pairs] if isinstance(v['pairs_results'], list) else [np.nan] * n_pairs)
            for k, v in metrics.items()
        })

        # Substituir inf/-inf
        df_metrics = df_metrics.replace([np.inf, -np.inf], np.nan)

        # Adicionar nomes dos pares
        if pairs_list:
            n_images = len(pairs_list[0])
            df_pairs_names = pd.DataFrame(pairs_list, columns=[f'Image_{i}' for i in range(n_images)])
            df_final = pd.concat([df_pairs_names, df_metrics], axis=1)
        else:
            df_final = df_metrics

        df_final = df_final.round(3).reset_index(drop=True)
        return df_final
