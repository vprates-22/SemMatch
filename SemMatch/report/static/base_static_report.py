import pandas as pd

from typing import Union
from abc import abstractmethod, ABCMeta

from semmatch.statistics.orchestrator import MetricsOrchestrator
from semmatch.report.static.modes import ReportMode


class BaseStaticReport(meta=ABCMeta):
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

        Parameters
        ----------
        None


        Returns
        -------
        pd.DataFrame
            A summary table with descriptive stats and a 'value' row.
        """
        metric_data = self.metric_orchestrator.get_results()
        metrics = {
            k: v for k, v in metric_data.items()
            if isinstance(v, dict) and 'pairs_results' in v
        }

        df_base = pd.DataFrame.from_dict(metrics, orient='index')

        # Create dataframe of pair values
        df_pairs = pd.DataFrame({k: v['pairs_results']
                                for k, v in metrics.items()})

        # Generate descriptive statistics
        df_describe = df_pairs.describe().T

        if 'count' in df_describe.index:
            df_describe = df_describe.drop(index='count')
        if 'count' in df_describe.columns:
            df_describe = df_describe.drop(columns='count')

        # Aggregated values (renamed to 'value')
        df_value = df_base[['aggregated']].T
        df_value.index = ['value']

        # Combine stats + value
        df_summary = pd.concat([df_value, df_describe.T])

        df_summary = df_summary.round(3)

        return df_summary

    def generate_pairs_table(self) -> pd.DataFrame:
        """
        Creates a dataframe with one row per pair, and metrics as columns.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            A dataframe where each row represents one 'pair' across all metrics.
        """
        metric_data = self.metric_orchestrator.get_results()
        metrics = {
            k: v for k, v in metric_data.items()
            if isinstance(v, dict) and 'pairs_results' in v
        }
        df_base = pd.DataFrame.from_dict(metrics, orient='index')

        # Explode the 'pairs_results' lists into separate rows
        df_pairs_exploded = df_base['pairs_results'].explode()

        # Add position index for each pair (0, 1, 2, ...)
        pair_position = df_pairs_exploded.groupby(level=0).cumcount()
        df_pairs_exploded.index = [df_pairs_exploded.index, pair_position]

        # Unstack so metrics become columns and positions become rows
        df_pairs_final = df_pairs_exploded.unstack(level=0)
        df_pairs_final.columns.name = None

        # Reset index for a clean look
        df_pairs_final = df_pairs_final.reset_index(drop=True)

        df_pairs_names = pd.DataFrame(metric_data['pairs'], columns=[
                                      f'Image_{i}' for i in range(len(metric_data['pairs'][0]))])
        df_pairs_final = pd.concat([df_pairs_names, df_pairs_final], axis=1)

        return df_pairs_final
