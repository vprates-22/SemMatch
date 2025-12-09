import numpy as np
import pandas as pd

from typing import Union
from abc import abstractmethod, ABCMeta

from semmatch.statistics.orchestrator import PipelineOrchestrator
from semmatch.report.static.modes import ReportMode


class BaseStaticReport(metaclass=ABCMeta):
    def __init__(self, metric_orchestrator: PipelineOrchestrator, mode: Union[int, ReportMode] = ReportMode.SHOW_SUMMARY_ONLY):
        self.metric_orchestrator = metric_orchestrator
        self.mode = ReportMode(mode)

    @abstractmethod
    def generate_report(self):
        raise NotImplementedError

    def _normalize_metric_data(self, metric_data: dict) -> dict:
        """
        Normalize metric data to a consistent flat structure.

        Returns a dict with keys:
          - 'pairs': list of pairs
          - 'metrics': dict mapping metric_key -> {'pairs_results': list, 'aggregated': value}

        Supports both legacy flat formats (metric -> {pairs_results, aggregated})
        and nested formats like the provided `test.json` (Category -> Metric -> Subkey -> {result, raw_results}).
        """
        if not isinstance(metric_data, dict):
            return {'datasets': {}}

        global_pairs = metric_data.get('pairs', [])

        def is_leaf(d: object) -> bool:
            """Return True if object looks like a metric leaf containing results."""
            if not isinstance(d, dict):
                return False
            keys = set(d.keys())
            return bool(keys & {'result', 'raw_results', 'pairs_results', 'aggregated'})

        def flatten(prefix: list[str], obj: dict, out: dict):
            """Recursively flatten nested dict `obj` into out mapping of tuple(prefix+path) -> leaf dict."""
            if is_leaf(obj):
                out[tuple(prefix)] = {
                    'pairs_results': obj.get('raw_results', obj.get('pairs_results', [])),
                    'aggregated': obj.get('result', obj.get('aggregated', None))
                }
                return

            for k, v in obj.items():
                if isinstance(v, dict):
                    flatten(prefix + [str(k)], v, out)

        datasets: dict = {}

        # For each top-level key except 'pairs', create a dataset
        for top_key, top_val in metric_data.items():
            if top_key == 'pairs':
                continue

            ds_pairs = global_pairs
            # if dataset has its own pairs key, prefer it
            if isinstance(top_val, dict) and 'pairs' in top_val:
                ds_pairs = top_val.get('pairs', global_pairs)

            metrics_flat: dict = {}
            if isinstance(top_val, dict):
                # If top_val itself looks like a leaf (flat metric), treat its immediate children as metrics
                # e.g., legacy flat inside a named dataset
                for k, v in top_val.items():
                    if k == 'pairs':
                        continue
                    if is_leaf(v):
                        metrics_flat[(k,)] = {
                            'pairs_results': v.get('raw_results', v.get('pairs_results', [])),
                            'aggregated': v.get('result', v.get('aggregated', None))
                        }

                # If no immediate leaves found, attempt recursive flatten
                if not metrics_flat:
                    flatten([], top_val, metrics_flat)

            # Build display keys according to rules:
            # - metric base name is the path without the last element
            # - internal key is the last element; if 'default', omit parentheses
            display_metrics: dict = {}
            for path_tuple, leaf in metrics_flat.items():
                if len(path_tuple) == 0:
                    # unexpected, skip
                    continue
                if len(path_tuple) == 1:
                    base = path_tuple[0]
                    internal = 'default'
                else:
                    base = '/'.join(path_tuple[:-1])
                    internal = path_tuple[-1]

                if internal == 'default' or internal == '' or internal is None:
                    display_name = base
                else:
                    display_name = f"{base} ({internal})"

                display_metrics[display_name] = leaf

            datasets[top_key] = {
                'pairs': ds_pairs,
                'metrics': display_metrics
            }

        return {'datasets': datasets}

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Creates a summary table combining descriptive statistics (like pd.describe)
        and aggregated values for each metric.
        """
        metric_data = self.metric_orchestrator.get_results()
        norm = self._normalize_metric_data(metric_data)

        datasets = norm.get('datasets', {})
        results: dict[str, pd.DataFrame] = {}

        for ds_name, ds in datasets.items():
            metrics = ds.get('metrics', {})
            pairs = ds.get('pairs', [])
            n_pairs = len(pairs)

            df_pairs = pd.DataFrame({
                k: (v['pairs_results'][:n_pairs] if isinstance(v['pairs_results'], list) else [np.nan] * n_pairs)
                for k, v in metrics.items()
            })

            df_pairs = df_pairs.replace([np.inf, -np.inf], np.nan)
            df_describe = df_pairs.describe().T.drop(columns=['count'], errors='ignore')

            df_value = pd.DataFrame({k: [v.get('aggregated', np.nan)] for k, v in metrics.items()}, index=['value'])
            df_summary = pd.concat([df_value, df_describe.T])
            df_summary = df_summary.replace([np.inf, -np.inf], np.nan).round(3)

            results[ds_name] = df_summary

        return results


    def generate_pairs_table(self) -> pd.DataFrame:
        """
        Creates a dataframe with one row per pair, and metrics as columns.
        """
        metric_data = self.metric_orchestrator.get_results()
        norm = self._normalize_metric_data(metric_data)

        datasets = norm.get('datasets', {})
        results: dict[str, pd.DataFrame] = {}

        for ds_name, ds in datasets.items():
            metrics = ds.get('metrics', {})
            pairs_list = ds.get('pairs', [])
            n_pairs = len(pairs_list)

            df_metrics = pd.DataFrame({
                k: (v['pairs_results'][:n_pairs] if isinstance(v['pairs_results'], list) else [np.nan] * n_pairs)
                for k, v in metrics.items()
            })

            df_metrics = df_metrics.replace([np.inf, -np.inf], np.nan)

            if pairs_list:
                n_images = len(pairs_list[0])
                df_pairs_names = pd.DataFrame(pairs_list, columns=[f'Image_{i}' for i in range(n_images)])
                df_final = pd.concat([df_pairs_names, df_metrics], axis=1)
            else:
                df_final = df_metrics

            df_final = df_final.round(3).reset_index(drop=True)
            results[ds_name] = df_final

        return results
