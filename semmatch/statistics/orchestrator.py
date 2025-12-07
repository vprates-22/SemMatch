
"""
Module: semmatch.statistics.orchestrator
----------------------------------------

This module defines the `PipelineOrchestrator` class, which is responsible for
managing and executing a data analysis pipeline. It coordinates data generation,
analysis, and metric computation based on a flexible `AnalysisPlan`.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Union, Any

from semmatch.configs.base import Config
from semmatch.statistics.metrics.base import BaseMetric
from semmatch.statistics.analyzers import DataAnalyzer
from semmatch.statistics.pipeline_data import RawDataInput, AnalysisResult
from semmatch.statistics.data_generators import DataGenerator


@dataclass
class AnalysisPlan:
    """
    Defines a single analysis step within the pipeline.

    Attributes
    ----------
    title : str
        A descriptive title for this analysis plan.
    analysis : type[DataAnalyzer]
        The `DataAnalyzer` class to be used for this analysis step.
    metrics : List[type[BaseMetric]]
        A list of `BaseMetric` classes to be applied to the results
        produced by the `analysis`.
    """
    title: str
    analysis: type[DataAnalyzer]
    metrics: List[type[BaseMetric]]

# Plan example:
# [
#   AnalysisPlan(title="Example Analysis",
#                analysis=ExampleAnalyzer,
#                metrics=[MetricA, MetricB, MetricC]),
#   AnalysisPlan(title="Another Analysis",
#                analysis=AnotherAnalyzer,
#                metrics=[MetricD, MetricE]),
#   ...
# ]


class PipelineOrchestrator:
    """
    Orchestrates the execution of a data analysis pipeline based on a defined plan.

    The orchestrator manages data generators, analyzers, and metrics, ensuring
    that data flows correctly through the pipeline and metrics are updated
    and computed as expected.

    Parameters
    ----------
    plan : List[AnalysisPlan]
        A list of `AnalysisPlan` objects defining the analysis steps,
        including the analyzer to use and the metrics to apply.
    config : Union[Config, Dict]
        Configuration settings for the pipeline, passed to generators,
        analyzers, and metrics.

    Raises
    ------
    ValueError
        If the provided `plan` is invalid, specifically if a metric is
        not compatible with its associated analyzer's result type.
    """

    def __init__(self, plan: List[AnalysisPlan], config: Union[Config, Dict]):
        error_message = self._validate_plan(plan)
        if error_message:
            separator = "\n\t"
            raise ValueError(
                f"Invalid plan:\n\t{separator.join(error_message)}")

        self.first = True

        self.config: Config = Config(config)
        self.plan: List[AnalysisPlan] = plan
        self.pairs: List[List[str]] = []

        self.orchestrator_data: Dict[str,
                                     Dict[type[BaseMetric], Dict[Any, BaseMetric]]] = {
            analysis_plan.title: {
                metric: {} for metric in analysis_plan.metrics
            } for analysis_plan in plan
        }

        self.data_analyzers_objects: List[DataAnalyzer] =\
            self._get_data_analyzers(plan)
        self.data_generators: List[DataGenerator] =\
            self._get_data_generators(plan)

    def _validate_plan(self, plan: List[AnalysisPlan]) -> list[str]:
        """
        Validates the provided analysis plan to ensure compatibility between
        analyzers and metrics.

        Checks if each metric specified in an `AnalysisPlan` is compatible
        with the `AnalysisResult` type produced by its associated `DataAnalyzer`.

        Parameters
        ----------
        plan : List[AnalysisPlan]
            The analysis plan to validate.

        Returns
        -------
        list[str]
            A list of error messages if the plan is invalid, or an empty list if valid.
        """
        errors = []

        for analysis_plan in plan:
            analysis = analysis_plan.analysis
            analysis_result_type = analysis.get_result_type()
            metrics = analysis_plan.metrics

            for metric in metrics:
                allowed_result_types = metric.get_allowed_result_types()

                if not any(issubclass(allowed, analysis_result_type)
                           for allowed in allowed_result_types):
                    errors.append(
                        f"Metric '{metric.__name__}' is not compatible with analysis "
                        f"'{analysis.__name__}' result type "
                        f"'{analysis_result_type.__name__}'"
                    )

        return errors

    def _get_data_analyzers(self, plan: List[AnalysisPlan]) -> List[DataAnalyzer]:
        """
        Instantiates all unique `DataAnalyzer` objects specified in the plan.

        Each analyzer is instantiated once, even if it appears in multiple
        `AnalysisPlan` entries.

        Parameters
        ----------
        plan : List[AnalysisPlan]
            The analysis plan containing analyzer types.

        Returns
        -------
        List[DataAnalyzer]
            A list of instantiated `DataAnalyzer` objects.
        """
        data_analyzers = set()
        data_analyzers_obj = list()
        for analysis_plan in plan:
            data_analyzers.add(analysis_plan.analysis)

        for data_analyzer in data_analyzers:
            data_analyzers_obj.append(data_analyzer(self.config))

        return data_analyzers_obj

    def _get_data_generators(self, plan: List[AnalysisPlan]) -> List[DataGenerator]:
        """
        Instantiates all unique `DataGenerator` objects required by the analyzers in the plan.

        Dependencies are extracted from analyzers, and each generator is instantiated
        once.

        Parameters
        ----------
        plan : List[AnalysisPlan]
            The analysis plan from which to extract generator dependencies.

        Returns
        -------
        List[DataGenerator]
            A list of instantiated `DataGenerator` objects.
        """
        data_generators = set()
        data_generators_obj = list()
        for analysis_plan in plan:
            deps = analysis_plan.analysis.get_dependencies()
            if not deps:
                continue

            if isinstance(deps, (list, tuple, set)):
                data_generators.update(deps)
            else:
                data_generators.add(deps)

        for data_generator in data_generators:
            data_generators_obj.append(data_generator(self.config))

        return data_generators_obj

    def _create_metrics_objects(self, analisys_result: Dict[type[DataAnalyzer], list[AnalysisResult]]) -> None:
        """
        Creates metric objects based on the initial analysis results.

        This method is called only once during the first `execute` call. It
        instantiates a metric object for each unique result key produced by
        each analyzer in the plan.

        Parameters
        ----------
        analisys_result : Dict[type[DataAnalyzer], list[AnalysisResult]]
            A dictionary mapping analyzer types to their initial list of `AnalysisResult` objects.
        """
        for analysis_plan in self.plan:
            title = analysis_plan.title
            analysis = analysis_plan.analysis
            metrics = analysis_plan.metrics

            analysis_result = analisys_result[analysis]

            for metric in metrics:
                for result in analysis_result:
                    self.orchestrator_data[title][metric][result.key] = metric(
                        config=self.config
                    )

    def execute(self, data: RawDataInput) -> None:
        """
        Executes one iteration of the pipeline with new raw data.

        This involves generating data, analyzing it, and updating all relevant
        metric objects with the results. If it's the first execution, metric
        objects are also created.

        Parameters
        ----------
        data : RawDataInput
            The raw input data for the current iteration.
        """
        path0 = Path(data.image0_path)
        path1 = Path(data.image1_path)

        self.pairs.append([path0.parent.name + '/' + path0.stem,
                          path1.parent.name + '/' + path1.stem])

        generated_data = {}
        for data_generator in self.data_generators:
            generated_data[data_generator.__class__
                           ] = data_generator.generate(data)

        analysis_result = {}
        for data_analyzer in self.data_analyzers_objects:
            analysis_result[data_analyzer.__class__] = data_analyzer.analyze(
                generated_data)

        if self.first:
            self._create_metrics_objects(analysis_result)
            self.first = False

        for analysis_plan in self.plan:
            title = analysis_plan.title
            analysis = analysis_plan.analysis
            metrics = analysis_plan.metrics

            analysis_results = analysis_result[analysis]

            for metric in metrics:
                for result in analysis_results:
                    metric_obj = self.orchestrator_data[title][metric][result.key]
                    metric_obj.update(data=result)

    def summarize(self) -> None:
        """
        Computes the final results for all metrics.

        This method iterates through all instantiated metric objects and calls
        their `summarize` method to finalize their results.
        """
        for analysis_plan in self.plan:
            title = analysis_plan.title
            for metric_type, metrics_dict in self.orchestrator_data[title].items():
                for metric_obj in metrics_dict.values():
                    metric_obj.compute()

    def reset(self) -> None:
        """
        Resets the internal state of all metric objects.

        This method iterates through all instantiated metric objects and calls
        their `reset` method, typically clearing any accumulated raw data.
        """
        for analysis_plan in self.plan:
            title = analysis_plan.title
            metrics = analysis_plan.metrics

            for metric in metrics:
                for _, metric_obj in self.orchestrator_data[title][metric].items():
                    metric_obj.reset()

    def get_results(self) -> Dict[type[BaseMetric], BaseMetric]:
        """
        Retrieves the computed results and raw data from all metrics.

        The results are structured hierarchically by analysis title, metric type,
        and result key, providing both the final computed value and the raw
        data accumulated during execution.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - "pairs": A list of processed image pair identifiers.
            - Analysis results structured by title, metric, and key, with "result" and "raw_results".
        """
        return {
            "pairs": self.pairs,
            **{
                analysis_title: {
                    metric.__name__: {
                        key: {
                            "result": metric_obj.get_result(),
                            "raw_results": metric_obj.get_raw_results()
                        }
                        for key, metric_obj in metrics_dict.items()
                    }
                    for metric, metrics_dict in analysis_metrics.items()
                }
                for analysis_title, analysis_metrics in self.orchestrator_data.items()
            }
        }
