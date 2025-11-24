from pathlib import Path
from typing import List, Dict, Union, Any

from semmatch.configs.base import Config
from semmatch.statistics.base_metric import BaseMetric, UpdateData


class MetricsOrchestrator:
    def __init__(self, metrics: List[type[BaseMetric]], config: Union[Config, Dict[str, Any]]):
        self.config: Config = Config(config)
        self.pairs: List[List[str]] = []

        self.main_metrics: List[type[BaseMetric]] = list(set(metrics))
        self.auxiliary_metrics: List[type[BaseMetric]
                                     ] = self._get_auxiliary_metrics()

        self.all_metrics: List[type[BaseMetric]] = self.main_metrics + \
            self.auxiliary_metrics

        has_cycle, cycle_path = self._verify_cycles()

        if has_cycle:
            raise Exception(
                f"The following circular dependency was found: {' -> '.join([metric.__name__ for metric in cycle_path])}")

        self.all_metrics: List[type[BaseMetric]] = self._sort_metrics()
        self.class_to_object: Dict[type[BaseMetric],
                                   BaseMetric] = self._instantiate_objects()

    def _get_auxiliary_metrics(self) -> List[type[BaseMetric]]:
        auxiliary_metrics = set()
        for metric in self.main_metrics:
            for dependency in metric.get_dependencies():
                if dependency not in self.main_metrics:
                    auxiliary_metrics.add(dependency)

        return list(auxiliary_metrics)

    def _verify_cycles(self) -> bool:
        if not self.all_metrics:
            return False

        metric_to_index = {k: i for i, k in enumerate(self.all_metrics)}
        seen = [False] * len(self.all_metrics)
        stack = []

        current_metric = self.all_metrics[0]

        while True:
            i = metric_to_index[current_metric]
            should_stack = True

            if seen[i]:
                stack.pop()
                if stack:
                    should_stack = False
                    current_metric = stack[-1]
                else:
                    if False not in seen:
                        return False, None

                    i = seen.index(False)
                    current_metric = self.all_metrics[i]

            dependencies = current_metric.get_dependencies()
            if should_stack and not seen[i]:
                if any(metric in stack for metric in dependencies):
                    return True, stack

                seen[i] = True
                stack.append(current_metric)

            not_seen_dependencies = [dependency for dependency in dependencies
                                     if not seen[metric_to_index[dependency]]]

            if not_seen_dependencies:
                current_metric = not_seen_dependencies[0]

    def _sort_metrics(self) -> List[type[BaseMetric]]:
        """
        Retorna uma lista ordenada de métricas onde cada métrica aparece
        depois de todas as métricas das quais depende.

        Levanta ValueError se houver ciclo.
        """
        graph = {metric: [] for metric in self.all_metrics}
        in_degree = {metric: 0 for metric in self.all_metrics}

        for metric in self.all_metrics:
            for dependency in metric.get_dependencies():
                graph[dependency].append(metric)
                in_degree[metric] += 1

        queue = [m for m in self.all_metrics if in_degree[m] == 0]
        sorted_metrics = []

        while queue:
            current = queue.pop(0)
            sorted_metrics.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_metrics) != len(self.all_metrics):
            raise ValueError("Cycle detected in metric dependencies")

        return sorted_metrics

    def _instantiate_objects(self) -> Dict[type[BaseMetric], BaseMetric]:
        class_to_object = {}

        for metric_class in self.all_metrics:
            class_to_object[metric_class] = metric_class(
                class_to_object, self.config)

        return class_to_object

    def update(self, data: UpdateData) -> None:
        p0 = Path(data.image0)
        p1 = Path(data.image1)

        self.pairs.append([p0.parent.name + '/' + p0.stem,
                          p1.parent.name + '/' + p1.stem])

        for metric in self.all_metrics:
            self.class_to_object[metric].update(data)

    def compute(self) -> None:
        for metric in self.all_metrics:
            self.class_to_object[metric].compute()

    def reset(self) -> None:
        for metric in self.all_metrics:
            self.class_to_object[metric].reset()

    def get_results(self) -> Dict[type[BaseMetric], BaseMetric]:
        return {
            "pairs": self.pairs,
            **{
                metric.__name__: {
                    "pairs_results": self.class_to_object[metric].get_raw_results(),
                    "aggregated": self.class_to_object[metric].get_result()
                }
                for metric in self.main_metrics
            }
        }
