from typing import Union

from semmatch.statistics.base_metric import BaseMetric, UpdateData
from semmatch.statistics.metrics.precision import Precision
from semmatch.statistics.metrics.recall import Recall


class F1Score(BaseMetric):
    _dependencies = [Precision, Recall]

    def _calculate_f1(
        self,
        precision: Union[float, None],
        recall: Union[float, None]
    ) -> Union[float, None]:
        has_valid_values = precision and recall and precision + recall != 0
        f1_score = 2 * precision * recall / (precision + recall) \
            if has_valid_values else None

        return f1_score

    def update(self, data: UpdateData) -> None:
        recall = self._dependency_objects[Recall].get_last_raw_result()
        precision = self._dependency_objects[Precision].get_last_raw_result()

        self._raw_results.append(self._calculate_f1(precision, recall))

    def compute(self) -> None:
        recall = self._dependency_objects[Recall].get_result()
        precision = self._dependency_objects[Precision].get_result()

        self._result = self._calculate_f1(precision, recall)
