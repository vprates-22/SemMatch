from semmatch.statistics.base_metric import BaseMetric, UpdateData


class Recall(BaseMetric):
    _hits = 0
    _total_positives = 0

    def update(self, data: UpdateData) -> None:
        pair_hits = data.inliers.sum()
        pair_valid_points = data.valid_projections.size

        self._raw_results.append(pair_hits / pair_valid_points)

        self._hits += pair_hits
        self._total_positives += pair_valid_points

    def compute(self) -> None:
        self._result = self._hits / self._total_positives if self._total_positives else None

    def reset(self) -> None:
        super().reset()

        self._hits = 0
        self._total_positives = 0
