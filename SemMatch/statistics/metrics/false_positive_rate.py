from semmatch.statistics.base_metric import BaseMetric, UpdateData


class FalsePositiveRatio(BaseMetric):
    _misses = 0
    _total = 0

    def update(self, data: UpdateData) -> None:
        pair_hit = (~data.inliers).sum()
        pair_negatives = data.inliers.size - data.valid_projections.size

        self._raw_results.append(pair_hit / pair_negatives)

        self._misses += pair_hit
        self._total += pair_negatives

    def compute(self) -> None:
        self._result = self._misses / self._total if self._total else None
