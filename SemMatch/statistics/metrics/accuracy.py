from semmatch.statistics.base_metric import BaseMetric, UpdateData


class Accuracy(BaseMetric):
    _hits: int = 0
    _total: int = 0

    def update(self, data: UpdateData) -> None:
        pair_hit = data.inliers.sum()
        pair_total = data.mkpts0.size(0)

        accuracy = (pair_hit / pair_total) if pair_total > 0 else None
        self._raw_results.append(accuracy)

        self._hits += pair_hit
        self._total += pair_total

    def compute(self) -> None:
        self._result = self._hits / self._total if self._total else None

    def reset(self) -> None:
        super().reset()

        self._hits = 0
        self._total = 0
