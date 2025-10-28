from semmatch.statistics.base_metric import BaseMetric, UpdateData


class Precision(BaseMetric):
    _hits = 0
    _total = 0

    def update(self, data: UpdateData) -> None:
        pair_hit = data.inliers.sum()
        pair_total = data.mkpts0.size

        self._raw_results.append(pair_hit / pair_total)

        self._hits += pair_hit
        self._total += pair_total

    def compute(self) -> None:
        self._result = self._hits / self._total if self._total else None

    def reset(self) -> None:
        super().reset()

        self._hits = 0
        self._total = 0
