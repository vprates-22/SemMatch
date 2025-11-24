from collections import defaultdict

from typing import Union

from semmatch.statistics.pipeline_data import MatchResult
from semmatch.statistics.metrics.base import BaseMetric


def _calculate(hits: int, total: int) -> Union[float, None]:
    return hits / total if total > 0 else None


class StandardMetric(BaseMetric):
    _allowed_result_type: list[type] = [MatchResult]


class Accuracy(StandardMetric):
    def __init__(self, config=None):
        super().__init__(config)

        self._hits = defaultdict(int)
        self._total = defaultdict(int)

    def update(self, data) -> None:
        if isinstance(data, MatchResult):
            data = [data]

        for match_data in data:
            pair_hit = match_data.true_positives + match_data.true_negatives
            pair_total = match_data.true_positives + match_data.false_positives\
                + match_data.false_negatives + match_data.true_negatives

            threshold = match_data.threshold
            accuracy = _calculate(pair_hit, pair_total)

            self._raw_results[threshold].append(accuracy)

            self._hits[threshold] += pair_hit
            self._total[threshold] += pair_total

    def compute(self) -> None:
        for threshold in self._raw_results:
            self._result[threshold] = _calculate(
                self._hits[threshold], self._total[threshold])

    def reset(self) -> None:
        super().reset()

        self._hits = defaultdict(int)
        self._total = defaultdict(int)


class Precision(StandardMetric):
    def __init__(self, config=None):
        super().__init__(config)

        self._hits = defaultdict(int)
        self._total = defaultdict(int)

    def update(self, data) -> None:
        if isinstance(data, MatchResult):
            data = [data]

        for match_data in data:
            threshold = match_data.threshold
            pair_hit = match_data.true_positives
            pair_total = match_data.true_positives + match_data.false_positives

            self._raw_results[threshold].append(
                _calculate(pair_hit, pair_total))

            self._hits[threshold] += pair_hit
            self._total[threshold] += pair_total

    def compute(self) -> None:
        for threshold in self._raw_results:
            self._result[threshold] = _calculate(
                self._hits[threshold], self._total[threshold])

    def reset(self) -> None:
        super().reset()

        self._hits = defaultdict(int)
        self._total = defaultdict(int)


class Recall(StandardMetric):
    def __init__(self, config=None):
        super().__init__(config)
        self._hits = defaultdict(int)
        self._total_positives = defaultdict(int)

    def update(self, data) -> None:
        if isinstance(data, MatchResult):
            data = [data]

        for match_data in data:
            threshold = match_data.threshold
            pair_hits = match_data.true_positives
            pair_valid_points = match_data.true_positives + match_data.false_negatives

            self._raw_results[threshold].append(pair_hits / pair_valid_points)

            self._hits[threshold] += pair_hits
            self._total_positives[threshold] += pair_valid_points

    def compute(self) -> None:
        for threshold in self._raw_results:
            self._result[threshold] = _calculate(
                self._hits[threshold], self._total_positives[threshold])

    def reset(self) -> None:
        super().reset()

        self._hits = defaultdict(int)
        self._total_positives = defaultdict


class FalsePositiveRatio(StandardMetric):
    def __init__(self, config=None):
        super().__init__(config)

        self._misses = defaultdict(int)
        self._total = defaultdict(int)

    def update(self, data) -> None:
        if isinstance(data, MatchResult):
            data = [data]

        for match_data in data:
            threshold = match_data.threshold
            pair_hit = match_data.false_positives
            pair_negatives = match_data.false_positives - match_data.true_negatives

            self._raw_results[threshold].append(
                _calculate(pair_hit, pair_negatives))

            self._misses[threshold] += pair_hit
            self._total[threshold] += pair_negatives

    def compute(self) -> None:
        for threshold in self._raw_results:
            self._result[threshold] = _calculate(
                self._misses[threshold], self._total[threshold])

    def reset(self) -> None:
        super().reset()

        self._misses = defaultdict(int)
        self._total = defaultdict(int)


class F1Score(StandardMetric):
    def __init__(self, config=None):
        super().__init__(config)

        self._fp: int = defaultdict(int)
        self._fn: int = defaultdict(int)
        self._tp: int = defaultdict(int)

    def _calculate_f1(self, tp: int, fp: int, fn: int) -> Union[float, None]:
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else None

    def update(self, data) -> None:
        if isinstance(data, MatchResult):
            data = [data]

        for match_data in data:
            threshold = match_data.threshold

            tp = match_data.true_positives
            fp = match_data.false_positives
            fn = match_data.false_negatives

            self._raw_results[threshold].append(self._calculate_f1(tp, fp, fn))

            self._tp[threshold] += tp
            self._fp[threshold] += fp
            self._fn[threshold] += fn

    def compute(self) -> None:
        for threshold in self._raw_results:
            self._result[threshold] = self._calculate_f1(
                self._tp[threshold], self._fp[threshold], self._fn[threshold])

    def reset(self) -> None:
        super().reset()

        self._fp = defaultdict(int)
        self._fn = defaultdict(int)
        self._tp = defaultdict(int)
