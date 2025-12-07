"""
Module: semmatch.statistics.metrics.standard
--------------------------------------------

This module defines standard classification metrics such as Accuracy, Precision,
Recall, False Positive Ratio, and F1 Score, designed for evaluating match-related
analysis results within the SemMatch statistics pipeline.
"""

from typing import Union

from semmatch.statistics.pipeline_data import MatchResult
from semmatch.statistics.metrics.base import BaseMetric


def _calculate(hits: int, total: int) -> Union[float, None]:
    """
    Calculates a ratio (hits / total).

    Parameters
    ----------
    hits : int
        The number of successful occurrences.
    total : int
        The total number of occurrences.

    Returns
    -------
    Union[float, None]
        The calculated ratio as a float, or None if total is zero to avoid
        ZeroDivisionError.
    """
    return hits / total if total > 0 else None


class StandardMetric(BaseMetric):
    """
    Base class for standard classification metrics.

    This class serves as a common base for metrics that process `MatchResult`
    objects. It sets the `_allowed_result_type` to `MatchResult`, ensuring
    that subclasses are compatible with match analysis data.

    Attributes
    ----------
    _allowed_result_type : list[type]
        Specifies that this metric processes `MatchResult` objects.
    """
    _allowed_result_type: list[type] = [MatchResult]


class Accuracy(StandardMetric):
    """
    Calculates the accuracy of match results.

    Accuracy is defined as (True Positives + True Negatives) / Total.

    Attributes
    ----------
    _hits : int
        Accumulates the sum of true positives and true negatives across updates.
    _total : int
        Accumulates the total number of samples across updates.
    """

    def __init__(self, config=None):
        super().__init__(config)

        self._hits = 0
        self._total = 0

    def update(self, data) -> None:
        pair_hit = data.true_positives + data.true_negatives
        pair_total = data.true_positives + data.false_positives\
            + data.false_negatives + data.true_negatives

        accuracy = _calculate(pair_hit, pair_total)

        self._raw_results.append(accuracy)

        self._hits += pair_hit
        self._total += pair_total

    def compute(self) -> None:
        self._result = _calculate(self._hits, self._total)

    def reset(self) -> None:
        super().reset()

        self._hits = 0
        self._total = 0


class Precision(StandardMetric):
    """
    Calculates the precision of match results.

    Precision is defined as True Positives / (True Positives + False Positives).

    Attributes
    ----------
    _hits : int
        Accumulates the sum of true positives across updates.
    _total : int
        Accumulates the sum of true positives and false positives across updates.
    """

    def __init__(self, config=None):
        super().__init__(config)

        self._hits = 0
        self._total = 0

    def update(self, data) -> None:
        pair_hit = data.true_positives
        pair_total = data.true_positives + data.false_positives

        self._raw_results.append(_calculate(pair_hit, pair_total))

        self._hits += pair_hit
        self._total += pair_total

    def compute(self) -> None:
        self._result = _calculate(self._hits, self._total)

    def reset(self) -> None:
        super().reset()

        self._hits = 0
        self._total = 0


class Recall(StandardMetric):
    """
    Calculates the recall of match results.

    Recall is defined as True Positives / (True Positives + False Negatives).

    Attributes
    ----------
    _hits : int
        Accumulates the sum of true positives across updates.
    _total_positives : int
        Accumulates the sum of true positives and false negatives across updates.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._hits = 0
        self._total_positives = 0

    def update(self, data) -> None:
        pair_hits = data.true_positives
        pair_valid_points = data.true_positives + data.false_negatives

        self._raw_results.append(pair_hits / pair_valid_points)

        self._hits += pair_hits
        self._total_positives += pair_valid_points

    def compute(self) -> None:
        self._result = _calculate(self._hits, self._total_positives)

    def reset(self) -> None:
        super().reset()

        self._hits = 0
        self._total_positives = 0


class FalsePositiveRatio(StandardMetric):
    """
    Calculates the False Positive Ratio (FPR) of match results.

    FPR is defined as False Positives / (False Positives + True Negatives).

    Attributes
    ----------
    _misses : int
        Accumulates the sum of false positives across updates.
    _total : int
        Accumulates the sum of false positives and true negatives across updates.
    """

    def __init__(self, config=None):
        super().__init__(config)

        self._misses = 0
        self._total = 0

    def update(self, data) -> None:
        pair_hit = data.false_positives
        pair_negatives = data.false_positives + data.true_negatives

        self._raw_results.append(_calculate(pair_hit, pair_negatives))

        self._misses += pair_hit
        self._total += pair_negatives

    def compute(self) -> None:
        self._result = _calculate(self._misses, self._total)

    def reset(self) -> None:
        super().reset()

        self._misses = 0
        self._total = 0


class F1Score(StandardMetric):
    """
    Calculates the F1 Score of match results.

    The F1 Score is the harmonic mean of precision and recall, defined as
    2 * (Precision * Recall) / (Precision + Recall) or
    2 * TP / (2 * TP + FP + FN).

    Attributes
    ----------
    _fp : int
        Accumulates the sum of false positives across updates.
    _fn : int
        Accumulates the sum of false negatives across updates.
    _tp : int
        Accumulates the sum of true positives across updates.
    """

    def __init__(self, config=None):
        super().__init__(config)

        self._fp: int = 0
        self._fn: int = 0
        self._tp: int = 0

    def _calculate_f1(self, tp: int, fp: int, fn: int) -> Union[float, None]:
        """
        Helper method to calculate the F1 score for given true positives, false positives,
        and false negatives.

        Parameters
        ----------
        tp : int
            Number of true positives.
        fp : int
            Number of false positives.
        fn : int
            Number of false negatives.

        Returns
        -------
        Union[float, None]
            The calculated F1 score as a float, or None if the denominator is zero.
        """
        return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else None

    def update(self, data) -> None:
        tp = data.true_positives
        fp = data.false_positives
        fn = data.false_negatives

        self._raw_results.append(self._calculate_f1(tp, fp, fn))

        self._tp += tp
        self._fp += fp
        self._fn += fn

    def compute(self) -> None:
        self._result = self._calculate_f1(self._tp, self._fp, self._fn)

    def reset(self) -> None:
        super().reset()

        self._fp = 0
        self._fn = 0
        self._tp = 0
