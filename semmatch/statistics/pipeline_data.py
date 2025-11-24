from dataclasses import dataclass

import numpy as np
from typing import List
from numpy.typing import NDArray
from semmatch.datasets.base import BaseDataset

# --- Camada 1: Input ---


@dataclass(frozen=True)
class RawDataInput:
    """Dados 100% crus para uma iteração."""
    image0: NDArray
    image1: NDArray
    mkpts0: NDArray
    mkpts1: NDArray
    dataset: BaseDataset
    pair_index: int

# --- Camada 2: Dados Gerados ---


class GeneratedData:
    """Classe base marcadora para dados gerados."""
    pass


@dataclass(frozen=True)
class KeypointData(GeneratedData):
    mkpts0: NDArray
    mkpts1: NDArray


@dataclass(frozen=True)
class InlierData(GeneratedData):
    inliers: NDArray[np.bool_]
    threshold: float


@dataclass(frozen=True)
class ProjectionData(GeneratedData):
    projections: NDArray[np.float32]
    valid: NDArray[np.bool_]
    mkpts1: NDArray[np.float32]

@dataclass(frozen=True)
class PoseData(GeneratedData):
    threshold: float

    R_est: NDArray[np.float32]
    t_est: NDArray[np.float32]
    R_gt: NDArray[np.float32]
    t_gt: NDArray[np.float32]

@dataclass(frozen=True)
class MaskData(GeneratedData):
    mkpts0: NDArray
    mkpts1: NDArray
    masks0: NDArray
    masks1: NDArray

# --- Camada 3: Resultados da Análise ---


class AnalysisResult:
    """Classe base marcadora para resultados de análise."""
    pass


@dataclass(frozen=True)
class MatchResult(AnalysisResult):
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    threshold: float


@dataclass(frozen=True)
class ErrorResult(AnalysisResult):
    errors: list[float]

@dataclass(frozen=True)
class PoseErrorResult(AnalysisResult):
    rotation_errors: List[float]
    translation_errors: List[float]

    threshold: float