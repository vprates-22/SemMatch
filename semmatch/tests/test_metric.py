import torch
import numpy as np

import pytest
from SemMatch.statistics.metrics import UpdateData
from SemMatch.statistics.accuracy import Accuracy

@pytest.fixture(autouse=True)
def reset_metric():
    """Garante que a métrica esteja limpa antes de cada teste."""
    Accuracy.reset()
    yield
    Accuracy.reset()


def test_accuracy_single_update():
    # Cria dados de entrada simulados
    inliers = np.array([1, 1, 0, 1])  # 3 inliers
    mkpts0 = torch.rand((4, 2))      # 4 correspondências

    data = UpdateData(
        image0="img0.png",
        image1="img1.png",
        mkpts0=mkpts0,
        mkpts1=torch.rand((4, 2)),
        inliers=inliers,
        mask_hits=np.array([]),
        lpips_loss=[],
        valid_projections=np.array([]),
    )

    Accuracy.update(data)
    Accuracy.compute()

    expected = 3 / 4
    result = Accuracy.get_result()["Accuracy"]

    assert pytest.approx(result, 0.001) == expected
    assert len(Accuracy._raw_results) == 1
    assert pytest.approx(Accuracy._raw_results[0], 0.001) == expected


def test_accuracy_multiple_updates():
    # Primeira atualização
    data1 = UpdateData(
        image0="a",
        image1="b",
        mkpts0=torch.rand((5, 2)),
        mkpts1=torch.rand((5, 2)),
        inliers=np.array([1, 1, 1, 0, 0]),  # 3/5
        mask_hits=np.array([]),
        lpips_loss=[],
        valid_projections=np.array([]),
    )

    # Segunda atualização
    data2 = UpdateData(
        image0="c",
        image1="d",
        mkpts0=torch.rand((3, 2)),
        mkpts1=torch.rand((3, 2)),
        inliers=np.array([1, 0, 1]),  # 2/3
        mask_hits=np.array([]),
        lpips_loss=[],
        valid_projections=np.array([]),
    )

    Accuracy.update(data1)
    Accuracy.update(data2)
    Accuracy.compute()

    # Total inliers: 5, total correspondências: 8
    expected = 5 / 8
    result = Accuracy.get_result()["Accuracy"]

    assert pytest.approx(result, 0.001) == expected
    assert len(Accuracy._raw_results) == 2


def test_accuracy_zero_matches():
    data = UpdateData(
        image0="x",
        image1="y",
        mkpts0=torch.empty((0, 2)),  # zero correspondências
        mkpts1=torch.empty((0, 2)),
        inliers=np.array([]),
        mask_hits=np.array([]),
        lpips_loss=[],
        valid_projections=np.array([]),
    )

    Accuracy.update(data)
    Accuracy.compute()
    result = Accuracy.get_result()["Accuracy"]

    assert result == 0.0
    assert Accuracy._raw_results == []
