import numpy as np
import pytest

from cmapss_rul.metrics import nasa_score, rmse, summary


def test_rmse():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_nasa_score_is_asymmetric():
    early = nasa_score([50], [40])  # predicted 10 cycles too early
    late = nasa_score([50], [60])  # predicted 10 cycles too late
    assert early == pytest.approx(np.exp(10 / 13) - 1)
    assert late == pytest.approx(np.e - 1)
    assert late > early


def test_perfect_prediction():
    s = summary(np.arange(10), np.arange(10))
    assert s["rmse"] == 0 and s["nasa_score"] == 0 and s["late_pct"] == 0
