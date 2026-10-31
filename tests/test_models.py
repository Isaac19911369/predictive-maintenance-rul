import numpy as np
import pytest

from cmapss_rul.benchmark import split_units
from cmapss_rul.features import RegimeNormalizer
from cmapss_rul.metrics import rmse
from cmapss_rul.models import MODELS


@pytest.mark.parametrize("name", list(MODELS))
def test_models_learn_degradation(name, fleet):
    train, test, rul = fleet
    norm = RegimeNormalizer().fit(train)
    tr, te = norm.transform(train), norm.transform(test)
    fit, valid = split_units(tr, 0.2, seed=0)
    kwargs = {"epochs": 15} if name == "cnn1d" else {}
    model = MODELS[name](norm.sensors, window=20, cap=125, **kwargs).fit(fit, valid)
    pred = model.predict_last(te)
    assert pred.shape == (test["unit"].nunique(),)
    assert ((pred >= 0) & (pred <= 125)).all()
    baseline = rmse(np.minimum(rul, 125), np.full_like(rul, np.minimum(rul, 125).mean()))
    assert rmse(np.minimum(rul, 125), pred) < baseline, name
