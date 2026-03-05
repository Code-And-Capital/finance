from unittest import mock

import numpy as np
import pandas as pd
import pytest

from bt.core import Strategy
from bt.algos.weighting import (
    WeighEqually,
    WeighSpecified,
    ScaleWeights,
    WeighERC,
    WeighTarget,
    WeighInvVol,
    WeighMeanVar,
    WeighRandomly,
    LimitWeights,
    TargetVol,
)


def test_weight_equally():
    algo = WeighEqually()
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    weights = s.temp["weights"]
    assert weights["c1"] == 0.5
    assert weights["c2"] == 0.5


def test_weight_specified():
    algo = WeighSpecified(c1=0.6, c2=0.4)
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    weights = s.temp["weights"]
    assert weights["c1"] == 0.6
    assert weights["c2"] == 0.4


def test_scale_weights():
    s = Strategy("s")
    algo = ScaleWeights(-0.5)

    s.temp["weights"] = {"c1": 0.5, "c2": -0.4, "c3": 0}
    assert algo(s)
    assert s.temp["weights"] == {"c1": -0.25, "c2": 0.2, "c3": 0}


@mock.patch.object(WeighERC, "calc_erc_weights")
def test_weigh_erc(mock_erc):
    algo = WeighERC(lookback=pd.DateOffset(days=5))
    mock_erc.return_value = pd.Series({"c1": 0.3, "c2": 0.7})

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[4])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    rets = mock_erc.call_args[0][0]
    assert len(rets) == 4
    assert s.temp["weights"] == {"c1": 0.3, "c2": 0.7}


def test_weigh_target():
    algo = WeighTarget("target")
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    target = pd.DataFrame(index=dts[:2], columns=["c1", "c2"], data=0.5)
    target.loc[dts[1], "c1"] = 1.0
    target.loc[dts[1], "c2"] = 0.0

    s.setup(data, target=target)

    s.update(dts[0])
    assert algo(s)
    assert s.temp["weights"] == {"c1": 0.5, "c2": 0.5}

    s.update(dts[1])
    assert algo(s)
    assert s.temp["weights"] == {"c1": 1.0, "c2": 0.0}

    s.update(dts[2])
    assert not algo(s)


def test_weigh_inv_vol():
    algo = WeighInvVol(lookback=pd.DateOffset(days=5))

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    data.loc[dts[1], "c1"] = 105
    data.loc[dts[2], "c1"] = 95
    data.loc[dts[3], "c1"] = 105
    data.loc[dts[4], "c1"] = 95

    data.loc[dts[1], "c2"] = 100.1
    data.loc[dts[2], "c2"] = 99.9
    data.loc[dts[3], "c2"] = 100.1
    data.loc[dts[4], "c2"] = 99.9

    s.setup(data)
    s.update(dts[4])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    weights = s.temp["weights"]
    assert weights["c2"] > weights["c1"]
    assert weights["c1"] == pytest.approx(0.020, 3)
    assert weights["c2"] == pytest.approx(0.980, 3)


@mock.patch.object(WeighMeanVar, "calc_mean_var_weights")
def test_weigh_mean_var(mock_mv):
    algo = WeighMeanVar(lookback=pd.DateOffset(days=5))
    mock_mv.return_value = pd.Series({"c1": 0.3, "c2": 0.7})

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[4])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    rets = mock_mv.call_args[0][0]
    assert len(rets) == 4
    assert s.temp["weights"] == {"c1": 0.3, "c2": 0.7}


def test_weigh_randomly():
    s = Strategy("s")
    s.temp["selected"] = ["c1", "c2", "c3"]

    algo = WeighRandomly()
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 3
    assert sum(weights.values()) == 1.0

    algo = WeighRandomly((0.3, 0.5), 0.95)
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 3
    assert sum(weights.values()) == pytest.approx(0.95)
    for c in s.temp["selected"]:
        assert weights[c] <= 0.5
        assert weights[c] >= 0.3


def test_limit_weights():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.temp["weights"] = {"c1": 0.6, "c2": 0.2, "c3": 0.2}

    algo = LimitWeights(0.5)
    assert algo(s)
    assert s.temp["weights"] == {"c1": 0.5, "c2": 0.25, "c3": 0.25}

    s.temp["weights"] = {"c1": 0.6, "c2": 0.2, "c3": 0.2}
    algo = LimitWeights(0.3)
    assert algo(s)
    assert s.temp["weights"] == {}

    s.temp["weights"] = {"c1": 0.4, "c2": 0.3, "c3": 0.3}
    algo = LimitWeights(0.5)
    assert algo(s)
    assert s.temp["weights"] == {"c1": 0.4, "c2": 0.3, "c3": 0.3}


def test_target_vol():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=7)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    data.loc[dts[0], "c1"] = 95
    data.loc[dts[1], "c1"] = 105
    data.loc[dts[2], "c1"] = 95
    data.loc[dts[3], "c1"] = 105
    data.loc[dts[4], "c1"] = 95
    data.loc[dts[5], "c1"] = 105
    data.loc[dts[6], "c1"] = 95

    data.loc[dts[0], "c2"] = 99
    data.loc[dts[1], "c2"] = 101
    data.loc[dts[2], "c2"] = 99
    data.loc[dts[3], "c2"] = 101
    data.loc[dts[4], "c2"] = 99
    data.loc[dts[5], "c2"] = 101
    data.loc[dts[6], "c2"] = 99

    target_vol_algo = TargetVol(
        0.1,
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=1),
        covar_method="standard",
        annualization_factor=1,
    )

    s.setup(data)
    s.update(dts[6])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}

    assert target_vol_algo(s)
    weights = s.temp["weights"]
    assert np.isclose(weights["c2"], weights["c1"])

    unannualized_c2_weight = weights["c1"]

    target_vol_algo = TargetVol(
        0.1 * np.sqrt(252),
        lookback=pd.DateOffset(days=5),
        lag=pd.DateOffset(days=1),
        covar_method="standard",
        annualization_factor=252,
    )

    s.setup(data)
    s.update(dts[6])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}

    assert target_vol_algo(s)
    weights = s.temp["weights"]
    assert np.isclose(weights["c2"], weights["c1"])
    assert np.isclose(unannualized_c2_weight, weights["c2"])
