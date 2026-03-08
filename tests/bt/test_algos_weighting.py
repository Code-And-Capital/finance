from unittest import mock

import numpy as np
import pandas as pd
import pytest

from bt.core import Strategy
from bt.algos.weighting.core import WeightAlgo
from bt.algos.weighting.optimizers.base_optimizer import BaseOptimizer
from bt.algos.weighting import (
    WeightCurrent,
    WeightEqually,
    WeightFixed,
    ScaleWeights,
    WeightRiskParity,
    WeightFixedSchedule,
    WeightInvVol,
    WeightMarket,
    WeightMeanVar,
    WeightMinVar,
    WeightMaxDiversification,
    WeightRandomly,
    LimitDeltas,
    LimitWeights,
)


class _DummyWeightAlgo(WeightAlgo):
    def __call__(self, target):
        return True


def test_weight_algo_to_weight_dict_from_series():
    algo = _DummyWeightAlgo()
    weights = pd.Series({"c1": 0.4, "c2": np.nan, "c3": 0.6})
    out = algo._to_weight_dict(weights)
    assert out == {"c1": 0.4, "c3": 0.6}


def test_weight_algo_write_weights_and_history():
    algo = _DummyWeightAlgo(track_allocation_history=True)
    temp = {}
    now = pd.Timestamp("2020-01-01")
    out = algo._write_weights(
        temp,
        pd.Series({"c1": 0.25, "c2": 0.75}),
        now=now,
        record_history=True,
    )
    assert out == {"c1": 0.25, "c2": 0.75}
    assert temp["weights"] == {"c1": 0.25, "c2": 0.75}
    assert now in algo.allocation_history.index
    assert algo.allocation_history.loc[now, "c1"] == pytest.approx(0.25)
    assert algo.allocation_history.loc[now, "c2"] == pytest.approx(0.75)


def test_weight_equally():
    algo = WeightEqually()
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


def test_weight_equally_uses_set_and_solve():
    algo = WeightEqually()
    assert isinstance(algo.optimizer, BaseOptimizer)
    algo.optimizer.set_problem(["c1", "c2", "c3"])
    result = algo.optimizer.solve_problem()
    allocations = result["weights"]
    assert allocations == {
        "c1": pytest.approx(1 / 3),
        "c2": pytest.approx(1 / 3),
        "c3": pytest.approx(1 / 3),
    }


def test_weight_equally_empty_selected_produces_empty_weights():
    algo = WeightEqually()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = []

    assert algo(s)
    assert s.temp["weights"] == {}


def test_weight_equally_deduplicates_selected_names():
    algo = WeightEqually()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c1", "c2"]

    assert algo(s)
    assert s.temp["weights"] == {"c1": 0.5, "c2": 0.5}


def test_weight_equally_returns_false_for_non_list_selected():
    algo = WeightEqually()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = "c1"

    assert not algo(s)


def test_weight_current_requires_callable_first_weight_algo():
    with pytest.raises(TypeError, match="first_weight_algo"):
        WeightCurrent(first_weight_algo="not-callable")


def test_weight_current_runs_first_algo_once_then_uses_child_weights():
    first_algo = mock.Mock(
        side_effect=lambda target: target.temp.__setitem__("weights", {"c1": 1.0})
        or True
    )
    algo = WeightCurrent(first_weight_algo=first_algo)

    class _Target:
        pass

    target = _Target()
    target.temp = {"selected": ["c1", "c3"]}
    target.now = pd.Timestamp("2020-01-01")
    target.children = {}

    assert algo(target)
    assert first_algo.call_count == 1
    assert target.temp["weights"] == {"c1": 1.0}
    assert target.now in algo.allocation_history.index
    assert algo.allocation_history.loc[target.now, "c1"] == pytest.approx(1.0)

    target.now = pd.Timestamp("2020-01-02")
    target.children = {
        "c1": mock.Mock(weight=0.25),
        "c2": mock.Mock(weight=0.0),
        "c3": mock.Mock(weight=-0.10),
    }
    assert algo(target)
    assert first_algo.call_count == 1
    assert set(target.temp["weights"]) == {"c1", "c3"}
    assert sum(target.temp["weights"].values()) == pytest.approx(1.0)
    assert target.temp["weights"]["c1"] == pytest.approx(1.6666666666666667)
    assert target.temp["weights"]["c3"] == pytest.approx(-0.6666666666666666)
    assert target.now in algo.allocation_history.index
    assert algo.allocation_history.loc[target.now, "c1"] == pytest.approx(
        target.temp["weights"]["c1"]
    )
    assert algo.allocation_history.loc[target.now, "c3"] == pytest.approx(
        target.temp["weights"]["c3"]
    )


def test_weight_current_returns_false_when_first_algo_fails():
    first_algo = mock.Mock(return_value=False)
    algo = WeightCurrent(first_weight_algo=first_algo)

    class _Target:
        pass

    target = _Target()
    target.temp = {"selected": ["c1"]}
    target.now = pd.Timestamp("2020-01-01")
    target.children = {}

    assert not algo(target)
    assert not algo.has_run_first


def test_weight_current_returns_false_for_invalid_context():
    algo = WeightCurrent(first_weight_algo=mock.Mock(return_value=True))
    target = mock.MagicMock(spec=[])
    assert not algo(target)


def test_weight_current_returns_false_for_non_list_selected():
    algo = WeightCurrent(first_weight_algo=mock.Mock(return_value=True))

    class _Target:
        pass

    target = _Target()
    target.temp = {"selected": "c1"}
    target.now = pd.Timestamp("2020-01-01")
    target.children = {}
    assert not algo(target)


def test_weight_current_first_algo_output_is_not_re_sliced_or_normalized():
    first_algo = mock.Mock(
        side_effect=lambda target: target.temp.__setitem__(
            "weights", {"c1": 0.7, "c2": 0.3}
        )
        or True
    )
    algo = WeightCurrent(first_weight_algo=first_algo)

    class _Target:
        pass

    target = _Target()
    target.temp = {"selected": ["c1"]}
    target.now = pd.Timestamp("2020-01-01")
    target.children = {}

    assert algo(target)
    assert target.temp["weights"] == {"c1": 0.7, "c2": 0.3}


def test_weight_current_returns_false_when_first_algo_sets_invalid_weights_payload():
    first_algo = mock.Mock(
        side_effect=lambda target: target.temp.__setitem__("weights", "bad") or True
    )
    algo = WeightCurrent(first_weight_algo=first_algo)

    class _Target:
        pass

    target = _Target()
    target.temp = {"selected": ["c1"]}
    target.now = pd.Timestamp("2020-01-01")
    target.children = {}

    assert not algo(target)
    assert not algo.has_run_first


def test_weight_current_returns_false_when_children_not_dict_after_first_run():
    first_algo = mock.Mock(
        side_effect=lambda target: target.temp.__setitem__("weights", {"c1": 1.0})
        or True
    )
    algo = WeightCurrent(first_weight_algo=first_algo)

    class _Target:
        pass

    target = _Target()
    target.temp = {"selected": ["c1"]}
    target.now = pd.Timestamp("2020-01-01")
    target.children = {}
    assert algo(target)

    target.now = pd.Timestamp("2020-01-02")
    target.children = ["not", "a", "dict"]
    assert not algo(target)


def test_weight_equally_returns_false_when_temp_missing():
    algo = WeightEqually()
    target = mock.MagicMock(spec=[])
    assert not algo(target)


def test_weight_equally_records_allocation_history_with_loc():
    algo = WeightEqually()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    assert dts[0] in algo.allocation_history.index
    assert algo.allocation_history.loc[dts[0], "c1"] == pytest.approx(0.5)
    assert algo.allocation_history.loc[dts[0], "c2"] == pytest.approx(0.5)

    s.update(dts[1])
    s.temp["selected"] = ["c1"]
    assert algo(s)
    assert dts[1] in algo.allocation_history.index
    assert algo.allocation_history.loc[dts[1], "c1"] == pytest.approx(1.0)


def test_weight_specified():
    algo = WeightFixed(c1=0.6, c2=0.4)
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    weights = s.temp["weights"]
    assert weights["c1"] == 0.6
    assert weights["c2"] == 0.4


def test_weight_specified_intersects_with_selected():
    algo = WeightFixed(c1=0.6, c2=0.4, c3=0.0)
    s = Strategy("s")
    s.temp["selected"] = ["c2"]

    assert algo(s)
    assert s.temp["weights"] == {"c2": 1.0}


def test_weight_specified_records_allocation_history():
    algo = WeightFixed(c1=0.6, c2=0.4)
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    assert dts[0] in algo.allocation_history.index
    assert algo.allocation_history.loc[dts[0], "c1"] == pytest.approx(0.6)
    assert algo.allocation_history.loc[dts[0], "c2"] == pytest.approx(0.4)


def test_weight_specified_returns_false_for_non_list_selected():
    algo = WeightFixed(c1=0.6, c2=0.4)
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = "c1"

    assert not algo(s)


def test_weight_specified_empty_overlap_writes_empty_weights_and_history_row():
    algo = WeightFixed(c1=0.6, c2=0.4)
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c3"]

    assert algo(s)
    assert s.temp["weights"] == {}
    assert dts[0] in algo.allocation_history.index


def test_scale_weights():
    s = Strategy("s")
    algo = ScaleWeights(-0.5)

    s.temp["weights"] = {"c1": 0.5, "c2": -0.4, "c3": 0}
    assert algo(s)
    assert s.temp["weights"] == {"c1": -0.25, "c2": 0.2, "c3": 0}


def test_scale_weights_records_allocation_history():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}
    algo = ScaleWeights(0.5)

    assert algo(s)
    assert s.temp["weights"] == {"c1": 0.25, "c2": 0.25}
    assert dts[0] in algo.allocation_history.index
    assert algo.allocation_history.loc[dts[0], "c1"] == pytest.approx(0.25)
    assert algo.allocation_history.loc[dts[0], "c2"] == pytest.approx(0.25)


def test_scale_weights_returns_false_when_temp_missing():
    algo = ScaleWeights(0.5)
    target = mock.MagicMock(spec=[])
    assert not algo(target)


def test_scale_weights_returns_false_for_invalid_weights_payload():
    s = Strategy("s")
    algo = ScaleWeights(0.5)
    s.temp["weights"] = "invalid"

    assert not algo(s)


def test_scale_weights_empty_weights_writes_empty_and_records_row():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["weights"] = {}
    algo = ScaleWeights(2.0)

    assert algo(s)
    assert s.temp["weights"] == {}
    assert dts[0] in algo.allocation_history.index


def test_scale_weights_raises_for_non_numeric_scale():
    with pytest.raises(TypeError, match="scale"):
        ScaleWeights("bad")


def test_weigh_erc():
    algo = WeightRiskParity()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]
    s.temp["covariance"] = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.0004]],
        index=["c1", "c2"],
        columns=["c1", "c2"],
    )

    assert algo(s)
    weights = s.temp["weights"]
    assert set(weights.keys()) == {"c1", "c2"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["c2"] > weights["c1"]


def test_weigh_target():
    algo = WeightFixedSchedule("target")
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


def test_weigh_target_intersects_with_selected():
    algo = WeightFixedSchedule("target")
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    target = pd.DataFrame(index=dts, columns=["c1", "c2"], data=[[0.6, 0.4]])

    s.setup(data, target=target)
    s.update(dts[0])
    s.temp["selected"] = ["c1"]

    assert algo(s)
    assert s.temp["weights"] == {"c1": 1.0}


def test_weigh_target_records_allocation_history_when_date_present():
    algo = WeightFixedSchedule("target")
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    target = pd.DataFrame(index=dts[:1], columns=["c1", "c2"], data=0.5)
    s.setup(data, target=target)

    s.update(dts[0])
    assert algo(s)
    assert dts[0] in algo.allocation_history.index
    assert algo.allocation_history.loc[dts[0], "c1"] == pytest.approx(0.5)
    assert algo.allocation_history.loc[dts[0], "c2"] == pytest.approx(0.5)


def test_weight_target_accepts_dataframe_source():
    dts = pd.date_range("2010-01-01", periods=1)
    weights_df = pd.DataFrame([[0.2, 0.8]], index=dts, columns=["c1", "c2"])
    algo = WeightFixedSchedule(weights_df)
    s = Strategy("s")
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    assert s.temp["weights"] == {"c1": pytest.approx(0.2), "c2": pytest.approx(0.8)}


def test_weight_target_returns_false_for_non_list_selected():
    dts = pd.date_range("2010-01-01", periods=1)
    weights_df = pd.DataFrame([[0.5, 0.5]], index=dts, columns=["c1", "c2"])
    algo = WeightFixedSchedule(weights_df)
    s = Strategy("s")
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = "c1"

    assert not algo(s)


def test_weight_target_returns_false_when_source_key_missing():
    algo = WeightFixedSchedule("missing_weights")
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    assert not algo(s)


def test_weight_target_empty_overlap_writes_empty_weights_and_history_row():
    dts = pd.date_range("2010-01-01", periods=1)
    weights_df = pd.DataFrame([[0.6, 0.4]], index=dts, columns=["c1", "c2"])
    algo = WeightFixedSchedule(weights_df)
    s = Strategy("s")
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c3"]

    assert algo(s)
    assert s.temp["weights"] == {}
    assert dts[0] in algo.allocation_history.index


def test_weight_min_var():
    algo = WeightMinVar()

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]
    s.temp["covariance"] = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]],
        index=["c1", "c2"],
        columns=["c1", "c2"],
    )

    assert algo(s)
    weights = s.temp["weights"]
    assert set(weights.keys()) == {"c1", "c2"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["c2"] > weights["c1"]


def test_weight_max_diversification():
    algo = WeightMaxDiversification()

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]
    s.temp["covariance"] = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]],
        index=["c1", "c2"],
        columns=["c1", "c2"],
    )

    assert algo(s)
    weights = s.temp["weights"]
    assert set(weights.keys()) == {"c1", "c2"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["c2"] > weights["c1"]


def test_weigh_inv_vol():
    algo = WeightInvVol()

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]
    s.temp["covariance"] = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.0004]],
        index=["c1", "c2"],
        columns=["c1", "c2"],
    )

    assert algo(s)
    weights = s.temp["weights"]
    assert weights["c2"] > weights["c1"]
    assert weights["c1"] == pytest.approx(0.091, 3)
    assert weights["c2"] == pytest.approx(0.909, 3)


def test_weight_inv_vol_set_and_solve():
    algo = WeightInvVol()
    assert isinstance(algo.optimizer, BaseOptimizer)
    cov = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]],
        index=["c1", "c2"],
        columns=["c1", "c2"],
    )
    algo.optimizer.set_problem(cov)
    result = algo.optimizer.solve_problem()
    weights = result["weights"]
    assert set(weights.keys()) == {"c1", "c2"}
    assert weights["c1"] == pytest.approx(1.0 / 3.0)
    assert weights["c2"] == pytest.approx(2.0 / 3.0)


def test_weigh_inv_vol_returns_false_for_invalid_context():
    algo = WeightInvVol()
    target = mock.MagicMock(spec=[])
    assert not algo(target)


def test_weight_inv_vol_empty_selected_records_history_row():
    algo = WeightInvVol()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = []
    s.temp["covariance"] = pd.DataFrame([[1.0]], index=["c1"], columns=["c1"])

    assert algo(s)
    assert s.temp["weights"] == {}
    assert dts[0] in algo.allocation_history.index


def test_weight_inv_vol_raises_on_invalid_covariance_type():
    algo = WeightInvVol()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]
    s.temp["covariance"] = "not-a-dataframe"

    with pytest.raises(TypeError, match="covariance"):
        algo(s)


def test_weight_market():
    algo = WeightMarket("market_caps")
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    market_caps = pd.DataFrame(
        [[100.0, 300.0]],
        index=dts,
        columns=["c1", "c2"],
    )

    s.setup(data, market_caps=market_caps)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    assert s.temp["weights"]["c1"] == pytest.approx(0.25)
    assert s.temp["weights"]["c2"] == pytest.approx(0.75)


def test_weight_market_set_and_solve():
    algo = WeightMarket()
    assert isinstance(algo.optimizer, BaseOptimizer)
    caps = pd.Series({"c1": 10.0, "c2": 30.0})
    algo.optimizer.set_problem(caps, ["c1", "c2"])
    result = algo.optimizer.solve_problem()
    weights = result["weights"]
    assert weights["c1"] == pytest.approx(0.25)
    assert weights["c2"] == pytest.approx(0.75)


def test_weight_market_empty_selected_records_history_row():
    algo = WeightMarket("market_caps")
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    market_caps = pd.DataFrame([[100.0]], index=dts, columns=["c1"])
    s.setup(data, market_caps=market_caps)
    s.update(dts[0])
    s.temp["selected"] = []

    assert algo(s)
    assert s.temp["weights"] == {}
    assert dts[0] in algo.allocation_history.index


def test_weight_market_returns_false_for_non_list_selected():
    algo = WeightMarket("market_caps")
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    market_caps = pd.DataFrame([[100.0]], index=dts, columns=["c1"])
    s.setup(data, market_caps=market_caps)
    s.update(dts[0])
    s.temp["selected"] = "c1"

    assert not algo(s)


def test_weight_market_returns_false_when_now_missing_in_market_caps():
    algo = WeightMarket("market_caps")
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    market_caps = pd.DataFrame(
        [[100.0]],
        index=[pd.Timestamp("2010-01-02")],
        columns=["c1"],
    )
    s.setup(data, market_caps=market_caps)
    s.update(dts[0])
    s.temp["selected"] = ["c1"]

    assert not algo(s)


def test_weight_market_raises_on_invalid_market_cap_data_type():
    algo = WeightMarket("market_caps")
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data, market_caps="not-a-dataframe")
    s.update(dts[0])
    s.temp["selected"] = ["c1"]

    with pytest.raises(TypeError, match="must be a DataFrame"):
        algo(s)


def test_market_weight_optimizer_filters_to_selected_before_normalizing():
    algo = WeightMarket()
    caps = pd.Series({"c1": 10.0, "c2": 30.0, "c3": 60.0})
    algo.optimizer.set_problem(caps, ["c1", "c2"])
    result = algo.optimizer.solve_problem()
    weights = result["weights"]
    assert set(weights.keys()) == {"c1", "c2"}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["c1"] == pytest.approx(0.25)
    assert weights["c2"] == pytest.approx(0.75)


def test_market_weight_optimizer_drops_nonpositive_and_nan_caps():
    algo = WeightMarket()
    caps = pd.Series({"c1": 10.0, "c2": np.nan, "c3": 0.0, "c4": -5.0})
    algo.optimizer.set_problem(caps, ["c1", "c2", "c3", "c4"])
    result = algo.optimizer.solve_problem()
    assert result["weights"] == {"c1": pytest.approx(1.0)}


@mock.patch("bt.algos.weighting.mean_variance.MeanVarianceOptimizer.solve_problem")
def test_weigh_mean_var(mock_solve):
    algo = WeightMeanVar()
    mock_solve.return_value = {
        "weights": {"c1": 0.3, "c2": 0.7},
        "success": True,
        "status": "optimal",
        "message": "ok",
    }

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=5)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    s.setup(data)
    s.update(dts[4])
    s.temp["selected"] = ["c1", "c2"]
    s.temp["expected_returns"] = pd.Series({"c1": 0.01, "c2": 0.02})
    s.temp["covariance"] = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]],
        index=["c1", "c2"],
        columns=["c1", "c2"],
    )

    assert algo(s)
    assert algo.optimizer.problem_data["universe"] == ["c1", "c2"]
    assert algo.optimizer.problem_data["asset_count"] == 2
    assert s.temp["weights"] == {"c1": 0.3, "c2": 0.7}


def test_weigh_randomly():
    s = Strategy("s")
    s.temp["selected"] = ["c1", "c2", "c3"]

    algo = WeightRandomly()
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 3
    assert sum(weights.values()) == pytest.approx(1.0)

    algo = WeightRandomly((0.3, 0.5))
    assert algo(s)
    weights = s.temp["weights"]
    assert len(weights) == 3
    assert sum(weights.values()) == pytest.approx(1.0)
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
    assert s.temp["weights"] == {"c1": 0.6, "c2": 0.2, "c3": 0.2}

    s.temp["weights"] = {"c1": 0.4, "c2": 0.3, "c3": 0.3}
    algo = LimitWeights(0.5)
    assert algo(s)
    assert s.temp["weights"] == {"c1": 0.4, "c2": 0.3, "c3": 0.3}


def test_limit_weights_records_allocation_history():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["weights"] = {"c1": 0.6, "c2": 0.2, "c3": 0.2}

    algo = LimitWeights(0.5)
    assert algo(s)
    assert dts[0] in algo.allocation_history.index
    assert algo.allocation_history.loc[dts[0], "c1"] == pytest.approx(0.5)
    assert algo.allocation_history.loc[dts[0], "c2"] == pytest.approx(0.25)
    assert algo.allocation_history.loc[dts[0], "c3"] == pytest.approx(0.25)


def test_limit_weights_returns_false_for_invalid_context():
    algo = LimitWeights(0.5)
    target = mock.MagicMock(spec=[])
    assert not algo(target)


def test_limit_weights_writes_empty_when_weights_missing():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    algo = LimitWeights(0.5)

    assert algo(s)
    assert s.temp["weights"] == {}
    assert dts[0] in algo.allocation_history.index


def test_limit_weights_raises_for_invalid_limit():
    with pytest.raises(ValueError, match="0 < limit <= 1"):
        LimitWeights(0.0)
    with pytest.raises(ValueError, match="0 < limit <= 1"):
        LimitWeights(1.5)


def test_limit_weights_multi_iteration_redistribution_converges():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    # First pass caps c1, redistributes enough to push c2 above the cap.
    s.temp["weights"] = {"c1": 0.7, "c2": 0.2, "c3": 0.07, "c4": 0.03}

    algo = LimitWeights(0.4)
    assert algo(s)
    out = s.temp["weights"]
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["c1"] == pytest.approx(0.4)
    assert out["c2"] == pytest.approx(0.4)
    assert out["c3"] == pytest.approx(0.14)
    assert out["c4"] == pytest.approx(0.06)
    assert all(weight <= 0.4 + 1e-12 for weight in out.values())


def test_limit_deltas_global_limit_clips_and_normalizes():
    class _Target:
        pass

    target = _Target()
    target.now = pd.Timestamp("2010-01-01")
    target.temp = {"weights": {"c1": 0.7, "c2": 0.2, "c3": 0.1}}
    target.children = {
        "c1": mock.Mock(weight=0.4),
        "c2": mock.Mock(weight=0.3),
        "c3": mock.Mock(weight=0.3),
    }

    algo = LimitDeltas(0.1)
    assert algo(target)
    out = target.temp["weights"]
    assert sum(out.values()) == pytest.approx(1.0)
    for name, child in target.children.items():
        assert abs(out[name] - child.weight) <= 0.1 + 1e-9
    assert target.now in algo.allocation_history.index


def test_limit_deltas_empty_weights_writes_empty_and_records_history():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["weights"] = {}

    algo = LimitDeltas(0.1)
    assert algo(s)
    assert s.temp["weights"] == {}
    assert dts[0] in algo.allocation_history.index


def test_limit_deltas_returns_false_for_invalid_context_or_children():
    algo = LimitDeltas(0.1)
    target = mock.MagicMock(spec=[])
    assert not algo(target)

    class _Target:
        pass

    target2 = _Target()
    target2.temp = {"weights": {"c1": 1.0}}
    target2.now = pd.Timestamp("2020-01-01")
    target2.children = []
    assert not algo(target2)


def test_limit_deltas_raises_for_invalid_limit():
    with pytest.raises(ValueError, match="must be >= 0"):
        LimitDeltas(-0.1)


def test_limit_deltas_relaxes_effective_limit_to_keep_sum_to_one():
    class _Target:
        pass

    target = _Target()
    target.now = pd.Timestamp("2010-01-01")
    target.temp = {"weights": {"c1": 0.6, "c2": 0.4}}
    target.children = {
        "c1": mock.Mock(weight=0.0),
        "c2": mock.Mock(weight=0.0),
    }

    algo = LimitDeltas(0.1)
    assert algo(target)
    out = target.temp["weights"]
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["c1"] == pytest.approx(0.5)
    assert out["c2"] == pytest.approx(0.5)
