from unittest import mock

import pandas as pd
import pytest

from bt.core import Strategy, SecurityBase
from bt.algos.portfolio_ops import Rebalance, RebalanceOverTime


def test_rebalance():
    algo = Rebalance()
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 1}
    assert algo(s)
    assert s.value == 1000
    assert s.capital == 0
    c1 = s["c1"]
    assert c1.value == 1000
    assert c1.position == 10
    assert c1.weight == 1.0

    s.temp["weights"] = {"c2": 1}

    assert algo(s)
    assert s.value == 1000
    assert s.capital == 0
    c2 = s["c2"]
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 1000
    assert c2.position == 10
    assert c2.weight == 1.0


def test_rebalance_with_commissions():
    algo = Rebalance()

    s = Strategy("s")
    s.set_commissions(lambda q, p: 1)

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 1}

    assert algo(s)
    assert s.value == 999
    assert s.capital == 99
    c1 = s["c1"]
    assert c1.value == 900
    assert c1.position == 9
    assert c1.weight == 900 / 999.0

    s.temp["weights"] = {"c2": 1}

    assert algo(s)
    assert s.value == 997
    assert s.capital == 97
    c2 = s["c2"]
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 900
    assert c2.position == 9
    assert c2.weight == 900.0 / 997


def test_rebalance_with_cash():
    algo = Rebalance()

    s = Strategy("s")
    s.set_commissions(lambda q, p: 1)

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 1}
    s.temp["cash"] = 0.5

    assert algo(s)
    assert s.value == 999
    assert s.capital == 599
    c1 = s["c1"]
    assert c1.value == 400
    assert c1.position == 4
    assert c1.weight == 400.0 / 999

    s.temp["weights"] = {"c2": 1}
    s.temp["cash"] = 0.25

    assert algo(s)
    assert s.value == 997
    assert s.capital == 297
    c2 = s["c2"]
    assert c1.value == 0
    assert c1.position == 0
    assert c1.weight == 0
    assert c2.value == 700
    assert c2.position == 7
    assert c2.weight == 700.0 / 997


def test_rebalance_updatecount():
    algo = Rebalance()

    s = Strategy("s")
    s.use_integer_positions(False)

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4", "c5"], data=100)

    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 0.25, "c2": 0.25, "c3": 0.25, "c4": 0.25}

    update = SecurityBase.update
    SecurityBase._update_call_count = 0

    def side_effect(self, *args, **kwargs):
        SecurityBase._update_call_count += 1
        return update(self, *args, **kwargs)

    with mock.patch.object(SecurityBase, "update", side_effect):
        assert algo(s)

    assert s.value == 1000
    assert s.capital == 0
    assert SecurityBase._update_call_count == 8

    s.update(dts[1])
    s.temp["weights"] = {"c1": 0.5, "c2": 0.5}

    update = SecurityBase.update
    SecurityBase._update_call_count = 0

    def side_effect(self, *args, **kwargs):
        SecurityBase._update_call_count += 1
        return update(self, *args, **kwargs)

    with mock.patch.object(SecurityBase, "update", side_effect):
        assert algo(s)

    assert SecurityBase._update_call_count == 8

    s.update(dts[2])
    s.temp["weights"] = {"c1": 0.25, "c2": 0.25, "c3": 0.25, "c4": 0.25}

    update = SecurityBase.update
    SecurityBase._update_call_count = 0

    def side_effect(self, *args, **kwargs):
        SecurityBase._update_call_count += 1
        return update(self, *args, **kwargs)

    with mock.patch.object(SecurityBase, "update", side_effect):
        assert algo(s)

    assert SecurityBase._update_call_count == 6


def test_rebalance_over_time():
    target = mock.MagicMock()
    rb = mock.MagicMock()

    algo = RebalanceOverTime(n=2)
    algo._rb = rb

    target.temp = {"weights": {"a": 1, "b": 0}}

    a = mock.MagicMock()
    a.weight = 0.0
    b = mock.MagicMock()
    b.weight = 1.0
    target.children = {"a": a, "b": b}

    assert algo(target)
    w = target.temp["weights"]
    assert w == {"a": 0.5, "b": 0.5}

    called_tgt = rb.call_args[0][0]
    assert called_tgt.temp["weights"] == {"a": 0.5, "b": 0.5}

    a.weight = 0.5
    b.weight = 0.5
    target.temp = {}

    assert algo(target)
    assert target.temp["weights"] == {"a": 1.0, "b": 0.0}
    assert rb.call_count == 2

    a.weight = 1
    b.weight = 0
    target.temp = {}

    assert algo(target)
    assert rb.call_count == 2


def test_rebalance_rejects_invalid_weights_shape():
    algo = Rebalance()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = ["c1"]  # invalid shape
    assert not algo(s)


def test_rebalance_accepts_series_weights():
    algo = Rebalance()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = pd.Series({"c1": 1.0, "c2": None})
    assert algo(s)
    assert s["c1"].position == 10


@pytest.mark.parametrize("cash", [-0.1, 1.1, "bad"])
def test_rebalance_rejects_invalid_cash(cash):
    algo = Rebalance()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.adjust(1000)
    s.update(dts[0])

    s.temp["weights"] = {"c1": 1.0}
    s.temp["cash"] = cash
    assert not algo(s)


def test_rebalance_over_time_validates_n():
    with pytest.raises(TypeError, match="`n`"):
        RebalanceOverTime(n=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`n` must be > 0"):
        RebalanceOverTime(n=0)


def test_rebalance_over_time_rejects_invalid_weights_shape():
    algo = RebalanceOverTime(n=2)
    target = mock.MagicMock()
    target.temp = {"weights": ["a", "b"]}
    target.children = {}
    assert not algo(target)
