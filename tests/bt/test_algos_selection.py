import random
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from bt.core import Strategy
from bt.algos.selection import (
    SelectActive,
    SelectAll,
    SelectHasData,
    SelectN,
    SelectRandomly,
    SelectThese,
    SelectWhere,
)


def test_select_all():
    algo = SelectAll()

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # make sure don't keep nan
    s.update(dts[1])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = SelectAll(include_no_data=True)

    assert algo2(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = SelectAll(include_negative=True)

    assert algo3(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


@pytest.mark.parametrize("now_value", [None, "not-a-date", pd.Timestamp("1999-01-01")])
def test_select_all_handles_invalid_or_missing_now(now_value):
    algo = SelectAll()
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.now = now_value

    assert not algo(s)
    assert "selected" not in s.temp


def test_select_randomly_large_n_behaves_like_select_all():
    algo = SelectRandomly(n=9999)  # Behaves like SelectAll for small universes

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # make sure don't keep nan
    s.update(dts[1])

    assert algo(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = SelectRandomly(n=9999, include_no_data=True)

    assert algo2(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = SelectRandomly(n=9999, include_negative=True)

    assert algo3(s)
    selected = s.temp.pop("selected")
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_randomly():

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    data.loc[dts[0], "c1"] = np.nan
    data.loc[dts[0], "c2"] = 95
    data.loc[dts[0], "c3"] = -5

    s.setup(data)
    s.update(dts[0])

    algo = SelectRandomly(n=1)
    assert algo(s)
    assert s.temp.pop("selected") == ["c2"]

    random.seed(1000)
    algo = SelectRandomly(n=1, include_negative=True)
    assert algo(s)
    assert s.temp.pop("selected") == ["c3"]

    random.seed(1009)
    algo = SelectRandomly(n=1, include_no_data=True)
    assert algo(s)
    assert s.temp.pop("selected") == ["c1"]

    random.seed(1009)
    # If selected already set, it will further filter it
    s.temp["selected"] = ["c2"]
    algo = SelectRandomly(n=1, include_no_data=True)
    assert algo(s)
    assert s.temp.pop("selected") == ["c2"]


def test_select_randomly_validates_n():
    with pytest.raises(TypeError, match="`n` must be"):
        SelectRandomly(n=1.5)

    with pytest.raises(ValueError, match="`n` must be >= 0"):
        SelectRandomly(n=-1)


def test_select_randomly_filters_missing_universe_members():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "missing"]

    algo = SelectRandomly(n=9999, include_no_data=True)
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


@pytest.mark.parametrize("now_value", [None, "not-a-date", pd.Timestamp("1999-01-01")])
def test_select_randomly_handles_invalid_or_missing_now(now_value):
    algo = SelectRandomly(n=1)
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.now = now_value

    assert not algo(s)
    assert "selected" not in s.temp


def test_select_randomly_handles_malformed_state():
    algo = SelectRandomly(n=1)

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = []
    target.universe = pd.DataFrame(index=pd.date_range("2010-01-01", periods=1))
    target.now = pd.Timestamp("2010-01-01")
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = {}
    target.universe = []
    target.now = pd.Timestamp("2010-01-01")
    assert not algo(target)


def test_select_randomly_n_zero_selects_none():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    algo = SelectRandomly(n=0)
    assert algo(s)
    assert s.temp["selected"] == []


def test_select_randomly_n_greater_than_eligible_selects_all():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    algo = SelectRandomly(n=10)
    assert algo(s)
    assert set(s.temp["selected"]) == {"c1", "c2"}


def test_select_randomly_empty_candidates():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = []

    algo = SelectRandomly(n=1)
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_these():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    s.setup(data)
    s.update(dts[0])

    algo = SelectThese(["c1", "c2"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    algo = SelectThese(["c1"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c1" in selected

    # make sure don't keep nan
    s.update(dts[1])
    s.temp.pop("selected", None)

    algo = SelectThese(["c1", "c2"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # if specify include_no_data then 2
    algo2 = SelectThese(["c1", "c2"], include_no_data=True)
    s.temp.pop("selected", None)

    assert algo2(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    algo3 = SelectThese(["c1", "c2"], include_negative=True)
    s.temp.pop("selected", None)

    assert algo3(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_these_accepts_scalar_string():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    algo = SelectThese("c1")
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_these_uses_existing_selected_pool():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c2"]

    algo = SelectThese(["c1", "c2"], include_no_data=True)
    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_these_calls_select_all_when_selected_empty_or_missing():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    algo = SelectThese(["c1", "c2"], include_no_data=True)
    assert algo(s)
    assert set(s.temp["selected"]) == {"c1", "c2"}

    s.temp["selected"] = []
    assert algo(s)
    assert set(s.temp["selected"]) == {"c1", "c2"}


def test_select_these_deduplicates_tickers():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    algo = SelectThese(["c2", "c1", "c2", "c1"], include_no_data=True)
    assert algo(s)
    assert set(s.temp["selected"]) == {"c1", "c2"}


def test_select_these_ignores_tickers_not_in_universe():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    algo = SelectThese(["missing", "c1"], include_no_data=True)
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


@pytest.mark.parametrize("now_value", [None, "not-a-date", pd.Timestamp("1999-01-01")])
def test_select_these_handles_invalid_or_missing_now(now_value):
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.now = now_value

    algo = SelectThese(["c1"])
    assert not algo(s)
    assert "selected" not in s.temp


def test_select_these_handles_malformed_state():
    algo = SelectThese(["c1"])

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = []
    target.universe = pd.DataFrame(index=pd.date_range("2010-01-01", periods=1))
    target.now = pd.Timestamp("2010-01-01")
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = {}
    target.universe = []
    target.now = pd.Timestamp("2010-01-01")
    assert not algo(target)


def test_select_these_validates_tickers_input():
    with pytest.raises(ValueError, match="must not be empty"):
        SelectThese([])

    with pytest.raises(ValueError, match="must not contain empty strings"):
        SelectThese(["c1", ""])

    with pytest.raises(TypeError, match="must contain only strings"):
        SelectThese(["c1", 123])


def test_select_where_all():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = np.nan
    data.loc[dts[1], "c2"] = 95
    data.loc[dts[2], "c1"] = -5

    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)

    s.setup(data, where=where)
    s.update(dts[0])

    algo = SelectWhere("where")
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # make sure don't keep nan
    s.update(dts[1])

    algo = SelectThese(["c1", "c2"])
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected
    del s.temp["selected"]

    # if specify include_no_data then 2
    algo2 = SelectWhere("where", include_no_data=True)

    assert algo2(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected

    # behavior on negative prices
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected
    del s.temp["selected"]

    algo3 = SelectWhere("where", include_negative=True)

    assert algo3(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_where():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)
    where.loc[dts[1]] = False
    where.loc[dts[2], "c1"] = False

    algo = SelectWhere("where")

    s.setup(data, where=where)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected
    del s.temp["selected"]

    s.update(dts[1])
    assert algo(s)
    assert s.temp["selected"] == []
    del s.temp["selected"]

    s.update(dts[2])
    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_where_legacy():
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)
    where.loc[dts[1]] = False
    where.loc[dts[2], "c1"] = False

    algo = SelectWhere(where)

    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected
    del s.temp["selected"]

    s.update(dts[1])
    assert algo(s)
    assert s.temp["selected"] == []
    del s.temp["selected"]

    s.update(dts[2])
    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_where_validates_signal_type():
    with pytest.raises(TypeError, match="`signal` must be"):
        SelectWhere(signal=123)


def test_select_where_ignores_selected_names_not_in_signal():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    where = pd.DataFrame(index=dts, columns=["c1"], data=True)
    s.setup(data, where=where)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]

    algo = SelectWhere("where")
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_where_normalizes_row_truthiness():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    where = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=0)
    where.loc[dts[0], "c1"] = 1
    where.loc[dts[0], "c2"] = 0
    where.loc[dts[0], "c3"] = pd.NA
    s.setup(data, where=where)
    s.update(dts[0])

    algo = SelectWhere("where")
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_where_uses_existing_selected_pool():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)
    s.setup(data, where=where)
    s.update(dts[0])
    s.temp["selected"] = ["c2"]

    algo = SelectWhere("where")
    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_where_calls_select_all_when_selected_empty_or_missing():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    where = pd.DataFrame(index=dts, columns=["c1", "c2"], data=True)
    s.setup(data, where=where)
    s.update(dts[0])

    algo = SelectWhere("where")
    assert algo(s)
    assert set(s.temp["selected"]) == {"c1", "c2"}

    s.temp["selected"] = []
    assert algo(s)
    assert set(s.temp["selected"]) == {"c1", "c2"}


@pytest.mark.parametrize("now_value", [None, "not-a-date", pd.Timestamp("1999-01-01")])
def test_select_where_handles_invalid_or_missing_now(now_value):
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    where = pd.DataFrame(index=dts, columns=["c1"], data=True)
    s.setup(data, where=where)
    s.now = now_value

    algo = SelectWhere("where")
    assert not algo(s)
    assert "selected" not in s.temp


def test_select_where_handles_malformed_state():
    algo = SelectWhere("where")

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = []
    target.universe = pd.DataFrame(index=pd.date_range("2010-01-01", periods=1))
    target.now = pd.Timestamp("2010-01-01")
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = {}
    target.universe = []
    target.now = pd.Timestamp("2010-01-01")
    assert not algo(target)


def test_select_where_returns_false_when_now_not_in_signal_index():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    where = pd.DataFrame(index=[dts[0]], columns=["c1"], data=True)
    s.setup(data, where=where)
    s.update(dts[1])
    s.temp["selected"] = ["c1"]

    algo = SelectWhere("where")
    assert not algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_n():
    algo = SelectN(n=1, sort_descending=True)

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[2], "c1"] = 105
    data.loc[dts[2], "c2"] = 95

    s.setup(data)
    s.update(dts[2])
    first = data.iloc[0]
    last = data.iloc[-1]
    s.temp["stat"] = (last / first) - 1
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c1" in selected

    algo = SelectN(n=1, sort_descending=False)
    s.temp["selected"] = ["c1", "c2"]
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected

    # if n is larger than available, select all available
    algo = SelectN(n=3, sort_descending=False)
    s.temp["selected"] = ["c1", "c2"]
    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 2
    assert "c1" in selected
    assert "c2" in selected


def test_select_n_perc():
    algo = SelectN(n=0.5, sort_descending=True)

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[2], "c1"] = 105
    data.loc[dts[2], "c2"] = 95

    s.setup(data)
    s.update(dts[2])
    s.temp["stat"] = data.calc_total_return()
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c1" in selected


def test_select_n_handles_missing_stat():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    algo = SelectN(n=1)
    s.temp["selected"] = ["c1"]
    assert not algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_n_validates_n_type():
    with pytest.raises(TypeError, match="numeric"):
        SelectN(n="1")

    with pytest.raises(TypeError, match="integer"):
        SelectN(n=1.2)

    with pytest.raises(TypeError, match="numeric"):
        SelectN(n=True)

    with pytest.raises(TypeError, match="stat_key"):
        SelectN(n=1, stat_key=1)


def test_select_n_validates_bool_flags():
    with pytest.raises(TypeError, match="sort_descending"):
        SelectN(n=1, sort_descending=1)


def test_select_n_percentage_uses_floor_with_min_one():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": 3.0, "c2": 2.0, "c3": 1.0})
    s.temp["selected"] = ["c1", "c2", "c3"]
    algo = SelectN(n=0.5)

    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_n_drops_non_finite_values():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": np.inf, "c2": 2.0, "c3": -np.inf})
    s.temp["selected"] = ["c1", "c2", "c3"]
    algo = SelectN(n=1)

    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_n_handles_missing_or_invalid_temp():
    algo = SelectN(n=1)

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = []
    assert not algo(target)


def test_select_n_requires_selected():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": 1.0, "c2": 2.0})
    algo = SelectN(n=1)
    assert not algo(s)


def test_select_n_uses_custom_stat_key():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["custom_stat"] = pd.Series({"c1": 1.0, "c2": 3.0})
    s.temp["selected"] = ["c1", "c2"]
    algo = SelectN(n=1, stat_key="custom_stat")

    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_active_uses_select_all_when_selected_missing_or_empty():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.perm["closed"] = {"c1"}
    s.perm["rolled"] = {"c2"}

    algo = SelectActive()
    assert algo(s)
    assert s.temp["selected"] == ["c3"]

    s.temp["selected"] = []
    assert algo(s)
    assert s.temp["selected"] == ["c3"]


def test_select_active_handles_malformed_temp_or_perm():
    algo = SelectActive()

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = {}
    target.perm = []
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = []
    target.perm = {}
    assert not algo(target)


def test_select_active_accepts_non_set_inactive_collections():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2", "c3", "c4"]
    s.perm["rolled"] = ["c2"]
    s.perm["closed"] = ("c4",)

    algo = SelectActive()
    assert algo(s)
    assert s.temp["selected"] == ["c1", "c3"]


def test_select_has_data():
    algo = SelectHasData(lookback=pd.DateOffset(days=3))

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=10)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[0], "c1"] = np.nan
    data.loc[dts[1], "c1"] = np.nan

    s.setup(data)
    s.update(dts[2])

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 1
    assert "c2" in selected


def test_select_has_data_preselected():
    algo = SelectHasData(lookback=pd.DateOffset(days=3))

    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=3)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[0], "c1"] = np.nan
    data.loc[dts[1], "c1"] = np.nan

    s.setup(data)
    s.update(dts[2])
    s.temp["selected"] = ["c1"]

    assert algo(s)
    selected = s.temp["selected"]
    assert len(selected) == 0


def test_select_has_data_uses_select_all_when_selected_missing_or_empty():
    algo = SelectHasData(lookback=pd.DateOffset(days=1))
    s = Strategy("s")

    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    data.loc[dts[1], "c1"] = float("nan")

    s.setup(data)
    s.update(dts[1])

    assert algo(s)
    assert s.temp["selected"] == ["c2"]

    s.temp["selected"] = []
    assert algo(s)
    assert s.temp["selected"] == ["c2"]


def test_select_has_data_handles_malformed_state():
    algo = SelectHasData(lookback=pd.DateOffset(days=1))

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = []
    target.universe = pd.DataFrame(index=pd.date_range("2010-01-01", periods=1))
    target.now = pd.Timestamp("2010-01-01")
    assert not algo(target)

    target = mock.MagicMock()
    target.temp = {}
    target.universe = []
    target.now = pd.Timestamp("2010-01-01")
    assert not algo(target)


@pytest.mark.parametrize("now_value", [None, "not-a-date", pd.Timestamp("1999-01-01")])
def test_select_has_data_handles_invalid_or_missing_now(now_value):
    algo = SelectHasData(lookback=pd.DateOffset(days=1))
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.now = now_value

    assert not algo(s)
    assert "selected" not in s.temp
