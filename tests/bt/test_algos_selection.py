import random
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from bt.core import Strategy
from bt.algos.selection import (
    AddSecurity,
    SectorDoubleSort,
    RemoveSecurities,
    SelectActive,
    SelectAll,
    SelectHasData,
    SelectIsOpen,
    SelectN,
    SelectQuantile,
    SelectRandomly,
    SelectSector,
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


def test_select_n_uses_select_all_when_selected_missing():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": 1.0, "c2": 2.0})
    algo = SelectN(n=1)
    assert algo(s)
    assert s.temp["selected"] == ["c2"]


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


def test_select_n_updates_stats():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": 3.0, "c2": 2.0, "c3": 1.0})
    s.temp["selected"] = ["c1", "c2", "c3"]

    algo = SelectN(n=2)
    assert algo(s)
    assert dts[0] in algo.stats.index
    assert list(algo.stats.columns) == ["TOTAL_NAMES", "MEAN", "MEDIAN"]


def test_select_quantile_selects_requested_bucket_descending():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": 4.0, "c2": 3.0, "c3": 2.0, "c4": 1.0})
    s.temp["selected"] = ["c1", "c2", "c3", "c4"]

    algo = SelectQuantile(n_tiles=2, tile=1, stat_key="stat", sort_descending=True)
    assert algo(s)
    assert s.temp["selected"] == ["c1", "c2"]


def test_select_quantile_selects_requested_bucket_ascending():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": 4.0, "c2": 3.0, "c3": 2.0, "c4": 1.0})
    s.temp["selected"] = ["c1", "c2", "c3", "c4"]

    algo = SelectQuantile(n_tiles=2, tile=1, stat_key="stat", sort_descending=False)
    assert algo(s)
    assert s.temp["selected"] == ["c4", "c3"]


def test_select_quantile_validates_inputs():
    with pytest.raises(ValueError, match="n_tiles"):
        SelectQuantile(n_tiles=1, tile=1, stat_key="stat")

    with pytest.raises(ValueError, match="tile"):
        SelectQuantile(n_tiles=4, tile=5, stat_key="stat")

    with pytest.raises(TypeError, match="stat_key"):
        SelectQuantile(n_tiles=4, tile=1, stat_key="")

    with pytest.raises(TypeError, match="sort_descending"):
        SelectQuantile(n_tiles=4, tile=1, sort_descending=1)


def test_select_quantile_returns_false_on_missing_required_state():
    algo = SelectQuantile(n_tiles=4, tile=1, stat_key="stat")
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    assert not algo(s)
    s.temp["stat"] = pd.Series({"c1": 1.0})
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_quantile_updates_stats():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": 4.0, "c2": 3.0, "c3": 2.0, "c4": 1.0})
    s.temp["selected"] = ["c1", "c2", "c3", "c4"]

    algo = SelectQuantile(n_tiles=2, tile=2, stat_key="stat")
    assert algo(s)
    assert dts[0] in algo.stats.index
    assert list(algo.stats.columns) == ["TOTAL_NAMES", "MEAN", "MEDIAN"]


def test_double_sorting_selects_best_quantile_per_sector_descending():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    sector_wide = pd.DataFrame(
        index=dts,
        data={"c1": ["Tech"], "c2": ["Tech"], "c3": ["Energy"], "c4": ["Energy"]},
    )
    s.setup(data, sector_wide=sector_wide)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": 10.0, "c2": 5.0, "c3": 8.0, "c4": 1.0})
    s.temp["selected"] = ["c1", "c2", "c3", "c4"]

    algo = SectorDoubleSort(n_tiles=2, stat_key="stat", sort_descending=True)
    assert algo(s)
    assert s.temp["selected"] == ["c1", "c3"]
    assert dts[0] in algo.stats.index


def test_double_sorting_selects_best_quantile_per_sector_ascending():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    sector_wide = pd.DataFrame(
        index=dts,
        data={"c1": ["Tech"], "c2": ["Tech"], "c3": ["Energy"], "c4": ["Energy"]},
    )
    s.setup(data, sector_wide=sector_wide)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": 10.0, "c2": 5.0, "c3": 8.0, "c4": 1.0})
    s.temp["selected"] = ["c1", "c2", "c3", "c4"]

    algo = SectorDoubleSort(n_tiles=2, stat_key="stat", sort_descending=False)
    assert algo(s)
    assert s.temp["selected"] == ["c2", "c4"]


def test_double_sorting_respects_preselected_subset():
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    sector_wide = pd.DataFrame(
        index=dts,
        data={"c1": ["Tech"], "c2": ["Tech"], "c3": ["Energy"], "c4": ["Energy"]},
    )
    s.setup(data, sector_wide=sector_wide)
    s.update(dts[0])
    s.temp["stat"] = pd.Series({"c1": 10.0, "c2": 5.0, "c3": 8.0, "c4": 1.0})
    s.temp["selected"] = ["c1", "c2", "c4"]

    algo = SectorDoubleSort(n_tiles=2, stat_key="stat")
    assert algo(s)
    assert s.temp["selected"] == ["c1", "c4"]


def test_double_sorting_validates_inputs():
    with pytest.raises(ValueError, match="`n_tiles`"):
        SectorDoubleSort(n_tiles=1)
    with pytest.raises(TypeError, match="stat_key"):
        SectorDoubleSort(n_tiles=2, stat_key="")
    with pytest.raises(TypeError, match="sort_descending"):
        SectorDoubleSort(n_tiles=2, sort_descending=1)
    with pytest.raises(TypeError, match="sector_data"):
        SectorDoubleSort(n_tiles=2, sector_data=123)


def test_double_sorting_returns_false_on_missing_state():
    algo = SectorDoubleSort(n_tiles=2)
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    assert not algo(s)
    s.temp["stat"] = pd.Series({"c1": 1.0})
    assert not algo(s)
    s.temp["selected"] = ["c1"]
    assert not algo(s)


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


def test_select_is_open_filters_candidate_pool_to_open_positions():
    algo = SelectIsOpen()

    dts = pd.date_range("2010-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)

    target = mock.MagicMock()
    target.temp = {"selected": ["c2", "c1"]}
    target.universe = universe
    target.now = dts[0]
    target.children = {
        "c1": mock.MagicMock(weight=0.5),
        "c3": mock.MagicMock(weight=0.0),
        "outside": mock.MagicMock(weight=1.0),
    }

    assert algo(target)
    assert target.temp["selected"] == ["c1"]


def test_select_is_open_uses_select_all_when_selected_missing_or_empty():
    algo = SelectIsOpen()
    dts = pd.date_range("2010-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    target = mock.MagicMock()
    target.temp = {"selected": []}
    target.universe = universe
    target.now = dts[0]
    target.children = {
        "c1": mock.MagicMock(weight=0.2),
        "c2": mock.MagicMock(weight=0.0),
    }

    assert algo(target)
    assert target.temp["selected"] == ["c1"]


def test_select_is_open_returns_false_on_malformed_state():
    algo = SelectIsOpen()

    target = mock.MagicMock(spec=[])
    assert not algo(target)

    target = mock.MagicMock()
    dts = pd.date_range("2010-01-01", periods=1)
    target.temp = {}
    target.universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    target.now = dts[0]
    target.children = []
    assert not algo(target)


def test_select_is_open_treats_near_zero_weight_as_closed():
    algo = SelectIsOpen()
    dts = pd.date_range("2010-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)

    target = mock.MagicMock()
    target.temp = {"selected": ["c1"]}
    target.universe = universe
    target.now = dts[0]
    target.children = {"c1": mock.MagicMock(weight=1e-13)}

    assert algo(target)
    assert target.temp["selected"] == []


def test_select_is_open_skips_candidates_missing_from_children():
    algo = SelectIsOpen()
    dts = pd.date_range("2010-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    target = mock.MagicMock()
    target.temp = {"selected": ["c1", "c2"]}
    target.universe = universe
    target.now = dts[0]
    target.children = {"c1": mock.MagicMock(weight=0.5)}

    assert algo(target)
    assert target.temp["selected"] == ["c1"]


def test_select_is_open_returns_false_when_weight_accessor_raises():
    algo = SelectIsOpen()
    dts = pd.date_range("2010-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)

    class _Child:
        @property
        def weight(self):
            raise RuntimeError("weight unavailable")

    target = mock.MagicMock()
    target.temp = {"selected": ["c1"]}
    target.universe = universe
    target.now = dts[0]
    target.children = {"c1": _Child()}

    assert not algo(target)


def test_select_is_open_ignores_non_string_selected_entries():
    algo = SelectIsOpen()
    dts = pd.date_range("2010-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)

    target = mock.MagicMock()
    target.temp = {"selected": ["c1", 123, None]}
    target.universe = universe
    target.now = dts[0]
    target.children = {
        "c1": mock.MagicMock(weight=0.5),
        "c2": mock.MagicMock(weight=0.0),
    }

    assert algo(target)
    assert target.temp["selected"] == ["c1"]


def test_remove_securities_removes_from_existing_selection():
    algo = RemoveSecurities(["c2", "c4"])
    dts = pd.date_range("2010-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    target = mock.MagicMock()
    target.temp = {"selected": ["c1", "c2", "c3", "c4"]}
    target.universe = universe
    target.now = dts[0]

    assert algo(target)
    assert target.temp["selected"] == ["c1", "c3"]


def test_remove_securities_sets_empty_when_selected_missing():
    algo = RemoveSecurities(["c1"])
    dts = pd.date_range("2010-01-01", periods=1)
    universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    target = mock.MagicMock()
    target.temp = {}
    target.universe = universe
    target.now = dts[0]

    assert algo(target)
    assert target.temp["selected"] == []


def test_remove_securities_returns_false_on_malformed_temp():
    algo = RemoveSecurities(["c1"])
    target = mock.MagicMock()
    target.temp = []
    dts = pd.date_range("2010-01-01", periods=1)
    target.universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    target.now = dts[0]
    assert not algo(target)


def test_add_security_filters_closed_missing_and_non_positive():
    algo = AddSecurity(["c1", "c2", "c3", "missing"], include_negative=False)

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3", "c4"], data=100.0)
    data.loc[dts[0], "c2"] = -1.0
    data.loc[dts[0], "c3"] = np.nan
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c4"]
    s.perm["closed"] = {"c4"}

    assert algo(s)
    assert s.temp["selected"] == ["c1", "c4"]


def test_add_security_include_negative_keeps_non_null_prices():
    algo = AddSecurity(["c1", "c2", "c3"], include_negative=True)

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    data.loc[dts[0], "c2"] = -1.0
    data.loc[dts[0], "c3"] = np.nan
    s.setup(data)
    s.update(dts[0])

    assert algo(s)
    assert s.temp["selected"] == ["c1", "c2"]


def test_add_security_returns_false_on_invalid_now():
    algo = AddSecurity(["c1"])
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.now = "not-a-date"

    assert not algo(s)
    assert "selected" not in s.temp


def test_add_security_returns_false_on_malformed_perm():
    algo = AddSecurity(["c1"])
    target = mock.MagicMock()
    dts = pd.date_range("2010-01-01", periods=1)
    target.temp = {}
    target.universe = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    target.now = dts[0]
    target.perm = []

    assert not algo(target)


def test_add_security_caches_closed_tickers_as_permanently_ineligible():
    algo = AddSecurity(["c1", "c2"])

    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)

    s.update(dts[0])
    s.perm["closed"] = {"c2"}
    assert algo(s)
    assert s.temp["selected"] == ["c1"]

    s.update(dts[1])
    s.perm["closed"] = set()
    s.temp["selected"] = []
    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_sector_filters_candidate_pool_by_sector():
    algo = SelectSector(["Tech"])
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2", "c3"], data=100.0)
    sector_wide = pd.DataFrame(
        index=dts,
        data={"c1": ["Tech"], "c2": ["Energy"], "c3": ["Tech"]},
    )
    s.setup(data, sector_wide=sector_wide)
    s.update(dts[0])
    s.temp["selected"] = ["c2", "c1", "c3"]

    assert algo(s)
    assert s.temp["selected"] == ["c1", "c3"]


def test_select_sector_uses_sector_wide_index_when_selected_missing():
    algo = SelectSector(["Tech"])
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    sector_wide = pd.DataFrame(
        index=dts,
        data={"c1": ["Tech"], "c2": ["Energy"]},
    )
    s.setup(data, sector_wide=sector_wide)
    s.update(dts[0])

    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_sector_returns_false_on_missing_or_invalid_sector_wide():
    algo = SelectSector(["Tech"])
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    s.setup(data)
    s.update(dts[0])

    assert not algo(s)

    s = Strategy("s")
    s.setup(data, sector_wide={"c1": "Tech"})
    s.update(dts[0])
    assert not algo(s)


def test_select_sector_returns_false_when_now_not_in_sector_wide_index():
    algo = SelectSector(["Tech"])
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=2)
    data = pd.DataFrame(index=dts, columns=["c1"], data=100.0)
    sector_wide = pd.DataFrame(
        index=[dts[0]],
        data={"c1": ["Tech"]},
    )
    s.setup(data, sector_wide=sector_wide)
    s.update(dts[1])

    assert not algo(s)


def test_select_sector_accepts_sector_data_dataframe_in_init():
    dts = pd.date_range("2010-01-01", periods=1)
    sector_df = pd.DataFrame(
        index=dts,
        data={"c1": ["Tech"], "c2": ["Energy"]},
    )
    algo = SelectSector(["Tech"], sector_data=sector_df)
    s = Strategy("s")
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    s.setup(data)
    s.update(dts[0])
    s.temp["selected"] = ["c2", "c1"]

    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_sector_accepts_custom_sector_key_string():
    algo = SelectSector(["Tech"], sector_data="custom_sector_wide")
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    custom_sector_wide = pd.DataFrame(
        index=dts,
        data={"c1": ["Tech"], "c2": ["Energy"]},
    )
    s.setup(data, custom_sector_wide=custom_sector_wide)
    s.update(dts[0])

    assert algo(s)
    assert s.temp["selected"] == ["c1"]


def test_select_sector_validates_sector_data_type():
    with pytest.raises(TypeError, match="sector_data"):
        SelectSector(["Tech"], sector_data=123)


def test_select_sector_excludes_missing_sector_labels_in_row():
    algo = SelectSector(["Tech"])
    s = Strategy("s")
    dts = pd.date_range("2010-01-01", periods=1)
    data = pd.DataFrame(index=dts, columns=["c1", "c2"], data=100.0)
    sector_wide = pd.DataFrame(
        index=dts,
        data={"c1": ["Tech"], "c2": [pd.NA]},
    )
    s.setup(data, sector_wide=sector_wide)
    s.update(dts[0])
    s.temp["selected"] = ["c1", "c2"]

    assert algo(s)
    assert s.temp["selected"] == ["c1"]


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
