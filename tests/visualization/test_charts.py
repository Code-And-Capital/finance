import pytest
import pandas as pd
from plotly.graph_objects import Scatter as GoScatter
from visualization.charts import Chart, Line, Scatter, Area, Candlestick


# -------------------------------
# Fixtures
# -------------------------------


@pytest.fixture
def df_small():
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0, 1, 2],
            "close": [1.5, 2.5, 3.5],
        }
    )


# -------------------------------
# Base functionality
# -------------------------------


def test_chart_init_requires_dataframe():
    with pytest.raises(TypeError):
        Chart(df=123)  # abstract class, but validation still triggers


def test_color_cycle(df_small):
    line = Line(df_small)
    c1 = line._next_color()
    c2 = line._next_color()
    assert c1 != c2
    assert isinstance(c1, str)


def test_ensure_columns_exist(df_small):
    line = Line(df_small)
    line._ensure_columns_exist(["a", "b"])  # ok
    with pytest.raises(KeyError):
        line._ensure_columns_exist(["missing"])


def test_coerce_x_values(df_small):
    line = Line(df_small)

    # None → index
    assert list(line._coerce_x_values(None)) == list(df_small.index)

    # "index" → index
    assert list(line._coerce_x_values("index")) == list(df_small.index)

    # column
    assert list(line._coerce_x_values("a")) == list(df_small["a"])

    # explicit sequence
    assert line._coerce_x_values([10, 11, 12]) == [10, 11, 12]

    # invalid input
    with pytest.raises(ValueError):
        line._coerce_x_values(123)


def test_trace_update_canonicalization(df_small):
    line = Line(df_small)
    line._add_trace_update({"update": {"marker": {"color": "red"}}})

    u = line.get_trace_updates()
    assert len(u) == 1
    assert "chart_id" in u[0]
    assert "update" in u[0]
    assert u[0]["update"]["marker"]["color"] == "red"


def test_layout_update(df_small):
    line = Line(df_small)
    line._add_layout_update({"template": "plotly_dark"})

    updates = line.get_layout_updates()
    assert len(updates) == 1
    assert updates[0]["template"] == "plotly_dark"
    assert "chart_id" in updates[0]


def test_axis_update(df_small):
    line = Line(df_small)
    line.xaxis(text="X Label", grid=True)

    updates = line.get_axis_updates()
    assert len(updates) == 1
    assert "xaxis" in updates[0]
    assert updates[0]["xaxis"]["title"]["text"] == "X Label"


def test_clear_methods(df_small):
    line = Line(df_small)
    line.traces = ["dummy"]
    line.trace_updates = [{"a": 1}]
    line.layout_updates = [{"b": 2}]
    line.annotations = [{"c": 3}]

    line.clear_traces()
    line.clear_trace_updates()
    line.clear_layout_updates()
    line.clear_annotations()

    assert line.traces == []
    assert line.trace_updates == []
    assert line.layout_updates == []
    assert line.annotations == []


# -------------------------------
# Subclass: Line
# -------------------------------


def test_line_create_single_series(df_small):
    line = Line(df_small).create(x="a", y="b")
    assert len(line.traces) == 1

    t = line.traces[0]
    assert isinstance(t, GoScatter)
    assert list(t.x) == list(df_small["a"])
    assert list(t.y) == list(df_small["b"])
    assert t.meta["y_col"] == "b"


def test_line_create_multiple_series(df_small):
    line = Line(df_small).create(x="a", y=["a", "b"])
    assert len(line.traces) == 2
    names = {t.name for t in line.traces}
    assert names == {"a", "b"}


# -------------------------------
# Subclass: Scatter
# -------------------------------


def test_scatter_create(df_small):
    scatter = Scatter(df_small).create(x="a", y="b", size=10, color="blue")
    assert len(scatter.traces) == 1
    t = scatter.traces[0]
    assert t.marker["size"] == 10
    assert t.marker["color"] == "blue"


def test_scatter_with_explicit_sequence_x(df_small):
    scatter = Scatter(df_small).create(x=[10, 20, 30], y="a")
    t = scatter.traces[0]
    assert list(t.x) == [10, 20, 30]


# -------------------------------
# Subclass: Area
# -------------------------------


def test_area_create(df_small):
    area = Area(df_small).create(x="a", y="b")
    assert len(area.traces) == 1
    t = area.traces[0]
    assert t.fill == "tozeroy"
    assert list(t.y) == list(df_small["b"])


# -------------------------------
# Subclass: Candlestick
# -------------------------------


def test_candlestick_create(df_small):
    c = Candlestick(df_small).create(
        x="a",
        open="open",
        high="high",
        low="low",
        close="close",
    )
    assert len(c.traces) == 1
    t = c.traces[0]
    assert list(t.open) == list(df_small["open"])
    assert list(t.close) == list(df_small["close"])


def test_candlestick_missing_column(df_small):
    with pytest.raises(KeyError):
        Candlestick(df_small).create(
            x="a", open="open", high="high", low="MISSING", close="close"
        )


# -------------------------------
# Hover and legend
# -------------------------------


def test_hover_generates_customdata(df_small):
    line = Line(df_small).create(x="a", y="b")
    line.hover(columns=["b"], use_custom_hover=True)

    updates = line.get_trace_updates()
    assert len(updates) == 1
    assert "customdata" in updates[0]["update"]


def test_legend_update(df_small):
    line = Line(df_small).create(x="a", y="b")
    line.legend(show=False)
    updates = line.get_layout_updates()
    assert any(u.get("showlegend") is False for u in updates)


# -------------------------------
# Quick styling
# -------------------------------


def test_quick_styling(df_small):
    line = Line(df_small).create(x="a", y="b")
    line.quick_styling(x_title="X", y_title="Y", rangeslider=True)

    # it should have added axis updates + hover + legend
    assert line.axis_updates != []
    assert line.layout_updates != []
