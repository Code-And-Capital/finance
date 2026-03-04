import pytest
import pandas as pd
from plotly.graph_objects import Scatter as GoScatter
from visualization.charts import (
    Area,
    Bar,
    Boxplot,
    Candlestick,
    Chart,
    Contour,
    Heatmap,
    Histogram,
    Indicator,
    Line,
    Pie,
    Scatter,
    ScatterGeo,
    Waterfall,
)


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


@pytest.fixture
def df_geo():
    return pd.DataFrame(
        {
            "LAT": [37.77, 40.71],
            "LON": [-122.42, -74.01],
            "CITY": ["San Francisco", "New York"],
            "SIZE": [10, 20],
            "COLOR": ["red", "blue"],
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


def test_bar_create_adds_layout_update(df_small):
    bar = Bar(df_small).create(
        x="a", y=["a", "b"], barmode="stack", bar_corner_radius=4
    )
    assert len(bar.traces) == 2
    updates = bar.get_layout_updates()
    assert any(u.get("barmode") == "stack" for u in updates)
    assert any(u.get("barcornerradius") == 4 for u in updates)


def test_bar_split_categories_creates_category_traces(df_small):
    bar = Bar(df_small).create(x="a", y="b", split_categories=True)
    assert len(bar.traces) == len(df_small)
    assert all(len(trace.x) == 1 for trace in bar.traces)


def test_histogram_multiple_series_sets_overlay_and_opacity(df_small):
    hist = Histogram(df_small).create(y=["a", "b"], bins=5, cumulative=True)
    assert len(hist.traces) == 2
    assert any(u.get("barmode") == "overlay" for u in hist.get_layout_updates())
    assert any(
        u.get("update", {}).get("opacity") == 0.75 for u in hist.get_trace_updates()
    )


def test_pie_create_uses_labels_and_values(df_small):
    pie = Pie(df_small).create(x="a", y="b", hole=0.3)
    assert len(pie.traces) == 1
    trace = pie.traces[0]
    assert len(trace.labels) == len(df_small)
    assert list(trace.values) == list(df_small["b"])
    assert trace.hole == 0.3


def test_heatmap_create(df_small):
    heatmap = Heatmap(df_small[["a", "b"]]).create(showscale=False)
    assert len(heatmap.traces) == 1
    trace = heatmap.traces[0]
    assert trace.showscale is False
    assert trace.z.shape == (3, 2)


def test_boxplot_create_with_grouping(df_small):
    box = Boxplot(df_small).create(y=["a", "b"], x="a", boxpoints="all", jitter=0.2)
    assert len(box.traces) == 2
    assert box.traces[0].boxpoints == "all"
    assert box.traces[0].jitter == 0.2


def test_indicator_create_from_column_and_delta(df_small):
    indicator = Indicator(df_small).create(
        value="b",
        mode="number+delta",
        delta_reference="a",
        number_format=".2f",
    )
    assert len(indicator.traces) == 1
    trace = indicator.traces[0]
    assert trace.value == float(df_small["b"].iloc[-1])
    assert trace.delta.reference == float(df_small["a"].iloc[-1])


def test_indicator_delta_mode_requires_reference(df_small):
    with pytest.raises(ValueError):
        Indicator(df_small).create(value=1, mode="number+delta")


def test_indicator_missing_column_raises(df_small):
    with pytest.raises(KeyError):
        Indicator(df_small).create(value="missing")


def test_waterfall_create_with_percent_text_and_title(df_small):
    wf = Waterfall(df_small).create(
        x=["Start", "Gain", "total"],
        y=[100, 50, "total"],
        text="percent",
        title="Waterfall",
        y_format=",.0f",
    )
    assert len(wf.traces) == 1
    trace = wf.traces[0]
    assert list(trace.measure) == ["relative", "relative", "total"]
    assert any(u.get("title") == "Waterfall" for u in wf.get_layout_updates())
    assert any("yaxis" in u for u in wf.get_axis_updates())


def test_waterfall_text_length_validation(df_small):
    with pytest.raises(ValueError):
        Waterfall(df_small).create(x=["A", "B"], y=[1, 2], text=["only-one"])


def test_contour_create_from_array(df_small):
    z = [[1, 2], [3, 4]]
    contour = Contour(df_small).create(
        z=z, x=[0, 1], y=[0, 1], showscale=False, name="c"
    )
    assert len(contour.traces) == 1
    trace = contour.traces[0]
    assert trace.showscale is False
    assert trace.name == "c"


def test_scattergeo_create_and_geo_layout(df_geo):
    geo = ScatterGeo(df_geo).create(
        lat="LAT", lon="LON", text="CITY", size="SIZE", color="COLOR"
    )
    assert len(geo.traces) == 1
    trace = geo.traces[0]
    assert len(trace.lat) == len(df_geo)
    assert len(trace.lon) == len(df_geo)

    geo.build_geo_layout_update(scope="usa", projection="albers usa")
    assert any("geo" in u for u in geo.get_layout_updates())


def test_scattergeo_invalid_projection_scope_pair_raises(df_geo):
    geo = ScatterGeo(df_geo)
    with pytest.raises(ValueError):
        geo.build_geo_layout_update(scope="world", projection="albers usa")


def test_scattergeo_size_column_must_exist(df_geo):
    geo = ScatterGeo(df_geo)
    with pytest.raises(KeyError):
        geo.create(lat="LAT", lon="LON", size="MISSING_SIZE_COL")
