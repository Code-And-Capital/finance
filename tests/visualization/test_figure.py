import pytest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization.figure import Figure


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
class MockChart:
    """A minimal chart-like object for testing."""

    def __init__(self, chart_id="c1", traces=None, default_type="xy"):
        self.chart_id = chart_id
        self.traces = traces or []
        self.trace_updates = []
        self.layout_updates = []
        self.axis_updates = []
        self.default_type = default_type


def make_trace(name="trace1"):
    return go.Scatter(y=[1, 2, 3], name=name)


# ----------------------------------------------------------------------
# Specs
# ----------------------------------------------------------------------
def test_auto_build_specs_simple():
    fig = Figure(rows=2, cols=2)
    c1 = MockChart()
    c2 = MockChart()

    fig.add_chart(c1, row=1, col=1)
    fig.add_chart(c2, row=2, col=2)

    specs = fig._auto_build_specs()

    assert specs[0][0] == {"type": "xy"}
    assert specs[1][1] == {"type": "xy"}


def test_auto_build_specs_with_span():
    fig = Figure(rows=2, cols=2)
    c1 = MockChart()

    fig.add_chart(c1, row=1, col=1, rowspan=2, colspan=2)

    specs = fig._auto_build_specs()

    assert specs[0][0]["type"] == "xy"
    assert specs[0][0]["rowspan"] == 2
    assert specs[0][0]["colspan"] == 2
    # All other cells must be None
    assert specs[0][1] is None
    assert specs[1][0] is None
    assert specs[1][1] is None


def test_auto_build_specs_bounds_error():
    fig = Figure(rows=1, cols=1)
    c1 = MockChart()

    with pytest.raises(ValueError):
        fig.add_chart(c1, row=1, col=1, rowspan=2)
        fig._auto_build_specs()


# ----------------------------------------------------------------------
# Build Process
# ----------------------------------------------------------------------
def test_build_adds_traces_correctly():
    fig = Figure(rows=1, cols=1)
    c = MockChart(traces=[make_trace("A"), make_trace("B")])

    fig.add_chart(c, row=1, col=1)
    fig.build()

    assert len(fig.fig.data) == 2
    names = [t.name for t in fig.fig.data]
    assert "A" in names
    assert "B" in names


def test_build_assigns_colors_consistently():
    fig = Figure(rows=1, cols=1)

    c1 = MockChart(traces=[make_trace("A")])
    c2 = MockChart(traces=[make_trace("A")])  # duplicate legend name, no legend shown

    fig.add_chart(c1, row=1, col=1)
    fig.add_chart(c2, row=1, col=1)
    fig.build()

    t1, t2 = fig.fig.data
    assert t1.line.color == t2.line.color
    assert t1.showlegend is True
    assert t2.showlegend is False


def test_build_candlestick_default_color_behavior():
    fig = Figure(rows=1, cols=1)
    trace = go.Candlestick(
        x=[1, 2], open=[1, 2], high=[2, 3], low=[0, 1], close=[1, 2], name="CS"
    )

    c = MockChart(traces=[trace])
    fig.add_chart(c, row=1, col=1)
    fig.build()

    t = fig.fig.data[0]
    assert isinstance(t, go.Candlestick)
    # Candlestick traces are forced to hide legends
    assert t.showlegend is False


def test_build_indicator_skips_color_assignment():
    fig = Figure(rows=1, cols=1)
    trace = go.Indicator(mode="number", value=42)

    c = MockChart(traces=[trace], default_type="indicator")
    fig.add_chart(c, row=1, col=1)
    fig.build()

    t = fig.fig.data[0]
    assert isinstance(t, go.Indicator)
    # Should not add marker/line color
    assert not hasattr(t, "marker") or t.marker is None


# ----------------------------------------------------------------------
# Updates
# ----------------------------------------------------------------------
def test_layout_updates_before_build():
    fig = Figure(rows=1, cols=1)
    fig.add_chart(MockChart(traces=[]))
    fig.layout(title="MyTitle")

    fig.build()
    assert fig.fig.layout.title.text == "MyTitle"


def test_layout_updates_after_build():
    fig = Figure(rows=1, cols=1)
    fig.add_chart(MockChart(traces=[]))
    fig.build()
    fig.layout(title="NewTitle")

    assert fig.fig.layout.title.text == "NewTitle"


def test_axis_updates():
    fig = Figure(rows=1, cols=1)
    fig.add_chart(MockChart(traces=[]))
    fig.build()

    fig.update_axes(x_tickformat="%Y", y_tickformat=".2f")

    assert fig.fig.layout.xaxis.tickformat == "%Y"
    assert fig.fig.layout.yaxis.tickformat == ".2f"


# ----------------------------------------------------------------------
# Trace Updates / Layout Updates / Axis Updates from charts
# ----------------------------------------------------------------------
def test_trace_updates_from_chart():
    trace = make_trace("A")
    c = MockChart(traces=[trace])
    c.chart_id = "chart1"
    c.trace_updates.append({"chart_id": "chart1", "update": {"opacity": 0.5}})

    fig = Figure(rows=1, cols=1)
    fig.add_chart(c, row=1, col=1)
    fig.build()

    t = fig.fig.data[0]
    assert t.opacity == 0.5


def test_layout_updates_from_chart():
    c = MockChart()
    c.chart_id = "chart1"
    c.layout_updates.append({"chart_id": "chart1", "title": "Chart Title"})

    fig = Figure(rows=1, cols=1)
    fig.add_chart(c, row=1, col=1)
    fig.build()

    assert fig.fig.layout.title.text == "Chart Title"


def test_axis_updates_from_chart():
    c = MockChart()
    c.chart_id = "chart1"
    c.axis_updates.append({"chart_id": "chart1", "xaxis": {"tickformat": "%d"}})

    fig = Figure(rows=1, cols=1)
    fig.add_chart(c, row=1, col=1)
    fig.build()

    assert fig.fig.layout.xaxis.tickformat == "%d"


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------
def test_save_html(monkeypatch):
    fig = Figure()
    fig.add_chart(MockChart(traces=[]))
    fig.build()

    # Patch figure.write_html
    called = {}

    def fake_write_html(path):
        called["path"] = path

    monkeypatch.setattr(fig.fig, "write_html", fake_write_html)

    fig.save("test.html", "html")
    assert called["path"] == "test.html"


def test_save_invalid_format():
    fig = Figure()
    fig.add_chart(MockChart(traces=[]))
    fig.build()

    with pytest.raises(ValueError):
        fig.save("x.txt", file_format="txt")
