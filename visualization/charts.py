from __future__ import annotations
import abc
import uuid
import itertools
import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------
# Utilities & defaults
# ---------------------------

DEFAULT_COLOR_CYCLE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def is_sequence(obj: Any) -> bool:
    """Return True if `obj` is a non-string sequence."""
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def ensure_list(x: Optional[Union[str, Iterable[str]]]) -> List[str]:
    """Coerce a string or iterable into a list of strings."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


# ---------------------------
# Base Chart
# ---------------------------


class Chart(abc.ABC):
    """
    Abstract base class for chart builders.

    Chart instances hold:
      - `traces`: a list of Plotly trace objects (Scatter, Bar, Pie, etc.)
      - `trace_updates`: trace-level updates intended to be applied by a Figure renderer
      - `layout_updates`: layout-level updates intended for the subplot where this chart sits
      - `axis_updates`: x/y axis updates for the subplot
      - `annotations`: annotations tied to this chart's subplot

    Subclasses MUST implement `create(...)` which populates `self.traces`.

    Parameters
    ----------
    df : pd.DataFrame
        Data source used to create traces.
    chart_id : str, optional
        Unique identifier used in trace.meta and updates. If not provided, a UUID is generated.
    color_cycle : Sequence[str], optional
        Color palette used for color assignment when needed.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        chart_id: Optional[str] = None,
        color_cycle: Optional[Sequence[str]] = None,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas.DataFrame")
        # instance state (no inline type hints per user's request)
        self.data = df.copy()
        self.traces = []
        self.trace_updates = []
        self.layout_updates = []
        self.axis_updates = []
        self.annotations = []
        self.chart_id = chart_id or str(uuid.uuid4())
        self._color_cycle = list(color_cycle or DEFAULT_COLOR_CYCLE)
        self._color_iter = itertools.cycle(self._color_cycle)

    # ---------- abstract ----------
    @abc.abstractmethod
    def create(self, *args, **kwargs) -> "Chart":
        """
        Populate `self.traces`.

        Returns
        -------
        Chart
            self to allow method chaining.
        """
        raise NotImplementedError

    # ---------- meta + color ----------
    def _meta_for(self, **extras) -> Dict[str, Any]:
        """Return a trace meta payload that includes chart_id and any extras."""
        payload = {"chart_id": self.chart_id}
        payload.update(extras)
        return payload

    def _next_color(self) -> str:
        """Return the next color from the internal cycle."""
        return next(self._color_iter)

    # ---------- validation / coercion ----------
    def _ensure_columns_exist(self, cols: Iterable[str]) -> None:
        """Raise KeyError if any column in `cols` is missing from the DataFrame."""
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise KeyError(f"Missing columns in data: {missing}")

    def _coerce_x_values(self, x: Union[str, Sequence, None]) -> Sequence:
        """
        Coerce a user-provided `x` into a sequence of x values.

        Accepts:
          - None (returns index)
          - "index" (returns index)
          - column name present in DataFrame (returns that column)
          - any sequence (returned as list)
        """
        if x is None:
            return self.data.index
        if isinstance(x, str) and x == "index":
            return self.data.index
        if isinstance(x, str) and x in self.data.columns:
            return self.data[x]
        if is_sequence(x):
            return list(x)
        raise ValueError("x must be None, 'index', a column name in df, or a sequence")

    # ---------- canonicalized update helpers ----------
    def _add_trace_update(self, update: Dict[str, Any]) -> None:
        """
        Store a trace update.

        Expected `update` format:
            {"update": {...}, "selector": {...} (optional)}

        The method will canonicalize the payload to include chart_id.
        """
        if "update" not in update:
            raise ValueError("trace update must include key 'update'")
        u = {"chart_id": self.chart_id, "update": update["update"]}
        if "selector" in update:
            u["selector"] = update["selector"]
        self.trace_updates.append(u)

    def _add_layout_update(self, update: Dict[str, Any]) -> None:
        """
        Store a layout-level update for the chart's subplot.

        The update dict is merged with {"chart_id": self.chart_id}.
        """
        u = {"chart_id": self.chart_id}
        u.update(update)
        self.layout_updates.append(u)

    def _add_axis_update(self, update: Dict[str, Any]) -> None:
        """Store an axis update like {"xaxis": {...}} or {"yaxis": {...}} with chart_id."""
        u = {"chart_id": self.chart_id}
        u.update(update)
        self.axis_updates.append(u)

    # ---------- layout helpers ----------
    def template(self, template: Union[str, Dict[str, Any]]) -> "Chart":
        """
        Store a Plotly template for this chart's subplot.

        Parameters
        ----------
        template : str or dict
            Template name (e.g., "plotly_dark") or a template dict.
        """
        self._add_layout_update({"template": template})
        return self

    def xaxis(
        self,
        text: Optional[str] = None,
        font_size: int = 12,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        grid: Optional[bool] = None,
        tickformat: Optional[str] = None,
        gridcolor: Optional[str] = None,
        rangemode: Optional[str] = None,
        tickangle: Optional[int] = None,
        rangeslider: Optional[bool] = None,
        rangebreaks: Optional[List[dict]] = None,
    ) -> "Chart":
        """
        Store x-axis configuration for the chart's subplot.

        Only non-None arguments are persisted so the Figure renderer can merge updates.
        """
        cfg = {}
        if text is not None:
            cfg["title"] = {"text": text, "font": {"size": font_size}}
        if start is not None or end is not None:
            cfg["range"] = [start, end]
        if grid is not None:
            cfg["showgrid"] = bool(grid)
        if tickformat is not None:
            cfg["tickformat"] = tickformat
        if gridcolor is not None:
            cfg["gridcolor"] = gridcolor
        if rangemode is not None:
            cfg["rangemode"] = rangemode
        if tickangle is not None:
            cfg["tickangle"] = tickangle
        if rangeslider is not None:
            cfg["rangeslider"] = {"visible": bool(rangeslider)}
        if rangebreaks is not None:
            cfg["rangebreaks"] = rangebreaks

        if cfg:
            self._add_axis_update({"xaxis": cfg})
        return self

    def yaxis(
        self,
        text: Optional[str] = None,
        font_size: int = 12,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        grid: Optional[bool] = None,
        tickformat: Optional[str] = None,
        gridcolor: Optional[str] = None,
        rangemode: Optional[str] = None,
        tickangle: Optional[int] = None,
    ) -> "Chart":
        """Store y-axis configuration (same semantics as xaxis)."""
        cfg = {}
        if text is not None:
            cfg["title"] = {"text": text, "font": {"size": font_size}}
        if start is not None or end is not None:
            cfg["range"] = [start, end]
        if grid is not None:
            cfg["showgrid"] = bool(grid)
        if tickformat is not None:
            cfg["tickformat"] = tickformat
        if gridcolor is not None:
            cfg["gridcolor"] = gridcolor
        if rangemode is not None:
            cfg["rangemode"] = rangemode
        if tickangle is not None:
            cfg["tickangle"] = tickangle

        if cfg:
            self._add_axis_update({"yaxis": cfg})
        return self

    def add_rangeslider(self, visible: bool = True) -> "Chart":
        """Enable/disable the x-axis rangeslider for the chart's subplot."""
        self._add_axis_update({"xaxis": {"rangeslider": {"visible": bool(visible)}}})
        return self

    def add_selector_buttons(
        self, buttons: Optional[List[dict]] = None, **kwargs
    ) -> "Chart":
        """
        Add range selector buttons for the x-axis.

        Parameters
        ----------
        buttons : list[dict], optional
            If None, a sensible default set is used.
        kwargs : dict
            Any additional keys forwarded into the rangeselector dict (e.g., bgcolor, x, y).
        """
        if buttons is None:
            buttons = [
                dict(step="all", label="All"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
            ]
        rangeselector = {"buttons": buttons}
        rangeselector.update(kwargs)
        self._add_axis_update({"xaxis": {"rangeselector": rangeselector}})
        return self

    def background(
        self, plot_bg: Optional[str] = None, paper_bg: Optional[str] = None
    ) -> "Chart":
        """
        Store background settings for the subplot.

        Parameters
        ----------
        plot_bg : str, optional
            Color for plot_bgcolor.
        paper_bg : str, optional
            Color for paper_bgcolor.
        """
        u = {}
        if plot_bg is not None:
            u["plot_bgcolor"] = plot_bg
        if paper_bg is not None:
            u["paper_bgcolor"] = paper_bg
        if u:
            self._add_layout_update(u)
        return self

    def margins(self, **kwargs) -> "Chart":
        """
        Store margin settings for the subplot.

        Accepts Plotly's margin arguments via kwargs (l=..., r=..., t=..., b=..., pad=...).
        """
        margin = {k: v for k, v in kwargs.items() if v is not None}
        if margin:
            self._add_layout_update({"margin": margin})
        return self

    # ---------- trace-level styling ----------
    def marker_style(
        self,
        size: Optional[Union[int, Sequence[int]]] = None,
        line_width: Optional[int] = None,
        line_color: Optional[str] = None,
    ) -> "Chart":
        """
        Store marker styling to be applied to traces.

        Parameters
        ----------
        size : int or sequence[int], optional
            Marker size or per-point sizes.
        line_width : int, optional
            Marker outline width.
        line_color : str, optional
            Marker outline color.
        """
        marker = {}
        if size is not None:
            marker["size"] = size
        if line_width is not None or line_color is not None:
            marker_line = {}
            if line_width is not None:
                marker_line["width"] = line_width
            if line_color is not None:
                marker_line["color"] = line_color
            marker["line"] = marker_line
        if marker:
            self._add_trace_update({"update": {"marker": marker}})
        return self

    def line_style(
        self, width: Optional[int] = None, dash: Optional[str] = None
    ) -> "Chart":
        """Store line styling (width, dash) for traces."""
        line = {}
        if width is not None:
            line["width"] = width
        if dash is not None:
            line["dash"] = dash
        if line:
            self._add_trace_update({"update": {"line": line}})
        return self

    def hover(
        self,
        columns: Optional[Sequence[str]] = None,
        template: Optional[str] = None,
        apply_to: Union[str, None] = None,
        decimals: int = 2,
        y_format: Optional[str] = None,
        x_format: Optional[str] = None,
        minimal_customdata: bool = False,
        use_custom_hover: bool = False,  # <-- new argument
    ) -> "Chart":
        """
        Configure hovertemplate and customdata for traces.

        Parameters
        ----------
        columns : sequence[str], optional
            Columns to include in customdata. If None, infers columns from traces' meta or names.
        template : str, optional
            Fully-specified Plotly hovertemplate. If None, auto-generated.
        apply_to : str or None
            If given, apply hover updates only to traces whose `.name` equals this string.
        decimals : int
            Decimal precision for autogenerated numeric fields.
        y_format, x_format : str, optional
            Format specifiers forwarded into template.
        minimal_customdata : bool
            If True, only attach necessary customdata (saves memory for large datasets).
        use_custom_hover : bool
            If False, use default Plotly hover (ignore template and customdata).
        """

        if not use_custom_hover:
            # Do nothing → Plotly default hover
            return self

        # infer columns when not provided
        if columns is None:
            inferred = []
            for t in self.traces:
                if apply_to is not None and getattr(t, "name", None) != apply_to:
                    continue
                meta = getattr(t, "meta", {}) or {}
                if isinstance(meta, dict) and "y_col" in meta:
                    inferred.append(meta["y_col"])
                else:
                    if getattr(t, "name", None) in self.data.columns:
                        inferred.append(t.name)
            columns = [c for c in dict.fromkeys(inferred) if c in self.data.columns]

        columns = list(columns or [])
        if columns:
            self._ensure_columns_exist(columns)
            # build customdata as numpy array then convert to list for JSON safety
            customdata = np.column_stack([self.data[c].to_numpy() for c in columns])
            customdata_payload = customdata.tolist()
        else:
            customdata_payload = None

        # construct template if not provided
        if template is None:
            x_t = f"%{{x|{x_format}}}" if x_format else "%{x}"
            y_t = f"%{{y:{y_format}}}" if y_format else f"%{{y:.{decimals}f}}"
            lines = [
                f"{c}: %{{customdata[{i}]:.{decimals}f}}<br>"
                for i, c in enumerate(columns)
            ]
            template = f"x: {x_t}<br>y: {y_t}<br>" + "".join(lines) + "<extra></extra>"

        update = {"update": {"hovertemplate": template}}
        if customdata_payload is not None:
            update["update"]["customdata"] = customdata_payload
        self._add_trace_update(update)
        return self

    # ---------- annotations & legend ----------
    def add_annotation(
        self,
        text: str,
        x: Any,
        y: Any,
        font_size: int = 12,
        showarrow: bool = False,
        **extra,
    ) -> "Chart":
        """
        Add an annotation object associated with this chart's subplot.

        Parameters
        ----------
        text : str
            Annotation text.
        x, y : scalar or str
            Annotation coordinates (data or paper coordinates depending on renderer).
        font_size : int
            Font size for the annotation text.
        showarrow : bool
            Whether to show an arrow pointing to (x, y).
        extra : dict
            Extra Plotly annotation properties forwarded as-is.
        """
        ann = {
            "chart_id": self.chart_id,
            "annotation": {
                "text": text,
                "x": x,
                "y": y,
                "font": {"size": font_size},
                "showarrow": bool(showarrow),
            },
        }
        ann["annotation"].update(extra)
        self.annotations.append(ann)
        return self

    def legend(
        self,
        show: bool = True,
        x: float = 1.02,
        y: float = 1,
        orientation: str = "v",
        xanchor: str = "left",
        yanchor: str = "top",
        bgcolor: str = "rgba(0,0,0,0)",
        bordercolor: str = "rgba(0,0,0,0)",
        borderwidth: int = 0,
        font_size: int = 12,
        traceorder: str = "normal",
    ) -> "Chart":
        """
        Store legend configuration for the subplot and toggle showlegend on stored traces.

        Notes
        -----
        The Figure renderer should merge these at layout level when rendering the subplot.
        """
        legend_cfg = {
            "chart_id": self.chart_id,
            "showlegend": bool(show),
            "legend": {
                "x": x,
                "y": y,
                "orientation": orientation,
                "xanchor": xanchor,
                "yanchor": yanchor,
                "bgcolor": bgcolor,
                "bordercolor": bordercolor,
                "borderwidth": borderwidth,
                "font": {"size": font_size},
                "traceorder": traceorder,
            },
        }
        self.layout_updates.append(legend_cfg)
        for t in self.traces:
            try:
                t.showlegend = bool(show)
            except Exception:
                # defensive: some trace types may not accept showlegend
                pass
        return self

    # ---------- housekeeping ----------
    def quick_styling(
        self,
        x_title: Optional[str] = None,
        y_title: Optional[str] = None,
        rangeslider: bool = False,
        selector_buttons: bool = False,
    ) -> "Chart":
        """
        Convenience method to apply commonly used stylistic settings.

        Examples
        --------
        chart.quick_styling(x_title="Date", y_title="Price", rangeslider=True)
        """
        if x_title is not None:
            self.xaxis(text=x_title, rangeslider=rangeslider)
        else:
            if rangeslider:
                self.add_rangeslider(True)
        if selector_buttons:
            self.add_selector_buttons()
        if y_title is not None:
            self.yaxis(text=y_title)
        # attach default hover/legend
        self.hover()
        self.legend(show=True)
        return self

    def clear_traces(self) -> "Chart":
        """Remove all stored traces from this Chart object."""
        self.traces.clear()
        return self

    def clear_layout_updates(self) -> "Chart":
        """Remove all stored layout updates."""
        self.layout_updates.clear()
        return self

    def clear_trace_updates(self) -> "Chart":
        """Remove all stored trace update instructions."""
        self.trace_updates.clear()
        return self

    def clear_annotations(self) -> "Chart":
        """Remove all stored annotations."""
        self.annotations.clear()
        return self

    def get_trace_updates(self) -> List[Dict[str, Any]]:
        """Return a copy of trace updates for a renderer to consume."""
        return list(self.trace_updates)

    def get_layout_updates(self) -> List[Dict[str, Any]]:
        """Return a copy of layout updates for a renderer to consume."""
        return list(self.layout_updates)

    def get_axis_updates(self) -> List[Dict[str, Any]]:
        """Return a copy of axis updates for a renderer to consume."""
        return list(self.axis_updates)

    def get_annotations(self) -> List[Dict[str, Any]]:
        """Return a copy of annotations for a renderer to consume."""
        return list(self.annotations)

    def __repr__(self) -> str:
        return f"<Chart id={self.chart_id} traces={len(self.traces)} layout_updates={len(self.layout_updates)}>"


# ---------------------------
# Subclasses
# ---------------------------


class Line(Chart):
    """
    Line chart builder.

    `create(x, y, ...)` will append one trace per `y` column.
    """

    def create(
        self,
        x: Union[str, Sequence, None],
        y: Union[str, Sequence[str]],
        width: int = 2,
        mode: str = "lines+markers",
        dash: Optional[str] = None,
        yaxis: Optional[str] = None,
    ) -> "Line":
        """
        Create line traces.

        Parameters
        ----------
        x : str or sequence or None
            Column name, 'index', or explicit sequence for x values.
        y : str or sequence[str]
            Column name(s) to plot.
        width : int
            Line width (px).
        mode : str
            Plotly mode ("lines", "markers", "lines+markers").
        dash : str, optional
            Dash style (e.g., "dash", "dot").
        yaxis : str, optional
            Secondary axis id if needed (e.g., "y2").
        """
        y_cols = ensure_list(y)
        self._ensure_columns_exist(y_cols)
        x_vals = self._coerce_x_values(x)

        for col in y_cols:
            trace = go.Scatter(
                x=x_vals,
                y=self.data[col].to_numpy(),
                mode=mode,
                name=str(col),
                line={"width": width, "dash": dash} if (width or dash) else None,
                meta=self._meta_for(y_col=col),
            )
            self.traces.append(trace)
        return self


class Scatter(Chart):
    """Scatter (marker) chart builder."""

    def create(
        self,
        x: Union[str, Sequence, None],
        y: Union[str, Sequence[str]],
        size: Optional[Union[int, Sequence[int]]] = None,
        yaxis: Optional[str] = None,
    ) -> "Scatter":
        """
        Create scatter (marker) traces.

        Parameters
        ----------
        x : str or sequence or None
            Column name, 'index', or explicit sequence for x values.
        y : str or sequence[str]
            Column name(s) to plot as markers.
        size : int or sequence[int], optional
            Marker size or per-point sizes.
        """
        y_cols = ensure_list(y)
        self._ensure_columns_exist(y_cols)
        x_vals = self._coerce_x_values(x)

        for col in y_cols:
            trace = go.Scatter(
                x=x_vals,
                y=self.data[col].to_numpy(),
                mode="markers",
                marker={"size": size} if size is not None else None,
                name=str(col),
                meta=self._meta_for(y_col=col),
            )
            self.traces.append(trace)
        return self


class Area(Chart):
    """Area (filled) chart builder."""

    def create(
        self, x: Union[str, Sequence, None], y: Union[str, Sequence[str]]
    ) -> "Area":
        """
        Create filled area traces (fill to zero).
        """
        y_cols = ensure_list(y)
        self._ensure_columns_exist(y_cols)
        x_vals = self._coerce_x_values(x)

        for col in y_cols:
            trace = go.Scatter(
                x=x_vals,
                y=self.data[col].to_numpy(),
                fill="tozeroy",
                name=str(col),
                meta=self._meta_for(y_col=col),
            )
            self.traces.append(trace)
        return self


class Candlestick(Chart):
    """Candlestick (OHLC) builder."""

    def create(
        self, x: Union[str, Sequence, None], open: str, high: str, low: str, close: str
    ) -> "Candlestick":
        """
        Create a Candlestick trace from OHLC columns.

        Parameters
        ----------
        x : str or sequence or None
            Column name, 'index', or explicit sequence for x values.
        open, high, low, close : str
            Column names for open/high/low/close.
        """
        self._ensure_columns_exist([open, high, low, close])
        x_vals = self._coerce_x_values(x)
        trace = go.Candlestick(
            x=x_vals,
            open=self.data[open].to_numpy(),
            high=self.data[high].to_numpy(),
            low=self.data[low].to_numpy(),
            close=self.data[close].to_numpy(),
            name="Candlestick",
            meta=self._meta_for(),
        )
        self.traces.append(trace)
        return self


class Bar(Chart):
    """Bar chart builder."""

    def create(
        self,
        x: Union[str, Sequence, None],
        y: Union[str, Sequence[str]],
        yaxis: Optional[str] = None,
        barmode: str = "group",
        bar_corner_radius: int = 0,
        split_categories: bool = False,
    ) -> "Bar":
        """
        Create bar traces.

        Parameters
        ----------
        x : str or sequence or None
            Column name, 'index', or explicit sequence for x values.
        y : str or sequence[str]
            Column name(s) to plot as bars.
        barmode : str
            Plotly layout.barmode for grouped/stacked bars.
        split_categories : bool
            If True and x represents categories, create one trace per category
            (trace.name == category). Useful to make legend items that match a Pie's labels.
        """
        y_cols = ensure_list(y)
        self._ensure_columns_exist(y_cols)
        x_vals = self._coerce_x_values(x)

        if not split_categories:
            for col in y_cols:
                trace = go.Bar(
                    x=x_vals,
                    y=self.data[col].to_numpy(),
                    name=str(col),
                    meta=self._meta_for(y_col=col),
                )
                self.traces.append(trace)
        else:
            # split mode: create one trace per category (single-value traces)
            if len(y_cols) == 1:
                col = y_cols[0]
                # determine categories from x (if x is a column) or index
                if isinstance(x, str) and x != "index" and x in self.data.columns:
                    cats = list(self.data[x].astype(str))
                    vals = self.data[col].to_numpy()
                else:
                    # x was index or sequence -> categories from index values
                    cats = list(self.data.index.astype(str))
                    vals = self.data[col].to_numpy()
                for cat, val in zip(cats, vals):
                    trace = go.Bar(
                        x=[cat],
                        y=[val],
                        name=str(cat),
                        marker={"color": self._next_color()},
                        meta=self._meta_for(y_col=col),
                    )
                    self.traces.append(trace)
            else:
                # fallback: when multiple y columns, keep default behavior
                for col in y_cols:
                    trace = go.Bar(
                        x=x_vals,
                        y=self.data[col].to_numpy(),
                        name=str(col),
                        meta=self._meta_for(y_col=col),
                    )
                    self.traces.append(trace)

        self._add_layout_update(
            {"barmode": barmode, "barcornerradius": bar_corner_radius}
        )
        return self


class Histogram(Chart):
    """Histogram builder."""

    def create(
        self,
        y: Union[str, Sequence[str]],
        yaxis: Optional[str] = None,
        bins: Optional[int] = None,
        histnorm: Optional[str] = None,
        cumulative: bool = False,
    ) -> "Histogram":
        """
        Create histogram traces.

        Parameters
        ----------
        y : str or sequence[str]
            Column(s) to histogram.
        bins : int, optional
            Number of bins.
        histnorm : str, optional
            Normalization mode (e.g., 'probability').
        cumulative : bool
            Whether to compute cumulative histogram.
        """
        y_cols = ensure_list(y)
        self._ensure_columns_exist(y_cols)

        for col in y_cols:
            trace = go.Histogram(
                x=self.data[col].to_numpy(),
                name=str(col),
                nbinsx=bins,
                histnorm=histnorm,
                cumulative_enabled=bool(cumulative),
                meta=self._meta_for(y_col=col),
            )
            self.traces.append(trace)

        if len(y_cols) > 1:
            self._add_layout_update({"barmode": "overlay"})
            self._add_trace_update({"update": {"opacity": 0.75}})
        return self


class Pie(Chart):
    """Pie chart builder."""

    def create(
        self,
        x: Union[str, Sequence, None],
        y: str,
        hole: float = 0.0,
        sort: bool = False,
        textinfo: str = "percent+label",
    ) -> "Pie":
        """
        Create a Pie trace.

        Parameters
        ----------
        x : str or sequence or None
            Column name or 'index' or explicit sequence to use as labels.
        y : str
            Column name containing values.
        hole : float
            Donut hole (0.0 == pie, 0.5 == donut).
        sort : bool
            Whether Plotly should sort the pie slices.
        textinfo : str
            Plotly pie 'textinfo' option (e.g., 'percent+label').
        """
        self._ensure_columns_exist([y])
        labels = self._coerce_x_values(x)
        labels_arr = np.asarray(labels).astype(str)
        values = self.data[y].to_numpy()
        trace = go.Pie(
            labels=labels_arr,
            values=values,
            hole=hole,
            sort=sort,
            textinfo=textinfo,
            meta=self._meta_for(y_col=y),
            marker={"line": {"color": "#000000", "width": 1}},
        )
        self.traces.append(trace)
        return self


class Heatmap(Chart):
    """Heatmap builder."""

    def create(self, showscale: bool = True) -> "Heatmap":
        """
        Create a Heatmap trace using the DataFrame matrix.

        The heatmap x axis uses the DataFrame index (coerced to strings) and y uses column names.
        """
        trace = go.Heatmap(
            x=self.data.index.astype(str),
            y=list(self.data.columns),
            z=self.data.values,
            showscale=bool(showscale),
            meta=self._meta_for(),
        )
        self.traces.append(trace)
        return self


class Boxplot(Chart):
    """Boxplot chart builder."""

    def create(
        self,
        y: Union[str, Sequence[str]],
        x: Optional[Union[str, Sequence, None]] = None,
        name: Optional[str] = None,
        boxmean: Union[str, bool, None] = None,
        orientation: str = "v",
        boxpoints: Union[bool, str] = False,
        jitter: float = 0.0,
    ) -> "Boxplot":
        """
        Create one or multiple Box traces.

        Parameters
        ----------
        y : str or sequence[str]
            Column name(s) used for box values. One trace is created per column.
        x : str or sequence or None
            Optional grouping/category values. If None, boxes are not grouped.
        name : str, optional
            Base trace name. If multiple y columns are provided, the column
            name is appended automatically.
        boxmean : str or bool, optional
            Whether to display the mean marker:
            - True → show mean
            - "sd" → show mean + standard deviation
            - False/None → do not show mean
        orientation : str
            "v" for vertical (default), "h" for horizontal boxplots.
        boxpoints : bool or str
            Show underlying data points. Valid Plotly values:
            - False (default) → hide points
            - "all", "outliers", "suspectedoutliers"
        jitter : float
            Jitter applied to boxpoints to reduce overlap.

        Returns
        -------
        Boxplot
            self
        """
        y_cols = ensure_list(y)
        self._ensure_columns_exist(y_cols)

        if x is None:
            x_vals = None
        else:
            x_vals = self._coerce_x_values(x)

        for col in y_cols:
            trace_name = name or str(col)
            if name and len(y_cols) > 1:
                trace_name = f"{name} - {col}"

            trace = go.Box(
                y=self.data[col] if orientation == "v" else None,
                x=x_vals if orientation == "v" else self.data[col],
                name=trace_name,
                marker={"color": self._next_color()},
                boxmean=boxmean,
                orientation=orientation,
                meta=self._meta_for(y_col=col),
                boxpoints=boxpoints,
                jitter=jitter,
            )
            self.traces.append(trace)

        return self


class Indicator(Chart):
    """Indicator chart builder."""

    def create(
        self,
        value: Union[str, float, int],
        title: Optional[str] = None,
        mode: str = "number",
        delta_reference: Optional[Union[str, float, int]] = None,
        number_format: Optional[str] = None,
        gauge: bool = False,
        gauge_shape: Optional[str] = None,
        gauge_range: Optional[Sequence[float]] = None,
        gauge_bar_color: Optional[str] = None,
        gauge_steps: Optional[List[dict]] = None,
        threshold_value: Optional[float] = None,
        domain: Optional[dict] = None,
    ) -> "Indicator":
        """
        Create a Plotly Indicator trace.

        Parameters
        ----------
        value : str or number
            Column name or static numeric value displayed by the indicator.
        title : str, optional
            Text displayed under the indicator.
        mode : str
            Indicator display mode. Valid combinations:
            "number", "delta", "gauge", "number+delta",
            "gauge+number", "gauge+number+delta".
        delta_reference : str or number, optional
            Reference value for computing the delta. Required when using a mode
            that includes "delta".
        number_format : str, optional
            Numeric formatting string (e.g. ".2f").
        gauge : bool
            If True, forces inclusion of gauge settings. The mode must support a gauge.
        gauge_shape : str, optional
            Gauge shape ("angular", "bullet", etc.).
        gauge_range : sequence[float], optional
            [min, max] axis range for the gauge.
        gauge_bar_color : str, optional
            Fill color of the gauge bar.
        gauge_steps : list[dict], optional
            Gauge steps, each containing {"range": [...], "color": ...}.
        threshold_value : float, optional
            Value at which to draw a gauge threshold line.
        domain : dict, optional
            Plotly domain definition {"x": [...], "y": [...]}.

        Returns
        -------
        Indicator
            self
        """
        # resolve value
        if isinstance(value, str):
            if value not in self.data.columns:
                raise KeyError(f"Column '{value}' not in dataframe.")
            val = float(self.data[value].iloc[-1])
        else:
            val = float(value)

        # resolve delta reference
        delta_cfg = None
        if "delta" in mode:
            if delta_reference is None:
                raise ValueError("delta_reference required when using delta mode.")

            if isinstance(delta_reference, str):
                if delta_reference not in self.data.columns:
                    raise KeyError(f"Column '{delta_reference}' not in dataframe.")
                delta_ref_val = float(self.data[delta_reference].iloc[-1])
            else:
                delta_ref_val = float(delta_reference)

            delta_cfg = {"reference": delta_ref_val}

        # gauge configuration
        gauge_cfg = None
        if gauge or "gauge" in mode:
            gauge_cfg = {"axis": {}}
            if gauge_range is not None:
                gauge_cfg["axis"]["range"] = list(gauge_range)
            if gauge_bar_color is not None:
                gauge_cfg["bar"] = {"color": gauge_bar_color}
            if gauge_steps is not None:
                gauge_cfg["steps"] = gauge_steps
            if gauge_shape is not None:
                gauge_cfg["shape"] = gauge_shape
            if threshold_value is not None:
                gauge_cfg["threshold"] = {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": threshold_value,
                }

        # number formatting
        number_cfg = None
        if number_format:
            number_cfg = {"valueformat": number_format}

        trace = go.Indicator(
            mode=mode,
            value=val,
            delta=delta_cfg,
            number=number_cfg,
            gauge=gauge_cfg,
            title={"text": title} if title else None,
            domain=domain,
            meta=self._meta_for(value_source=value),
        )

        self.traces.append(trace)
        return self


class Waterfall(Chart):
    """Waterfall chart builder."""

    def create(
        self,
        x: Sequence[Union[str, float, int]],
        y: Sequence[Union[str, float, int]],
        name: str = "Waterfall",
        increasing_color: str = "#2ecc71",
        decreasing_color: str = "#e74c3c",
        total_color: str = "#3498db",
        connector_width: int = 1,
        title: Optional[str] = None,
        text: Union[bool, Sequence[str], str, None] = None,
        textposition: str = "outside",
        y_format: Optional[str] = None,
    ) -> "Waterfall":
        """
        Create a Waterfall trace.

        Parameters
        ----------
        x : sequence
            Labels for each step. Any label equal to "total" (case-insensitive)
            is treated as a total bar.
        y : sequence
            Numeric values or column names. Totals may be given as "total".
        name : str
            Trace name.
        increasing_color : str
            Fill color for positive steps.
        decreasing_color : str
            Fill color for negative steps.
        total_color : str
            Fill color for total steps.
        connector_width : int
            Width of connectors between bars.
        title : str, optional
            Chart title.
        text : bool, sequence[str], str, or None
            Controls per-bar text:
            - True → show numeric values
            - list[str] → custom labels (must match length of x)
            - "percent" → display % of total
            - None → no labels
        textposition : str
            Plotly text position (default "outside").
        y_format : str, optional
            Tick format specifier for the y-axis (e.g., ",.2f").

        Returns
        -------
        Waterfall
            self
        """
        # resolve y
        resolved_y = []
        for v in y:
            if isinstance(v, str):
                if v.lower() == "total":
                    resolved_y.append(0)
                else:
                    if v not in self.data.columns:
                        raise KeyError(f"Column '{v}' not in dataframe.")
                    resolved_y.append(float(self.data[v].iloc[-1]))
            else:
                resolved_y.append(float(v))

        # measure types
        measures = ["total" if str(lbl).lower() == "total" else "relative" for lbl in y]

        # axis formatting
        if y_format:
            self._add_axis_update({"yaxis": {"tickformat": y_format}})

        # text formatting
        text_values = None
        if isinstance(text, (list, tuple)):
            if len(text) != len(x):
                raise ValueError("Length of text list must match length of x/y.")
            text_values = list(text)
        elif text is True:
            text_values = [str(v) for v in resolved_y]
        elif isinstance(text, str):
            if text.lower() == "percent":
                total = sum(v for v in resolved_y if isinstance(v, (int, float)))
                text_values = [
                    f"{v/total:.1%}" if total != 0 else "" for v in resolved_y
                ]
            else:
                raise ValueError(
                    "Unsupported text mode. Use True, list[str], or 'percent'."
                )

        trace = go.Waterfall(
            name=name,
            x=list(x),
            y=resolved_y,
            measure=measures,
            increasing={"marker": {"color": increasing_color}},
            decreasing={"marker": {"color": decreasing_color}},
            totals={"marker": {"color": total_color}},
            connector={"line": {"width": connector_width}},
            text=text_values,
            textposition=textposition if text_values is not None else None,
            meta=self._meta_for(value_source=y),
        )

        self.traces.append(trace)

        if title:
            self.update_layout(title=title)
        if y_format:
            self.update_yaxes(tickformat=y_format)

        return self
