import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any


class Figure:
    """
    High-level manager for building multi-subplot Plotly figures
    from Chart objects. Handles subplot layout creation, trace assignment,
    consistent coloring, and figure rendering.

    Example
    -------
    >>> fig = Figure(rows=2, cols=2, row_heights=[0.3, 0.7])
    >>> fig.add_chart(chart1, row=1, col=1)
    >>> fig.add_chart(chart2, row=2, col=1, colspan=2)
    >>> fig.layout(title="Dashboard Overview")
    >>> fig.build().show()
    """

    def __init__(self, rows: int = 1, cols: int = 1, **kwargs: Any) -> None:
        self.rows: int = rows
        self.cols: int = cols
        self.init_kwargs: Dict[str, Any] = kwargs
        self.fig: Optional[go.Figure] = None
        self.specs: Optional[List[List[Optional[Dict[str, Any]]]]] = None

        self._charts: List[Dict[str, Any]] = []
        self._layout_updates: Dict[str, Any] = {}
        self._used_legend_names: set[str] = set()
        self._global_color_map: Dict[str, str] = {}
        self._color_sequence: List[str] = px.colors.qualitative.Plotly
        self._color_index: int = 0

    # -------------------------------------------------------------------------
    # Chart Registration
    # -------------------------------------------------------------------------
    def add_chart(
        self, chart: Any, row: int = 1, col: int = 1, rowspan: int = 1, colspan: int = 1
    ) -> "Figure":
        """
        Register a chart and its placement within the subplot grid.

        Parameters
        ----------
        chart : Any
            Object expected to expose `.traces`, `.chart_id`, and update lists.
        row, col : int
            Subplot grid position (1-indexed).
        rowspan, colspan : int
            Number of rows or columns this chart spans.

        Returns
        -------
        Figure
            Self reference for chaining.
        """
        self._charts.append(
            {
                "chart": chart,
                "row": row,
                "col": col,
                "rowspan": rowspan,
                "colspan": colspan,
            }
        )
        return self

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------
    def _auto_build_specs(self) -> List[List[Optional[Dict[str, Any]]]]:
        """Automatically constructs subplot specs from registered charts."""
        specs = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        occupied: set[tuple[int, int]] = set()

        for entry in self._charts:
            row = entry["row"] - 1
            col = entry["col"] - 1
            rs = entry["rowspan"]
            cs = entry["colspan"]
            chart = entry["chart"]

            raw_type = getattr(chart, "default_type", chart.__class__.__name__.lower())
            trace_type = (
                "xy"
                if raw_type in ("line", "area", "scatter", "bar", "histogram", "ohlc")
                else raw_type
            )

            # Bounds check
            if row + rs > self.rows or col + cs > self.cols:
                raise ValueError(
                    f"Chart at ({row+1}, {col+1}) spans outside subplot grid."
                )

            specs[row][col] = {"type": trace_type}
            if cs > 1:
                specs[row][col]["colspan"] = cs
            if rs > 1:
                specs[row][col]["rowspan"] = rs

            # Mark occupied cells
            for r in range(row, row + rs):
                for c in range(col, col + cs):
                    occupied.add((r, c))
                    if not (r == row and c == col):
                        specs[r][c] = None
        return specs

    # -------------------------------------------------------------------------
    # Color & Legend Assignment
    # -------------------------------------------------------------------------
    def _assign_global_color_and_legend(self, trace):
        """
        Assign consistent colors for traces and deduplicate legends globally.
        Candlestick traces are left with default colors.
        """
        # Candlestick — keep defaults
        if isinstance(trace, go.Candlestick):
            name = getattr(trace, "name", None)
            if name and name not in self._used_legend_names:
                self._used_legend_names.add(name)
            trace.showlegend = False
            return trace

        # Multi-category traces
        categories = None
        if hasattr(trace, "labels") and trace.labels is not None:
            categories = trace.labels
        elif hasattr(trace, "names") and trace.names is not None:
            categories = trace.names

        if categories is not None:
            trace.marker = trace.marker or {}
            colors: List[str] = []
            new_categories_for_legend: List[str] = []

            for cat in categories:
                if cat in self._global_color_map:
                    color = self._global_color_map[cat]
                else:
                    color = self._color_sequence[
                        self._color_index % len(self._color_sequence)
                    ]
                    self._color_index += 1
                    self._global_color_map[cat] = color
                colors.append(color)

                if cat not in self._used_legend_names:
                    new_categories_for_legend.append(cat)
                    self._used_legend_names.add(cat)

            # Assign colors safely (Pie uses `colors` only)
            if hasattr(trace.marker, "colors"):
                trace.marker.colors = colors
            elif trace.__class__.__name__.lower() == "pie":
                trace.marker.colors = colors
            else:
                trace.marker.color = colors

            trace.showlegend = len(new_categories_for_legend) > 0
            return trace

        # Single-category traces
        name = getattr(trace, "name", None)
        if name in self._global_color_map:
            color = self._global_color_map[name]
        else:
            color = self._color_sequence[self._color_index % len(self._color_sequence)]
            self._color_index += 1
            self._global_color_map[name] = color

        # Apply color based on trace type
        if hasattr(trace, "marker") and trace.marker is not None:
            trace.marker.color = color
        elif hasattr(trace, "line") and trace.line is not None:
            trace.line.color = color
        elif hasattr(trace, "fillcolor"):
            trace.fillcolor = color

        # Legend deduplication
        if name in self._used_legend_names:
            trace.showlegend = False
        else:
            trace.showlegend = True
            if name:
                self._used_legend_names.add(name)

        return trace

    # -------------------------------------------------------------------------
    # Build Process
    # -------------------------------------------------------------------------
    def build(self) -> "Figure":
        """
        Build and assemble the Plotly subplot figure.
        Applies all trace, layout, and axis updates stored in the charts.

        Returns
        -------
        Figure
            Self reference for chaining or display.
        """
        self._used_legend_names.clear()
        self._global_color_map.clear()
        self._color_index = 0

        self.specs = self._auto_build_specs()
        self.fig = make_subplots(
            rows=self.rows, cols=self.cols, specs=self.specs, **self.init_kwargs
        )

        for entry in self._charts:
            chart = entry["chart"]
            row, col = entry["row"], entry["col"]

            # --- Add traces ---
            for trace in chart.traces:
                trace = self._assign_global_color_and_legend(trace)
                self.fig.add_trace(trace, row=row, col=col)

            # --- Trace-level updates ---
            for update in chart.trace_updates:
                for i, trace in enumerate(self.fig.data):
                    if update.get("chart_id") == chart.chart_id:
                        trace.update(update["update"])

            # --- Layout updates ---
            for update in chart.layout_updates:
                if update.get("chart_id") == chart.chart_id:
                    u = {k: v for k, v in update.items() if k != "chart_id"}
                    self.fig.update_layout(**u)

            # --- Axis updates ---
            for update in chart.axis_updates:
                if update.get("chart_id") != chart.chart_id:
                    continue
                axis_index = (row - 1) * self.cols + col
                if "xaxis" in update:
                    axis_key = f"xaxis{'' if axis_index == 1 else axis_index}"
                    self.fig.layout[axis_key].update(update["xaxis"])
                if "yaxis" in update:
                    axis_key = f"yaxis{'' if axis_index == 1 else axis_index}"
                    self.fig.layout[axis_key].update(update["yaxis"])

        if self._layout_updates:
            self.fig.update_layout(**self._layout_updates)

        return self

    # -------------------------------------------------------------------------
    # Layout Management
    # -------------------------------------------------------------------------
    def layout(
        self,
        *,
        title: Optional[str] = None,
        template: str = "plotly_dark",
        font_size: int = 14,
        margin: Dict[str, int] = dict(l=75, r=50, t=60, b=45),
        showlegend: Optional[bool] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> "Figure":
        """
        Store or apply global figure layout parameters.

        Returns
        -------
        Figure
            Self reference for chaining.
        """
        updates = {
            "title": title,
            "template": template,
            "font": dict(size=font_size),
            "margin": margin,
            "showlegend": showlegend,
            "height": height,
            "width": width,
        }
        updates = {k: v for k, v in updates.items() if v is not None}

        if self.fig is not None:
            self.fig.update_layout(**updates)
        else:
            self._layout_updates.update(updates)
        return self

    # -------------------------------------------------------------------------
    # Axis Management
    # -------------------------------------------------------------------------
    def update_axes(
        self,
        *,
        x_tickformat: Optional[str] = None,
        y_tickformat: Optional[str] = None,
        showgrid_x: Optional[bool] = None,
        showgrid_y: Optional[bool] = None,
    ) -> "Figure":
        """Convenience wrapper for global axis styling."""
        if not self.fig:
            raise RuntimeError("Figure must be built before updating axes.")

        if x_tickformat:
            self.fig.update_xaxes(tickformat=x_tickformat)
        if y_tickformat:
            self.fig.update_yaxes(tickformat=y_tickformat)
        if showgrid_x is not None:
            self.fig.update_xaxes(showgrid=showgrid_x)
        if showgrid_y is not None:
            self.fig.update_yaxes(showgrid=showgrid_y)
        return self

    # -------------------------------------------------------------------------
    # Rendering & Export
    # -------------------------------------------------------------------------
    def show(self) -> None:
        """Display the figure."""
        if self.fig is None:
            self.build()
        self.fig.show()

    def save(self, filename: str, file_format: str = "html") -> None:
        """Save the figure as HTML or static image."""
        if self.fig is None:
            self.build()

        file_format = file_format.lower()
        if file_format == "html":
            self.fig.write_html(filename)
        elif file_format in ["png", "jpg", "jpeg"]:
            self.fig.write_image(filename)
        else:
            raise ValueError("file_format must be 'html', 'png', 'jpg', or 'jpeg'")
