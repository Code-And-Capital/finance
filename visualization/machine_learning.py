from visualization.charts import Contour, Scatter
from visualization.figure import Figure
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple


class KNNPlotter:
    """
    Visualizes K-Nearest Neighbors (KNN) classifier decision boundaries
    alongside training and test data splits.

    This class handles the end-to-end process of:
    1. Creating a mesh grid over the feature space.
    2. Computing decision probabilities via the provided classifier.
    3. Plotting decision contours and scatter plots of the data.
    4. Applying layout styling and displaying the resulting figure.

    Parameters
    ----------
    clf : estimator object
        A trained classifier implementing `predict_proba`.
    x : array-like of shape (n_samples, n_features)
        Complete dataset used to define plot boundaries.
    x_train : array-like of shape (n_train, n_features)
        Training feature samples.
    x_test : array-like of shape (n_test, n_features)
        Test feature samples.
    y_train : array-like of shape (n_train,)
        Training labels.
    y_test : array-like of shape (n_test,)
        Test labels.
    margin : float, default=0.25
        Padding added around the feature space when computing boundaries.
    mesh_size : float, default=0.02
        Step size for the mesh grid resolution (smaller = smoother surface).

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    >>> plotter = KNNPlotter(knn, X, X_train, X_test, y_train, y_test)
    >>> plotter.plot(title="KNN Classifier Boundary")
    """

    def __init__(
        self,
        clf,
        x,
        x_train,
        x_test,
        y_train,
        y_test,
        margin: float = 0.25,
        mesh_size: float = 0.02,
    ):
        self.classifier = clf
        self.x = x
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.margin = margin
        self.mesh_size = mesh_size
        self.fig: Optional[Figure] = None
        self._computed = False

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------
    def _create_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a 2D mesh grid that spans the feature space.

        Returns
        -------
        xx : np.ndarray
            Grid of x-coordinates.
        yy : np.ndarray
            Grid of y-coordinates.
        xrange : np.ndarray
            Range of x-axis values.
        yrange : np.ndarray
            Range of y-axis values.
        """
        x_min, x_max = (
            self.x[:, 0].min() - self.margin,
            self.x[:, 0].max() + self.margin,
        )
        y_min, y_max = (
            self.x[:, 1].min() - self.margin,
            self.x[:, 1].max() + self.margin,
        )
        xrange = np.arange(x_min, x_max, self.mesh_size)
        yrange = np.arange(y_min, y_max, self.mesh_size)
        xx, yy = np.meshgrid(xrange, yrange)
        return xx, yy, xrange, yrange

    def _compute_boundaries(self) -> None:
        """
        Compute the classifier’s decision boundary probabilities over the mesh grid.

        This method predicts the probability of class 1 for each grid point,
        reshaping the output into the grid shape.
        """
        self.xx, self.yy, self.xrange, self.yrange = self._create_mesh()
        z = self.classifier.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])[:, 1]
        self.z = z.reshape(self.xx.shape)
        self._computed = True

    def _make_scatter(self, x, y, label: int, split: str, symbol: str) -> Scatter:
        """
        Construct a consistent `Scatter` chart for a given label/split.

        Parameters
        ----------
        x, y : array-like
            Feature coordinates to plot.
        label : int
            Class label (0 or 1).
        split : {'Train', 'Test'}
            Dataset split name for labeling the plot.
        symbol : str
            Marker symbol type (e.g., 'circle', 'square-dot').

        Returns
        -------
        scatter : Scatter
            Configured Scatter chart ready to be added to a Figure.
        """
        df = pd.DataFrame({"x": x, f"{split} | Label {label}": y})
        scatter = Scatter(df)
        scatter.create(
            x="x", y=f"{split} | Label {label}", mode="markers", marker_symbol=symbol
        )
        scatter.marker_style(size=12, line_width=1.5, color="lightyellow")
        return scatter

    def _make_contour(self) -> Contour:
        """
        Create a `Contour` chart representing the decision surface.

        Returns
        -------
        contour : Contour
            Configured contour chart for probability surface visualization.
        """
        df = pd.DataFrame()
        contour = Contour(df)
        contour.create(
            x=self.xrange,
            y=self.yrange,
            z=self.z,
            showscale=False,
            colorscale="RdBu",
            opacity=0.4,
            name="Decision Surface",
        )
        return contour

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def build_figure(self) -> "KNNPlotter":
        """
        Construct the composite figure with contour and scatter overlays.

        Returns
        -------
        self : KNNPlotter
            The current instance, allowing method chaining.
        """
        if not self._computed:
            self._compute_boundaries()

        self.fig = Figure()

        # Add scatter plots
        for label in [0, 1]:
            self.fig.add_chart(
                self._make_scatter(
                    self.x_train[self.y_train == str(label), 0],
                    self.x_train[self.y_train == str(label), 1],
                    label,
                    "Train",
                    "square" if label == 0 else "circle",
                )
            )
            self.fig.add_chart(
                self._make_scatter(
                    self.x_test[self.y_test == str(label), 0],
                    self.x_test[self.y_test == str(label), 1],
                    label,
                    "Test",
                    "square-dot" if label == 0 else "circle-dot",
                )
            )

        # Add decision contour
        self.fig.add_chart(self._make_contour())

        return self

    def apply_layout(
        self,
        *,
        title: Optional[str] = "KNN Decision Boundary",
        template: str = "plotly_dark",
        font_size: int = 14,
        margin: Dict[str, int] = dict(l=75, r=50, t=60, b=45),
        showlegend: bool = True,
        height: int = 700,
        width: int = 900,
    ) -> "KNNPlotter":
        """
        Apply consistent layout and styling options to the figure.

        Parameters
        ----------
        title : str, optional
            Title of the figure.
        template : str, default='plotly_dark'
            Plotly template style to apply.
        font_size : int, default=14
            Global font size.
        margin : dict, default=dict(l=75, r=50, t=60, b=45)
            Margins for layout spacing.
        showlegend : bool, default=True
            Whether to display legend.
        height, width : int, optional
            Pixel dimensions of the plot window.

        Returns
        -------
        self : KNNPlotter
            The current instance for chaining.
        """
        if not self.fig:
            raise RuntimeError("Figure not built. Call build_figure() first.")
        self.fig.layout(
            title=title,
            template=template,
            font_size=font_size,
            margin=margin,
            showlegend=showlegend,
            height=height,
            width=width,
        )
        return self

    def show(self) -> None:
        """
        Display the interactive plot.

        If no figure has been built, this method automatically
        constructs and styles one before showing it.
        """
        if not self.fig:
            self.build_figure().apply_layout()
        self.fig.show()

    def plot(self, *, layout: Optional[Dict[str, Any]] = None) -> None:
        """
        End-to-end convenience method to build, style, and show the plot.

        Parameters
        ----------
        layout : dict, optional
            Custom layout arguments forwarded to :meth:`apply_layout`.

        Notes
        -----
        This is equivalent to calling:
            `build_figure().apply_layout(**layout).show()`
        """
        self.build_figure()
        if layout:
            self.apply_layout(**layout)
        else:
            self.apply_layout()
        self.show()
