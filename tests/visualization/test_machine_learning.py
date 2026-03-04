import numpy as np

from visualization.machine_learning import KNNPlotter


class DummyClassifier:
    """Minimal classifier stub with predict_proba for boundary plotting tests."""

    def predict_proba(self, x):
        probs = np.full((x.shape[0], 2), 0.5, dtype=float)
        return probs


def test_knn_plotter_build_figure_with_numeric_labels():
    x = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.8, 0.9],
            [1.0, 1.1],
        ]
    )
    x_train = x[:2]
    x_test = x[2:]
    y_train = np.array([0, 1])
    y_test = np.array([0, 1])

    plotter = KNNPlotter(
        DummyClassifier(),
        x,
        x_train,
        x_test,
        y_train,
        y_test,
        mesh_size=0.5,
    )
    plotter.build_figure()

    assert plotter.fig is not None
    # 2 train scatters + 2 test scatters + 1 contour
    assert len(plotter.fig._charts) == 5


def test_knn_plotter_apply_layout_uses_default_margin_when_none():
    x = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.8, 0.9],
            [1.0, 1.1],
        ]
    )
    y = np.array([0, 1, 0, 1])
    plotter = KNNPlotter(DummyClassifier(), x, x, x, y, y, mesh_size=0.5)
    plotter.build_figure().apply_layout(margin=None)
    plotter.fig.build()

    assert plotter.fig.fig.layout.margin.l == 75
    assert plotter.fig.fig.layout.margin.r == 50
    assert plotter.fig.fig.layout.margin.t == 60
    assert plotter.fig.fig.layout.margin.b == 45
