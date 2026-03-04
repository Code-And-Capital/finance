import visualization


def test_visualization_exports_include_new_public_symbols():
    assert hasattr(visualization, "Figure")
    assert hasattr(visualization, "ScatterGeo")
    assert hasattr(visualization, "KNNPlotter")

    exported = set(getattr(visualization, "__all__", []))
    assert "Figure" in exported
    assert "ScatterGeo" in exported
    assert "KNNPlotter" in exported
