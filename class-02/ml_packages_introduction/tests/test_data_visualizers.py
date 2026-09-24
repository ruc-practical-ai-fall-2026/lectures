"""Unit tests for data_visualizers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import data_visualizers


def test_scatter_plot_dataset_plots_selected_dimensions_and_saves(
    monkeypatch,
) -> None:
    calls = {}

    monkeypatch.setattr(
        data_visualizers.plt,
        "scatter",
        lambda x, y, **kwargs: calls.update(x=x, y=y, kwargs=kwargs),
    )
    monkeypatch.setattr(
        data_visualizers.plt,
        "xlabel",
        lambda label: calls.update(xlabel=label),
    )
    monkeypatch.setattr(
        data_visualizers.plt,
        "ylabel",
        lambda label: calls.update(ylabel=label),
    )
    monkeypatch.setattr(
        data_visualizers.plt, "show", lambda: calls.update(show=True)
    )
    monkeypatch.setattr(
        data_visualizers.plt,
        "savefig",
        lambda filename: calls.update(filename=filename),
    )

    data_visualizers.scatter_plot_dataset(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.array([0.0, 1.0]),
        x_plot_dimension=2,
        y_plot_dimension=0,
    )

    np.testing.assert_array_equal(calls["x"], np.array([3.0, 6.0]))
    np.testing.assert_array_equal(calls["y"], np.array([1.0, 4.0]))
    np.testing.assert_array_equal(calls["kwargs"]["c"], np.array([0.0, 1.0]))
    assert calls["kwargs"]["cmap"] == "viridis"
    assert calls["kwargs"]["s"] == 20
    assert calls["xlabel"] == "X"
    assert calls["ylabel"] == "Y"
    assert calls["show"] is True
    assert calls["filename"] == "dataset.jpg"


def test_plot_2d_decision_surface_builds_grid_and_saves(monkeypatch) -> None:
    calls = {}

    monkeypatch.setattr(
        data_visualizers.plt,
        "figure",
        lambda **kwargs: calls.update(figure=kwargs),
    )
    monkeypatch.setattr(
        data_visualizers.plt,
        "imshow",
        lambda values, **kwargs: (
            calls.update(imshow=(values, kwargs)) or "heatmap"
        ),
    )
    monkeypatch.setattr(
        data_visualizers.plt,
        "colorbar",
        lambda heatmap: (
            calls.update(colorbar=heatmap)
            or type(
                "Colorbar",
                (),
                {
                    "set_label": lambda self, label: calls.update(
                        colorbar_label=label
                    )
                },
            )()
        ),
    )
    monkeypatch.setattr(
        data_visualizers.plt,
        "scatter",
        lambda x, y, **kwargs: calls.update(scatter=(x, y, kwargs)),
    )
    monkeypatch.setattr(
        data_visualizers.plt,
        "xlabel",
        lambda label: calls.update(xlabel=label),
    )
    monkeypatch.setattr(
        data_visualizers.plt,
        "ylabel",
        lambda label: calls.update(ylabel=label),
    )
    monkeypatch.setattr(
        data_visualizers.plt,
        "savefig",
        lambda filename: calls.update(filename=filename),
    )

    def plot_function(points: np.ndarray) -> np.ndarray:
        calls["points"] = points
        return points[:, 0] + points[:, 1]

    data_visualizers.plot_2d_decision_surface_and_features(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0.0, 1.0]),
        x_range=(-1.0, 1.0),
        y_range=(10.0, 20.0),
        n_grid_points=2,
        plot_function=plot_function,
    )

    np.testing.assert_array_equal(
        calls["points"],
        np.array([[-1.0, 10.0], [1.0, 10.0], [-1.0, 20.0], [1.0, 20.0]]),
    )
    assert calls["imshow"][1] == {
        "extent": (-1.0, 1.0, 10.0, 20.0),
        "origin": "lower",
        "cmap": "viridis",
    }
    assert calls["figure"] == {"figsize": (8, 6)}
    assert calls["colorbar"] == "heatmap"
    assert calls["colorbar_label"] == "NN Output"
    np.testing.assert_array_equal(
        calls["scatter"][2]["c"], np.array([0.0, 1.0])
    )
    assert calls["scatter"][2]["cmap"] == "brg"
    assert calls["scatter"][2]["s"] == 100
    assert calls["xlabel"] == "X_1"
    assert calls["ylabel"] == "X_2"
    assert calls["filename"] == "nn_output.jpg"
