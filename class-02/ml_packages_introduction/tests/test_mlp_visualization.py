"""Unit tests for mlp_visualization."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
import mlp_visualization


def make_mlp() -> SimpleNamespace:
    return SimpleNamespace(
        coefs_=[
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0], [6.0]]),
        ],
        n_outputs_=1,
    )


def test_get_mlp_min_max_width_returns_smallest_and_largest_layer() -> None:
    assert mlp_visualization.get_mlp_min_max_width(make_mlp()) == (1, 2)


def test_print_ascii_mlp_prints_one_line_per_layer(capsys) -> None:
    mlp_visualization.print_ascii_mlp(make_mlp())

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert lines[0].rstrip().endswith(" .  .")
    assert lines[1].rstrip().endswith(" .")
    assert lines[2].rstrip().endswith(" .")


def test_build_neurons_dataframe_centers_layers() -> None:
    neurons = mlp_visualization.build_neurons_dataframe(make_mlp())

    expected = pd.DataFrame(
        {
            "layer_number": [0, 0, 1, 1, 2],
            "neuron_number": [0, 1, 0, 1, 0],
            "x": [0, 0, 1, 1, 2],
            "y": [0.0, 1.0, 0.0, 1.0, 0.5],
        }
    )
    pd.testing.assert_frame_equal(neurons, expected)


def test_build_weights_dataframe_contains_each_connection() -> None:
    mlp = make_mlp()
    neurons = mlp_visualization.build_neurons_dataframe(mlp)

    weights = mlp_visualization.build_weights_dataframe(mlp, neurons)

    assert len(weights) == 6
    np.testing.assert_array_equal(
        np.array(weights.loc[0, "x"]).ravel(), [0, 1]
    )
    np.testing.assert_array_equal(
        np.array(weights.loc[0, "y"]).ravel(), [0.0, 0.0]
    )
    assert weights.loc[0, "weight"] == 1.0
    assert weights.loc[5, "from_layer_number"] == 1
    assert weights.loc[5, "to_layer_number"] == 2
    assert weights.loc[5, "weight"] == 6.0


def test_build_colormap_adds_colors_and_returns_normalization() -> None:
    weights = pd.DataFrame({"weight": [-1.0, 0.0, 1.0]})

    cmap, norm = mlp_visualization.build_colormap(weights, "viridis")

    assert cmap.name == "viridis"
    assert norm.vmin == -1.0
    assert norm.vmax == 1.0
    assert len(weights["colors"]) == 3
    np.testing.assert_array_equal(weights.loc[0, "colors"], cmap(norm(-1.0)))


def test_plot_weights_plots_each_weight(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        mlp_visualization.plt,
        "plot",
        lambda x, y, **kwargs: calls.append((x, y, kwargs)),
    )
    weights = pd.DataFrame(
        {
            "x": [[0, 1], [1, 2]],
            "y": [[0.0, 1.0], [1.0, 0.5]],
            "colors": ["red", "blue"],
        }
    )

    mlp_visualization.plot_weights(weights, linewidth=2.0, line_alpha=0.5)

    assert calls == [
        (
            [0, 1],
            [0.0, 1.0],
            {"color": "red", "linewidth": 2.0, "alpha": 0.5, "zorder": 0},
        ),
        (
            [1, 2],
            [1.0, 0.5],
            {"color": "blue", "linewidth": 2.0, "alpha": 0.5, "zorder": 0},
        ),
    ]


def test_build_dataframes_returns_neurons_and_weights() -> None:
    neurons, weights = mlp_visualization.build_dataframes(make_mlp())

    assert len(neurons) == 5
    assert len(weights) == 6
    assert list(neurons.columns) == ["layer_number", "neuron_number", "x", "y"]
    assert "weight" in weights.columns
