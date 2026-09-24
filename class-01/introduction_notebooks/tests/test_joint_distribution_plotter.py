import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODULE_PATH = (
    Path(__file__).parent.parent / "joint_distribution_plotter.py"
)
SPEC = importlib.util.spec_from_file_location(
    "joint_distribution_plotter", MODULE_PATH
)
joint_distribution_plotter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(joint_distribution_plotter)


def test_compute_safe_pearson_coefficient_returns_correlation():
    covariance = np.array([[4, 2], [2, 4]])

    result = joint_distribution_plotter.compute_safe_pearson_coefficient(
        covariance
    )

    assert result == pytest.approx(0.5)


def test_compute_safe_pearson_coefficient_handles_zero_variance():
    covariance = np.array([[0, 0], [0, 4]])

    assert (
        joint_distribution_plotter.compute_safe_pearson_coefficient(covariance)
        == 0
    )


def test_confidence_ellipse_adds_patch_with_expected_size():
    figure, axes = plt.subplots()
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([-1.0, 0.0, 1.0])

    ellipse = joint_distribution_plotter.confidence_ellipse(
        x, y, axes, n_std=2
    )

    assert ellipse in axes.patches
    assert ellipse.width == pytest.approx(2 * np.sqrt(2))
    assert ellipse.height == pytest.approx(0)
    plt.close(figure)


def test_confidence_ellipse_rejects_different_sized_inputs():
    figure, axes = plt.subplots()

    with pytest.raises(ValueError, match="same size"):
        joint_distribution_plotter.confidence_ellipse(
            np.array([1, 2]), np.array([1]), axes
        )

    plt.close(figure)


def test_scatter_hist_draws_scatter_and_marginal_histograms():
    figure, (axes, hist_x, hist_y) = plt.subplots(1, 3)
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([1.0, 0.0, -1.0])

    joint_distribution_plotter.scatter_hist(x, y, axes, hist_x, hist_y)

    assert len(axes.collections) == 1
    assert len(hist_x.patches) > 0
    assert len(hist_y.patches) > 0
    plt.close(figure)


def test_plot_joint_distribution_creates_titled_plot_with_ellipses():
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([1.0, 0.0, -1.0])

    joint_distribution_plotter.plot_joint_distribution(x, y, "Test title")

    figure = plt.gcf()
    assert figure.axes[0].get_title() == "Test title"
    assert len(figure.axes[0].patches) == 3
    plt.close(figure)
