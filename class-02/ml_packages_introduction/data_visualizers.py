from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Function = Callable[[np.ndarray], np.ndarray]


def scatter_plot_dataset(
    x_features: np.ndarray,
    y_labels: np.ndarray,
    x_plot_dimension: int,
    y_plot_dimension: int,
) -> None:
    """Scatter plots a dataset."""
    plt.scatter(
        x_features[:, x_plot_dimension],
        x_features[:, y_plot_dimension],
        c=y_labels,
        cmap="viridis",
        s=20,
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    plt.savefig("dataset.jpg")


def plot_2d_decision_surface_and_features(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.float64],
    x_range: Tuple,
    y_range: Tuple,
    n_grid_points: int,
    plot_function: Function,
):
    """Plots the decision surface of an algorithm in 2D feature space."""
    x: NDArray[np.float64] = np.linspace(x_range[0], x_range[1], n_grid_points)
    y: NDArray[np.float64] = np.linspace(x_range[0], x_range[1], n_grid_points)
    xx, yy = np.meshgrid(x, y)

    xy: NDArray[np.float64] = np.column_stack((xx.ravel(), yy.ravel()))
    zz: NDArray[np.float64] = plot_function(xy).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(
        zz,
        extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
        origin="lower",
        cmap="viridis",
    )

    cbar = plt.colorbar(heatmap)
    cbar.set_label("NN Output")

    plt.scatter(
        x_features[:, 0],
        x_features[:, 1],
        c=y_labels,
        cmap="brg",
        s=100,
    )

    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.savefig("nn_output.jpg")
