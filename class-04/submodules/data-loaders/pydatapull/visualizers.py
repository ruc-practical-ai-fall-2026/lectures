from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torch import Tensor
from torch import float32 as TorchFloat32
from torch import no_grad
from torch import tensor as TorchTensor

PlotFunction = Callable[[np.ndarray], np.ndarray]
TorchPlotFunction = Callable[[Tensor], Tensor]


def scatter_plot_dataset(
    x_features: np.ndarray,
    y_labels: np.ndarray,
    x_plot_dimension: int = 0,
    y_plot_dimension: int = 1,
) -> Tuple[Figure, plt.Axes]:
    """Scatter plot a tabular dataset.

    The user selects which dimensions to plot. The defaults are 0 and 1,
    i.e., the 0th and 1st column of the dataset. It is assumed that x_features
    is an NxM matrix where rows are samples and columns are dimensions. It is
    assumed y_labels is an N-element vector with a discrete numerical label
    for each sample. It is assumed the data has at least two dimensions (M>=2).

    **NOTE**: No input checking is performed for the user. Look before you
    leap!

    Args:
        x_features: An NxM matrix of features where each row is a sample and
            each column is a feature. Must have at least 2 columns.
        y_labels: An N-element vector where each element is the numerical
            label of the corresponding sample.
        x_plot_dimension: The dimension to use for the X-axis. Must be an
            integer. Defaults to 0.
        y_plot_dimension: The dimension to use for the Y-axis. Must be an
            integer. Defaults to 1.

    Returns:
        Returns the figure handle for the plot as a tuple where the first
            entry is the figure handle and the second is the axes handle.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(
        x_features[:, x_plot_dimension],
        x_features[:, y_plot_dimension],
        c=y_labels,
        alpha=0.5,
        cmap="viridis",
        s=20,
    )
    ax.set_xlabel("$x_{}$".format(x_plot_dimension), usetex=True)
    ax.set_ylabel("$x_{}$".format(y_plot_dimension), usetex=True)
    plt.tight_layout()
    plt.show()
    return fig, ax


def scatter_plot_dataset_torch(
    x_features: Tensor,
    y_labels: Tensor,
    x_plot_dimension: int = 0,
    y_plot_dimension: int = 1,
) -> Tuple[Figure, plt.Axes]:
    """Scatter plot a tabular dataset.

    The user selects which dimensions to plot. The defaults are 0 and 1,
    i.e., the 0th and 1st column of the dataset. It is assumed that x_features
    is an NxM matrix where rows are samples and columns are dimensions. It is
    assumed y_labels is an N-element vector with a discrete numerical label
    for each sample. It is assumed the data has at least two dimensions (M>=2).

    **NOTE**: No input checking is performed for the user. Look before you
    leap!

    Args:
        x_features: An NxM tensor of features where each row is a sample and
            each column is a feature. Must have at least 2 columns.
        y_labels: An Nx1-element tensor where each element is the numerical
            label of the corresponding sample.
        x_plot_dimension: The dimension to use for the X-axis. Must be an
            integer. Defaults to 0.
        y_plot_dimension: The dimension to use for the Y-axis. Must be an
            integer. Defaults to 1.

    Returns:
        Returns the figure handle for the plot as a tuple where the first
            entry is the figure handle and the second is the axes handle.
    """
    fig, ax = scatter_plot_dataset(
        x_features.detach().numpy(),
        y_labels.detach().numpy(),
        x_plot_dimension,
        y_plot_dimension,
    )
    return fig, ax


def plot_2d_decision_surface_and_features(
    x_features: np.ndarray,
    y_labels: np.ndarray,
    plot_function: PlotFunction,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    n_grid_points: int = 100,
    x_plot_dimension=0,
    y_plot_dimension=1,
) -> Tuple[Figure, plt.Axes]:
    """Plots the decision surface of an algorithm in 2D feature space.

    The user selects which dimensions to plot. The defaults are 0 and 1,
    i.e., the 0th and 1st column of the dataset. It is assumed that x_features
    is an NxM matrix where rows are samples and columns are dimensions. It is
    assumed y_labels is an N-element vector with a discrete numerical label
    for each sample. It is assumed the data has at least two dimensions (M>=2).

    **NOTE**: No input checking is performed for the user. Look before you
    leap!

    Args:
        x_features: An NxM matrix of features where each row is a sample and
            each column is a feature. Must have at least 2 columns.
        y_labels: An N-element vector where each element is the numerical
            label of the corresponding sample.
        plot_function: Callable function with the signature
            Callable[[np.ndarray], np.ndarray] (i.e., accepts an array
            of the same shape as x_features as input and returns an array
            of the same shape as y_labels).
        x_range: Tuple specifying the minimum and maximum x values.
        y_range: Tuple specifying the minimum and maximum y values.
        n_grid_points: The number of points to use per dimension to create the
            grid the function is evaluated over.
        x_plot_dimension: The dimension to use for the X-axis. Must be an
            integer. Defaults to 0.
        y_plot_dimension: The dimension to use for the Y-axis. Must be an
            integer. Defaults to 1.

    Returns:
        Returns the figure handle for the plot as a tuple where the first
            entry is the figure handle and the second is the axes handle.
    """
    x: np.ndarray = np.linspace(x_range[0], x_range[1], n_grid_points)
    y: np.ndarray = np.linspace(x_range[0], x_range[1], n_grid_points)
    xx, yy = np.meshgrid(x, y)

    xy: np.ndarray = np.column_stack((xx.ravel(), yy.ravel()))
    zz: np.ndarray = plot_function(xy).reshape(xx.shape)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    heatmap = ax.imshow(
        zz,
        extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
        origin="lower",
        cmap="Grays",
    )

    cbar = fig.colorbar(heatmap)
    cbar.set_label("Decision Value", usetex=True)

    ax.scatter(
        x_features[:, x_plot_dimension],
        x_features[:, y_plot_dimension],
        c=y_labels,
        cmap="viridis",
        s=20,
        alpha=0.5,
    )
    ax.set_xlabel("$x_{}$".format(x_plot_dimension), usetex=True)
    ax.set_ylabel("$x_{}$".format(y_plot_dimension), usetex=True)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_2d_decision_surface_and_features_torch(
    x_features: Tensor,
    y_labels: Tensor,
    plot_function: TorchPlotFunction,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    n_grid_points: int = 100,
    x_plot_dimension=0,
    y_plot_dimension=1,
) -> Tuple[Figure, plt.Axes]:
    """Plots the decision surface in 2D feature space using torch types.

    The user selects which dimensions to plot. The defaults are 0 and 1,
    i.e., the 0th and 1st column of the dataset. It is assumed that x_features
    is an NxM matrix where rows are samples and columns are dimensions. It is
    assumed y_labels is an N-element vector with a discrete numerical label
    for each sample. It is assumed the data has at least two dimensions (M>=2).

    **NOTE**: No input checking is performed for the user. Look before you
    leap!

    Args:
        x_features: An NxM tensor of features where each row is a sample and
            each column is a feature. Must have at least 2 columns.
        y_labels: An Nx1 tensor where each element is the numerical label of
            the corresponding sample.
        plot_function: Callable function with the signature
            Callable[[Tensor], Tensor] (i.e., accepts a Tensor of the same
            shape as x_features as input and returns a tensor of the same shape
            as y_labels).
        x_range: Tuple specifying the minimum and maximum x values.
        y_range: Tuple specifying the minimum and maximum y values.
        n_grid_points: The number of points to use per dimension to create the
            grid the function is evaluated over.
        x_plot_dimension: The dimension to use for the X-axis. Must be an
            integer. Defaults to 0.
        y_plot_dimension: The dimension to use for the Y-axis. Must be an
            integer. Defaults to 1.

    Returns:
        Returns the figure handle for the plot as a tuple where the first
            entry is the figure handle and the second is the axes handle.
    """
    forward = (
        lambda x: plot_function(TorchTensor(x, dtype=TorchFloat32))
        .detach()
        .numpy()
    )
    with no_grad():
        fig, ax = plot_2d_decision_surface_and_features(
            x_features=x_features.detach().numpy(),
            y_labels=y_labels.detach().numpy(),
            plot_function=forward,
            x_range=x_range,
            y_range=y_range,
            n_grid_points=n_grid_points,
            x_plot_dimension=x_plot_dimension,
            y_plot_dimension=y_plot_dimension,
        )
    return fig, ax
