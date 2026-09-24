"""Plot a joint distribution with containment ellipse.

This example is largely adapted from MatPlotLib's example here:
https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse


def compute_safe_pearson_coefficient(cov):
    """Compute Pearson Correlation Coefficient and set to zero if undefined.

    Uses the covariance matrix between two variables to compute the Pearson
    Correlation Coefficient (PCC) between them. A zero value is returned if the
    value of the PCC would have been undefined.

    Args:
        cov: 2x2 covariance matrix

    Returns:
        float
    """
    pearson_numerator = cov[0, 1]
    pearson_denominator = np.sqrt(cov[0, 0] * cov[1, 1])
    if pearson_denominator == 0:
        pearson = 0
    else:
        pearson = pearson_numerator / pearson_denominator
    return pearson


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """Plot the covariance confidence ellipse of x and y.

    Args:
        x: array-like, shape (n, ) input data.
        y: array-like, shape (n, ) input data.
        ax: matplotlib.axes.Axes object to draw the ellipse into.
        n_std: The number of standard deviations to determine ellipse radiuses.
        **kwargs: Forwarded to `~matplotlib.patches.Ellipse`

    Returns:
        matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)

    pearson = compute_safe_pearson_coefficient(cov)

    # Use a special case to obtain eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    # Calculating the standard deviation from the square root of the
    # variance and multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def scatter_hist(x, y, ax, ax_hist_x, ax_hist_y):
    """Make a scatter plot with histograms in the marginals

    Args:
        x: array-like, shape (n, ) input data.
        y: array-like, shape (n, ) input data.
        ax: matplotlib.axes.Axes object to draw the ellipse into.
        ax_hist_x: matplotlib.axes.Axes object to draw the ellipse into.
        ax_hist_y: matplotlib.axes.Axes object to draw the ellipse into.

    Returns:
        None
    """

    # Remove labels
    ax_hist_x.tick_params(axis="x", labelbottom=False)
    ax_hist_y.tick_params(axis="y", labelleft=False)

    # Create the scatter plot
    ax.scatter(x, y, c="black", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    # Determine nice limits by hand
    bin_width = 0.25
    xy_max = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xy_max / bin_width) + 1) * bin_width

    # Plot the marginal distributions
    bins = np.arange(-lim, lim + bin_width, bin_width)
    ax_hist_x.hist(x, bins=bins, color="black", alpha=0.5)
    ax_hist_y.hist(
        y, bins=bins, orientation="horizontal", color="black", alpha=0.5
    )


def plot_joint_distribution(x, y, title):
    """Plots a joint distribution, showing marginals and confidence ellipse

    Args:
        x: array-like, shape (n, ) input data.
        y: array-like, shape (n, ) input data.
        title: Title of the plot to be made.

    Returns:
        None
    """
    fig = plt.figure(layout="constrained")

    # Create the main axes, leaving 25% of the figure space at the top and on
    # the right to position marginals.
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()

    # The main axes' aspect can be fixed.
    ax.set(aspect=1)

    # Create marginal axes, which have 25% of the size of the main axes.  Note
    # that the inset axes are positioned *outside* (on the right and the top)
    # of the main axes, by specifying axes coordinates greater than 1.  Axes
    # coordinates less than 0 would likewise specify positions on the left and
    # the bottom of the main axes.
    ax_hist_x = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_hist_y = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

    # Draw the scatter plot and marginals.
    scatter_hist(x, y, ax, ax_hist_x, ax_hist_y)

    # Draw the confidence ellipses
    confidence_ellipse(x, y, ax, edgecolor="black", n_std=1)
    confidence_ellipse(x, y, ax, edgecolor="grey", n_std=2)
    confidence_ellipse(x, y, ax, edgecolor="lightgrey", n_std=3)

    ax.set_title(title)
