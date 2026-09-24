"""Provides simple tools for demonstrating linear and nonlinear functions"""

import numpy as np


def plot_dx(
    plot_axis,
    x_array,
    f_function,
    x0_start_value,
    dx_change,
    color_string="b",
    alpha_value="0.2",
):
    """Plots how a change in an input propagates through a function.

    This function adds polygons to a plot axis to show how a change, dx_change,
    in an input value, x0_start_value, propagates through a function,
    f_function.

    Args:
        plot_axis: the axis of the plot to add the polygons to.
        x_array: array-like, shape (n,), the array of x values of the function.
        f_function: callable function to be analyzed.
        x0_start_value: the start value of the change to be analyzed.
        dx_change: the change in the input value to be analyzed.
        color_string: the color of the polygons.
        alpha_value: the alpha (transparency) value of the polygons.

    Returns:
        None
    """
    x1 = x0_start_value + dx_change
    y_lower_edge = f_function(x_array)
    y_lower_edge[x_array <= x0_start_value] = f_function(x0_start_value)
    plot_axis.fill_between(
        x_array,
        y_lower_edge,
        f_function(x1),
        where=x_array < x1,
        facecolor=color_string,
        alpha=alpha_value,
    )
    plot_axis.fill_between(
        x_array,
        np.min(f_function(x_array)),
        f_function(x_array),
        where=(x_array >= x0_start_value) & (x_array <= x1),
        facecolor=color_string,
        alpha=alpha_value,
    )


def plot_dx_list(
    plot_axis,
    x_array,
    f_function,
    x0_start_values,
    dx_change_values,
    color_strings,
    alpha_value=0.2,
):
    """Plots how a list of changes in input propagate through a function.

    Each change in input is represented as shaded polygons. The polygons are
    added to the given plot axis to show how a list of changes, the values of
    dx_change_values_list, in input values, the values of x0_start_values_list
    affect the outputs of a function, f_function.

    Args:
        plot_axis: the axis of the plot to add the polygons to.
        x_array: array-like, shape (n,), the array of x values of the function.
        f_function: callable function to be analyzed.
        x0_start_values: list of start value of the changes to be analyzed.
        dx_change_values: list of changes to be analyzed.
        color_strings: list of colors for each group of polygons.
        alpha_value: the alpha (transparency) value of the polygons.

    Returns:
        None
    """
    for x0, dx, color in zip(x0_start_values, dx_change_values, color_strings):
        plot_dx(plot_axis, x_array, f_function, x0, dx, color, alpha_value)
