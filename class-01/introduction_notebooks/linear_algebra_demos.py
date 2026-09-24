import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


def compute_norm(x_component, y_component):
    """Compute the L2 norm of a 2-element vector."""
    return np.sqrt(x_component**2 + y_component**2)


class L2NormDemo:
    """Demonstrate computing the L2 norm of a 2-element vector."""

    def __init__(self):
        self._label_x_location = 7
        self._label_y_location = 9.5
        self._initial_x = 3.0
        self._initial_y = 4.0
        self._plot_max = 12
        self._plot_min = -12
        self._ORIGIN_X = 0
        self._ORIGIN_Y = 0

    def instantiate_plot(self):
        """Creates a plot to demonstrate the L2 norm."""
        self._create_plot()
        self._configure_plot_axes()
        self._plot_initial_vector()
        self._build_sliders()
        self._slider_x.on_changed(self._update)
        self._slider_y.on_changed(self._update)
        plt.show()

    def _build_display_text(self, norm):
        display_text_string = f"$\\|\\boldsymbol{{x}}\\|$ = {norm:.2f}"
        return display_text_string

    def _create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(left=0.1, bottom=0.3)

    def _configure_plot_axes(self):
        self.ax.set_xlim([self._plot_min, self._plot_max])
        self.ax.set_ylim([self._plot_min, self._plot_max])
        self.ax.axhline(self._ORIGIN_X, color="black", lw=2)
        self.ax.axvline(self._ORIGIN_Y, color="black", lw=2)
        self.ax.set_title("$L^2$ Norm Demonstration", usetex=True)
        self.ax.set_xlabel("$x_1$", usetex=True)
        self.ax.set_ylabel("$x_2$", usetex=True)
        xtick_labels = self.ax.get_xticklabels()
        ytick_labels = self.ax.get_yticklabels()
        tick_labels = xtick_labels + ytick_labels
        for label in tick_labels:
            label.set_fontsize(12)
            label.set_color("black")
            label.usetex = True

    def _plot_initial_vector(self):
        self.ax.arrow(
            self._ORIGIN_X,
            self._ORIGIN_Y,
            self._initial_x,
            self._initial_y,
            head_width=0.5,
            head_length=0.7,
            fc="black",
            ec="black",
        )
        norm = compute_norm(self._initial_x, self._initial_y)
        self._vector_length_text = self.ax.text(
            self._label_x_location,
            self._label_y_location,
            self._build_display_text(norm),
            fontsize=9,
            color="black",
            bbox=dict(facecolor="grey", edgecolor="black", pad=8.0),
        )

    def _build_sliders(self):
        x_slider_position = [0.1, 0.15, 0.8, 0.05]
        y_slider_position = [0.1, 0.05, 0.8, 0.05]
        ax_x = plt.axes(x_slider_position, facecolor="lightgoldenrodyellow")
        ax_y = plt.axes(y_slider_position, facecolor="lightgoldenrodyellow")
        self._slider_x = Slider(
            ax_x, "$x_1$", -10.0, 10.0, valinit=self._initial_x
        )
        self._slider_y = Slider(
            ax_y, "$x_2$", -10.0, 10.0, valinit=self._initial_y
        )

    def _update(self, val):
        new_x = self._slider_x.val
        new_y = self._slider_y.val
        [p.remove() for p in reversed(self.ax.patches)]
        self.ax.arrow(
            0,
            0,
            new_x,
            new_y,
            head_width=0.5,
            head_length=0.7,
            fc="black",
            ec="black",
        )
        norm = compute_norm(new_x, new_y)
        display_text = self._build_display_text(norm)
        self._vector_length_text.set_text(display_text)
        self.fig.canvas.draw_idle()


def raise_to_zero_power(x_variable):
    """Raise to the power of 0, defining 0**0 = 0"""
    x_equals_0_mask = x_variable == 0
    x_variable[~x_equals_0_mask] = 1
    return x_variable


def compute_lp_norm(x_component, y_component, p_parameter):
    """Compute the LP norm of a 2-element vector."""
    if p_parameter >= 1:
        lp_norm = (
            np.abs(x_component) ** p_parameter
            + np.abs(y_component) ** p_parameter
        ) ** (1 / p_parameter)
    elif p_parameter < 1 and p_parameter > 0:
        lp_norm = (
            np.abs(x_component) ** p_parameter
            + np.abs(y_component) ** p_parameter
        )
    elif p_parameter == 0:
        abs_x_raised_to_zero_power = raise_to_zero_power(np.abs(x_component))
        abs_y_raised_to_zero_power = raise_to_zero_power(np.abs(y_component))
        lp_norm = abs_x_raised_to_zero_power + abs_y_raised_to_zero_power

    return lp_norm


class LPNormDemo:
    """Demonstrate computing the LP norm of a 2-element vector."""

    def __init__(
        self,
        x_range=(-10, 10),
        y_range=(-10, 10),
        grid_size=25,
        n_init=2,
        n_min=0,
        n_max=10,
        cmap="viridis",
        contour_levels=10,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.grid_size = grid_size
        self.n_init = n_init
        self.n_min = n_min
        self.n_max = n_max
        self.cmap_name = cmap
        self.contour_levels = contour_levels

    def instantiate_plot(self):
        """Creates a plot to demonstrate the LP norm."""
        self._create_surface()
        self._create_plot()
        self._plot_initial_surface()
        self._create_colorbar()
        self._configure_plot_axes()
        self._build_slider()
        self.slider.on_changed(self._update)
        plt.show()

    def _build_slider(self):
        self.ax_slider = plt.axes(
            [0.1, 0.05, 0.8, 0.05], facecolor="lightgray"
        )
        self.slider = Slider(
            self.ax_slider,
            r"$P$",
            self.n_min,
            self.n_max,
            valinit=self.n_init,
            valstep=0.1,
        )

    def _create_colorbar(self):
        self.cbar = plt.colorbar(self.cmap, ax=self.ax)
        self.cbar.set_label(r"$L^P$ Norm", usetex=True)

    def _plot_initial_surface(self):
        self.cmap = self.ax.imshow(
            self.p_norm,
            extent=[*self.x_range, *self.y_range],
            origin="lower",
            cmap=self.cmap_name,
        )
        self.contours = self.ax.contour(
            self.x_component,
            self.y_component,
            self.p_norm,
            levels=self.contour_levels,
            colors="black",
            linewidths=1,
        )
        self.ax.clabel(self.contours, inline=True, fontsize=8, fmt="%1.1f")

    def _configure_plot_axes(self):
        self.ax.set_xlabel(r"$x_1$", usetex=True)
        self.ax.set_ylabel(r"$x_2$", usetex=True)
        self.ax.set_title(r"$L^P$ Norm Demonstration", usetex=True)
        xtick_labels = self.ax.get_xticklabels()
        ytick_labels = self.ax.get_yticklabels()
        cbar_labels = self.cbar.ax.get_yticklabels()
        tick_labels = xtick_labels + ytick_labels + cbar_labels
        for label in tick_labels:
            label.set_fontsize(12)
            label.set_color("black")
            label.usetex = True

    def _create_plot(self):
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.25)

    def _create_surface(self):
        self.x_component, self.y_component = np.meshgrid(
            np.linspace(*self.x_range, self.grid_size),
            np.linspace(*self.y_range, self.grid_size),
        )
        self.p_norm = compute_lp_norm(
            self.x_component, self.y_component, self.n_init
        )

    def _update(self, val):
        p_parameter = self.slider.val
        self.p_norm = compute_lp_norm(
            self.x_component, self.y_component, p_parameter
        )
        self.cmap.set_data(self.p_norm)
        self.cbar.update_normal(self.cmap)
        self.contours.remove()
        self.contours = self.ax.contour(
            self.x_component,
            self.y_component,
            self.p_norm,
            levels=self.contour_levels,
            colors="black",
            linewidths=1,
        )
        self.ax.clabel(self.contours, inline=True, fontsize=8, fmt="%1.1f")
        self.fig.canvas.draw_idle()
