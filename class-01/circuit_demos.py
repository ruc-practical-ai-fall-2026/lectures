import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


class AmplifierDemo:
    """Demonstrate a nonlinear amplifier."""

    def __init__(
        self,
        amplitude_init=1,
        amplitude_min=0.1,
        amplitude_max=5,
    ):
        self.amplitude_init = amplitude_init
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.y_limit = 6

    def instantiate_plot(self, time_seconds, input_signal):
        """Creates a plot to demonstrate amplifier distortion."""
        self.time_ms = time_seconds * 1000
        self.input_signal = input_signal
        self._compute_amplifier_outputs()
        self._create_plot()
        self._configure_plot_axes()
        self._plot_amplifier_outputs()
        self._build_sliders()
        self.slider.on_changed(self._update)
        self.ax.legend(["Nonlinear", "Linear"], loc="upper right")
        plt.show()

    def _build_sliders(self):
        self.ax_slider = plt.axes(
            [0.1, 0.05, 0.8, 0.05], facecolor="lightgray"
        )
        self.slider = Slider(
            self.ax_slider,
            "A",
            self.amplitude_min,
            self.amplitude_max,
            valinit=self.amplitude_init,
            valstep=0.1,
        )

    def _compute_amplifier_outputs(self):
        self.nonlinear_amplifier_output = (
            self.compute_nonlinear_amplifier_output(self.amplitude_init)
        )
        self.linear_amplifier_output = self.compute_linear_amplifier_output(
            self.amplitude_init
        )

    def _configure_plot_axes(self):
        self.ax.set_xlabel(r"$t$ (ms)", usetex=True)
        self.ax.set_ylabel(r"$y(t)$ (V)", usetex=True)
        self.ax.set_title(r"Amplifier Outputs", usetex=True)
        self.ax.set_ylim([-self.y_limit, self.y_limit])
        xtick_labels = self.ax.get_xticklabels()
        ytick_labels = self.ax.get_yticklabels()
        tick_labels = xtick_labels + ytick_labels
        for label in tick_labels:
            label.set_fontsize(12)
            label.set_color("black")
            label.usetex = True

    def _plot_amplifier_outputs(self):
        (self.nonlinear_amplifier_line_plot,) = self.ax.plot(
            self.time_ms,
            self.nonlinear_amplifier_output,
            color="black",
            linestyle="-",
        )
        (self.linear_amplifier_line_plot,) = self.ax.plot(
            self.time_ms,
            self.linear_amplifier_output,
            color="black",
            linestyle="--",
        )

    def _create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(left=0.1, bottom=0.25)

    def compute_nonlinear_amplifier_output(self, amplitude):
        """Compute nonlinear amplifier output"""
        return np.tanh(amplitude * self.input_signal)

    def compute_linear_amplifier_output(self, amplitude):
        """Compute linear amplifier output"""
        return amplitude * self.input_signal

    def _update(self, val):
        """Update the plot based on the slider value."""
        amplitude = self.slider.val
        self.nonlinear_amplifier_output = (
            self.compute_nonlinear_amplifier_output(amplitude)
        )
        self.linear_amplifier_output = self.compute_linear_amplifier_output(
            amplitude
        )
        self.nonlinear_amplifier_line_plot.set_ydata(
            self.nonlinear_amplifier_output
        )
        self.linear_amplifier_line_plot.set_ydata(self.linear_amplifier_output)
        self.ax.legend(["Nonlinear", "Linear"], loc="upper right")
        self.fig.canvas.draw_idle()


class ADCModel:
    """Model an analog to digital converter."""

    def __init__(self, bit_depth=3):
        self.bit_depth = bit_depth
        self.quantization_levels = 2**self.bit_depth

    def quantize_signal(self, time_seconds, input_signal):
        """Quantize the signal."""
        self.time_seconds = time_seconds
        self.analog_signal = input_signal
        signal_minimum = np.min(input_signal)
        input_signal_no_offset = input_signal - signal_minimum
        scaling_factor = (self.quantization_levels - 1) / np.max(
            input_signal_no_offset
        )
        input_signal_no_offset_scaled = input_signal_no_offset * scaling_factor
        input_signal_digitized = np.round(input_signal_no_offset_scaled)
        self.digital_signal = (
            input_signal_digitized / scaling_factor + signal_minimum
        )

    def plot_signals(self):
        """Plot the input analog signal and the output digitized signal."""
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.time_seconds,
            self.analog_signal,
            label="Input Signal",
            linestyle="-",
            color="black",
        )
        plt.step(
            self.time_seconds,
            self.digital_signal,
            label="Quantized Signal",
            linestyle="--",
            color="black",
            where="mid",
        )

        plt.xlabel(r"$t$ (s)", usetex=True)
        plt.ylabel(r"$x$ (V)", usetex=True)
        plt.title(f"ADC with Bit Depth = {self.bit_depth}")
        plt.legend()
        plt.show()
