"""Defines educational tools for working with signals."""

import numpy as np


def add_white_gaussian_noise(
    signal_array: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add white Gaussian noise to the signal array.

    This function is similar to MATLAB's add white Gaussian noise function.

    Args:
        signal_array: The signal for which noise is to be added.
        snr_db: The desired signal to noise ratio in 10*log10 dB.
        rng: The random number generator to use.

    Returns:
        The noisy signal as an array of the same shape as the original.
    """
    snr = 10 ** (snr_db / 10)
    signal_power = np.average(np.abs(signal_array) ** 2)
    noise_power = signal_power / snr
    mu_noise_mean = 0
    sigma_noise_std_dev = np.sqrt(noise_power)
    noise = rng.normal(mu_noise_mean, sigma_noise_std_dev, signal_array.shape)
    signal_with_noise = signal_array + noise
    return signal_with_noise


def generate_tone_signal(
    time_seconds,
    pulse_start_seconds,
    pulse_end_seconds,
    frequency1_hz,
):
    envelope = np.heaviside(
        time_seconds - pulse_start_seconds, 1
    ) - np.heaviside(time_seconds - pulse_end_seconds, 1)
    tone_signal = envelope * np.sin(2 * np.pi * frequency1_hz * time_seconds)
    return tone_signal
