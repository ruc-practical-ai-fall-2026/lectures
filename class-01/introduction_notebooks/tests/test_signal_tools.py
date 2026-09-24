import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parent.parent / "signal_tools.py"
SPEC = importlib.util.spec_from_file_location("signal_tools", MODULE_PATH)
signal_tools = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(signal_tools)


def test_add_white_gaussian_noise_is_reproducible_with_seeded_generator():
    signal = np.ones(4)

    result = signal_tools.add_white_gaussian_noise(
        signal, snr_db=0, rng=np.random.default_rng(42)
    )
    expected_noise = np.random.default_rng(42).normal(0, 1, signal.shape)

    np.testing.assert_allclose(result, signal + expected_noise)


def test_add_white_gaussian_noise_preserves_shape_and_zero_signal():
    signal = np.zeros((2, 3))

    result = signal_tools.add_white_gaussian_noise(
        signal, snr_db=10, rng=np.random.default_rng(42)
    )

    assert result.shape == signal.shape
    np.testing.assert_array_equal(result, signal)


def test_generate_tone_signal_is_zero_outside_pulse():
    time = np.arange(0, 1, 0.25)

    result = signal_tools.generate_tone_signal(time, 0.25, 0.75, 1)

    np.testing.assert_allclose(result, [0, 1, 0, 0], atol=1e-6)
