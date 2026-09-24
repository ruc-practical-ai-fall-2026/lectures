import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parent.parent / "linear_algebra_demos.py"
SPEC = importlib.util.spec_from_file_location(
    "linear_algebra_demos", MODULE_PATH
)
linear_algebra_demos = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(linear_algebra_demos)


def test_compute_norm_returns_euclidean_norm():
    assert linear_algebra_demos.compute_norm(3, 4) == 5


def test_raise_to_zero_power_preserves_nonzero_values_and_counts_zeros():
    values = np.array([0, 2, -3])

    result = linear_algebra_demos.raise_to_zero_power(values)

    np.testing.assert_array_equal(result, [0, 1, 1])
    assert result is values


def test_compute_lp_norm_for_p_greater_than_one():
    result = linear_algebra_demos.compute_lp_norm(3, 4, 2)

    assert result == 5


def test_compute_lp_norm_for_p_between_zero_and_one():
    result = linear_algebra_demos.compute_lp_norm(3, 4, 0.5)

    assert result == pytest.approx(np.sqrt(3) + 2)


def test_compute_lp_norm_for_zero_counts_nonzero_components():
    result = linear_algebra_demos.compute_lp_norm(
        np.array([0, 3]), np.array([4, 0]), 0
    )

    np.testing.assert_array_equal(result, [1, 1])


def test_lp_norm_demo_creates_expected_surface():
    demo = linear_algebra_demos.LPNormDemo(
        x_range=(-1, 1), y_range=(-1, 1), grid_size=3, n_init=2
    )

    demo._create_surface()

    assert demo.x_component.shape == (3, 3)
    assert demo.y_component.shape == (3, 3)
    np.testing.assert_array_equal(
        demo.p_norm,
        np.array(
            [
                [np.sqrt(2), 1, np.sqrt(2)],
                [1, 0, 1],
                [np.sqrt(2), 1, np.sqrt(2)],
            ]
        ),
    )
