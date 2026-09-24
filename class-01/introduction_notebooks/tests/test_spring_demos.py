import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parent.parent / "spring_demos.py"
SPEC = importlib.util.spec_from_file_location("spring_demos", MODULE_PATH)
spring_demos = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(spring_demos)


def test_compute_spring_force_linear_uses_normal_spring_constant():
    model = spring_demos.SpringForceModel(2, 4, 1, 3, 1)

    result = model.compute_spring_force_linear(np.array([-2, 0, 2]))

    np.testing.assert_array_equal(result, [4, 0, -4])


def test_compute_spring_force_piecewise_handles_both_sides_and_limits():
    model = spring_demos.SpringForceModel(2, 4, 1, 3, 1)
    displacement = np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4])

    result = model.compute_spring_force_piecewise(displacement)

    np.testing.assert_array_equal(
        result, [0, 6, 2, 1, 0, -1, -2, -6, 0]
    )


def test_fit_polynomial_model_matches_linear_spring_data():
    model = spring_demos.SpringForceModel(2, 2, 1, 3, 1)
    displacement = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    model.fit_polynomial_model(displacement)

    np.testing.assert_allclose(model.coefficients, [-2, 0], atol=1e-12)
    np.testing.assert_allclose(
        model.compute_spring_force_polynomial(np.array([-0.5, 0.5])),
        [1, -1],
        atol=1e-12,
    )
