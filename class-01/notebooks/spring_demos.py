"""Demonstrates linear and nonlinear spring models for student examples."""

import numpy as np


class SpringForceModel:
    """Models a spring force using piecewise and polynomial models."""

    def __init__(
        self,
        spring_constant_normal_regime_k1,
        spring_constant_unrecoverable_regime_k2,
        recoverable_limit_x1,
        breakdown_limit_x2,
        n_coefficients,
    ):
        """Initializes the spring force model"""
        self.n_coefficients = n_coefficients
        self.spring_constant_normal_regime_k1 = (
            spring_constant_normal_regime_k1
        )
        self.spring_constant_unrecoverable_regime_k2 = (
            spring_constant_unrecoverable_regime_k2
        )
        self.recoverable_limit_x1 = recoverable_limit_x1
        self.breakdown_limit_x2 = breakdown_limit_x2
        self.coefficients = None

    def fit_polynomial_model(self, displacement_x):
        self._spring_force_piecewise = self.compute_spring_force_piecewise(
            displacement_x
        )
        self.coefficients = np.polyfit(
            displacement_x,
            self._spring_force_piecewise,
            deg=self.n_coefficients,
        )

    def compute_spring_force_linear(self, displacement_x):
        """Computes a simple linear model of a spring force."""
        spring_force_f = (
            -self.spring_constant_normal_regime_k1 * displacement_x
        )
        return spring_force_f

    def compute_spring_force_piecewise(self, displacement_x):
        """Computes a piecewise model of a spring force."""
        self._extract_piecewise_masks(displacement_x)
        self._compute_piecewise_boundary_values()
        spring_force_f = self._compute_boundary_adjusted_piecewise_force(
            displacement_x
        )
        return spring_force_f

    def _compute_boundary_adjusted_piecewise_force(self, displacement_x):
        spring_force_f = np.zeros(displacement_x.shape)
        spring_force_f[self._linear_regime_mask] = (
            -self.spring_constant_normal_regime_k1
            * displacement_x[self._linear_regime_mask]
        )
        spring_force_f[self._unrecoverable_regime_mask] = (
            -self.spring_constant_unrecoverable_regime_k2
            * displacement_x[self._unrecoverable_regime_mask]
        )
        spring_force_f[self._breakdown_regime_mask] = 0

        spring_force_f[
            self._non_positive_mask & self._unrecoverable_regime_mask
        ] += self._unrecoverable_region_adjustment_offset
        spring_force_f[
            ~self._non_positive_mask & self._unrecoverable_regime_mask
        ] -= self._unrecoverable_region_adjustment_offset

        return spring_force_f

    def _compute_piecewise_boundary_values(self):
        self._normal_regime_boundary_value_k1 = (
            -self.spring_constant_normal_regime_k1 * self.recoverable_limit_x1
        )
        self._normal_regime_boundary_value_k2 = (
            -self.spring_constant_unrecoverable_regime_k2
            * self.recoverable_limit_x1
        )
        self._unrecoverable_region_adjustment_offset = (
            self._normal_regime_boundary_value_k2
            - self._normal_regime_boundary_value_k1
        )

    def _extract_piecewise_masks(self, displacement_x):
        self._non_positive_mask = displacement_x <= 0
        self._linear_regime_mask = (
            np.abs(displacement_x) < self.recoverable_limit_x1
        )
        self._unrecoverable_regime_mask = (
            np.abs(displacement_x) >= self.recoverable_limit_x1
        ) & (np.abs(displacement_x) < self.breakdown_limit_x2)
        self._breakdown_regime_mask = (
            np.abs(displacement_x) > self.breakdown_limit_x2
        )

    def compute_spring_force_polynomial(self, displacement_x):
        """Computes polynomial model of a spring force."""
        spring_force_f = np.polyval(self.coefficients, displacement_x)
        return spring_force_f
