"""Trace-safe complete imaging objective for pixelized-source JAX fits."""

from __future__ import annotations

import math

import autofit as af
import autolens as al


LIKELIHOOD_SCHEMA = "imaging_figure_of_merit_v1"
LIKELIHOOD_TAG = "fomv1"
LIKELIHOOD_GUARD_TAG = "lg1"
INVALID_FOM_SENTINEL = -1.0e30


def imaging_figure_of_merit_ceiling(noise_normalization):
    """Zero-residual upper bound for a positive-semidefinite inversion."""
    return -0.5 * noise_normalization


def guarded_imaging_figure_of_merit(fit, *, use_jax: bool):
    value = fit.figure_of_merit
    ceiling = imaging_figure_of_merit_ceiling(fit.noise_normalization)
    if use_jax:
        import jax.numpy as jnp

        value = jnp.asarray(value)
        ceiling = jnp.asarray(ceiling)
        tolerance = jnp.maximum(
            50.0, 1.0e-8 * jnp.maximum(jnp.abs(ceiling), 1.0)
        )
        valid = (
            jnp.isfinite(value)
            & jnp.isfinite(ceiling)
            & (value <= ceiling + tolerance)
        )
        return jnp.where(
            valid, value, jnp.asarray(INVALID_FOM_SENTINEL, dtype=value.dtype)
        )

    value = float(value)
    ceiling = float(ceiling)
    tolerance = max(50.0, 1.0e-8 * max(abs(ceiling), 1.0))
    if not math.isfinite(value) or not math.isfinite(ceiling):
        raise af.exc.FitException
    if value > ceiling + tolerance:
        raise af.exc.FitException
    return value


class AnalysisImagingFigureOfMerit(al.AnalysisImaging):
    """Always return the full inversion evidence, with a JAX-safe guard."""

    likelihood_schema = LIKELIHOOD_SCHEMA

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        penalty = self.log_likelihood_penalty_from(instance=instance)
        try:
            fit = self.fit_from(instance=instance)
            return guarded_imaging_figure_of_merit(
                fit, use_jax=bool(self._use_jax)
            ) - penalty
        except af.exc.FitException:
            raise
        except Exception:
            raise af.exc.FitException
