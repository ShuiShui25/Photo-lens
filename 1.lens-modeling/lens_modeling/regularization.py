"""Regularization schemes for the stage-4 pixelized source.

``Constant`` regularization penalises only first-order differences between
neighbouring source pixels.  Its matrix therefore has a near-null direction
along the global constant mode -- measured on a 4x4 mesh, the smallest
eigenvalue is 1e-8, which is just the numerical term added to keep the matrix
positive-definite.  A smooth pedestal spanning the whole source plane is
consequently almost free, which is how the outer, weakly-constrained pixels end
up absorbing image-plane flux that no other component claims.

The schemes here add a zeroth-order term, which places a diagonal cost on a
pixel's absolute value and lifts that null direction.

``al.reg.ConstantZeroth`` would do this, but is broken in autolens 2026.1.21.3:
``regularization_matrix_from`` omits the required ``neighbors_sizes`` argument,
and the underlying function builds ``eye(P)`` (max neighbours per pixel) where
``eye(S)`` (number of pixels) is meant.  Both matrices are assembled here
instead.
"""

from __future__ import annotations

import numpy as np

from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.regularization.constant import (
    constant_regularization_matrix_from,
)


class ConstantPlusZeroth(AbstractRegularization):
    """Constant gradient regularization plus a uniform zeroth-order term.

    The zeroth-order term applies equally to every pixel, so it also biases the
    genuine source low.  Prefer :class:`ConstantPlusBrightnessZeroth` when an
    adapt-image is available.
    """

    def __init__(self, coefficient: float = 1.0, coefficient_zeroth: float = 1.0):
        super().__init__()
        self.coefficient = coefficient
        self.coefficient_zeroth = coefficient_zeroth

    def regularization_weights_from(self, linear_obj, xp=np) -> np.ndarray:
        return self.coefficient * xp.ones(linear_obj.params)

    def regularization_matrix_from(self, linear_obj, xp=np) -> np.ndarray:
        matrix = constant_regularization_matrix_from(
            coefficient=self.coefficient,
            neighbors=linear_obj.neighbors,
            neighbors_sizes=linear_obj.neighbors.sizes,
            xp=xp,
        )
        return matrix + xp.eye(linear_obj.params) * self.coefficient_zeroth**2.0


class ConstantPlusBrightnessZeroth(AbstractRegularization):
    """Constant gradient regularization plus a signal-weighted zeroth-order term.

    The zeroth-order weight of each pixel is ``coefficient_zeroth * (1 -
    pixel_signal)``, following ``al.reg.BrightnessZeroth``.  ``pixel_signal`` is
    normalised to ``[0, 1]`` from the adapt-image, so bright source pixels carry
    almost no zeroth-order penalty while empty outskirts are pulled hard toward
    zero.  Unlike :class:`ConstantPlusZeroth` this leaves the reconstructed
    source flux essentially unbiased.

    ``al.reg.BrightnessZeroth`` supplies *only* the diagonal term -- on its own
    it would leave the source with no smoothing at all -- so the gradient term
    is retained here alongside it.

    Requires an adapt-image for the source galaxy; see
    ``pipeline.build_adapt_images_from_stage3``.
    """

    def __init__(
        self,
        coefficient: float = 1.0,
        coefficient_zeroth: float = 1.0,
        signal_scale: float = 1.0,
    ):
        super().__init__()
        self.coefficient = coefficient
        self.coefficient_zeroth = coefficient_zeroth
        self.signal_scale = signal_scale

    def zeroth_weights_from(self, linear_obj, xp=np) -> np.ndarray:
        # The mapper may return a small number of negative adapt signals even
        # when the image-plane adapt image is non-negative (e.g. six pixels,
        # minimum -0.083 in a representative fit).  Fractional signal scales would
        # turn those values into NaNs before the regularization matrix is
        # assembled.  Brightness is defined on [0, 1], so enforce that physical
        # domain before exponentiation instead of clipping the final weights.
        pixel_signals = linear_obj.pixel_signals_from(signal_scale=1.0, xp=xp)
        pixel_signals = xp.clip(pixel_signals, 0.0, 1.0) ** self.signal_scale
        return self.coefficient_zeroth * (1.0 - pixel_signals)

    def regularization_weights_from(self, linear_obj, xp=np) -> np.ndarray:
        return self.coefficient * xp.ones(linear_obj.params)

    def regularization_matrix_from(self, linear_obj, xp=np) -> np.ndarray:
        matrix = constant_regularization_matrix_from(
            coefficient=self.coefficient,
            neighbors=linear_obj.neighbors,
            neighbors_sizes=linear_obj.neighbors.sizes,
            xp=xp,
        )
        zeroth = self.zeroth_weights_from(linear_obj=linear_obj, xp=xp) ** 2.0
        return matrix + xp.diag(zeroth)
