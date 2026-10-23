from __future__ import annotations

import warnings

import numpy as np
from astropy.stats import biweight_location, sigma_clip
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    EllipticalAperture,
    RectangularAperture,
)
from scipy.stats import skew


def region_to_aperture(region_pix):
    """Convert region classes supported directly by photutils."""
    name = region_pix.__class__.__name__
    try:
        if name == "CirclePixelRegion":
            return CircularAperture(
                (float(region_pix.center.x), float(region_pix.center.y)),
                r=float(region_pix.radius),
            )
        if name == "EllipsePixelRegion":
            return EllipticalAperture(
                (float(region_pix.center.x), float(region_pix.center.y)),
                a=float(region_pix.width / 2.0),
                b=float(region_pix.height / 2.0),
                theta=float(region_pix.angle.value),
            )
        if name == "RectanglePixelRegion":
            return RectangularAperture(
                (float(region_pix.center.x), float(region_pix.center.y)),
                w=float(region_pix.width),
                h=float(region_pix.height),
                theta=float(region_pix.angle.value),
            )
    except Exception:
        return None
    return None


def region_weight_image(
    region_pix,
    image_shape: tuple[int, int],
    subpixels: int,
    mode: str = "subpixels",
) -> np.ndarray | None:
    """Create center or fractional weights without a silent polygon fallback."""
    if mode not in {"center", "subpixels"}:
        raise ValueError(f"Unsupported mask mode: {mode}")

    aperture = region_to_aperture(region_pix)
    if aperture is not None:
        method = "center" if mode == "center" else "subpixel"
        mask = aperture.to_mask(method=method, subpixels=int(subpixels))
    elif mode == "center":
        mask = region_pix.to_mask(mode="center")
    else:
        try:
            mask = region_pix.to_mask(mode="exact")
        except NotImplementedError:
            mask = region_pix.to_mask(
                mode="subpixels", subpixels=int(subpixels)
            )

    weights = mask.to_image(image_shape) if mask is not None else None
    if weights is None:
        return None
    weights = np.asarray(weights, dtype=float)
    weights[~np.isfinite(weights)] = 0.0

    if mode == "subpixels" and "Polygon" in region_pix.__class__.__name__:
        # Photo-lens writes source/background as exact pixel-cell outlines.
        # On their native grid every vertex sits on a pixel edge (half-integer
        # coordinate) and exactly binary weights are the correct answer.
        vertices = np.column_stack((region_pix.vertices.x, region_pix.vertices.y))
        offset = np.abs(((vertices + 0.5) + 0.5) % 1.0 - 0.5)
        pixel_aligned = bool(np.all(offset < 1.0e-4))
        if not pixel_aligned and not np.any((weights > 0.0) & (weights < 1.0)):
            raise AssertionError(
                "Polygon weights are binary; subpixel mask did not take effect"
            )
    return weights


def robust_background_level(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    mean = float(np.mean(values))
    absolute_skew = abs(float(skew(values))) if values.size > 2 else 0.0
    if absolute_skew < 0.3:
        return median
    if absolute_skew <= 1.0:
        return 3.0 * median - 2.0 * mean
    return float(biweight_location(values))


def annulus_measurement(
    image: np.ndarray,
    lens_xy: tuple[float, float],
    r_in: float = 30.0,
    r_out: float = 55.0,
) -> dict:
    annulus = CircularAnnulus(lens_xy, r_in=r_in, r_out=r_out)
    mask_image = annulus.to_mask(method="center").to_image(image.shape)
    if mask_image is None:
        raise ValueError("Background annulus lies outside image")
    mask = np.asarray(mask_image, dtype=bool)
    work = np.full(image.shape, np.nan)
    work[mask] = image[mask]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        clipped = sigma_clip(work, sigma=3, maxiters=2)
    values = clipped.data[~clipped.mask]
    values = values[np.isfinite(values)]
    if values.size:
        values = np.delete(values, np.argmax(values))
    if values.size < 2:
        raise ValueError("Too few usable background-annulus pixels")
    return {
        "mask": mask,
        "area": float(annulus.area),
        "level": robust_background_level(values),
        "pixel_sigma": float(np.std(values, ddof=1)),
        "samples": int(values.size),
    }
