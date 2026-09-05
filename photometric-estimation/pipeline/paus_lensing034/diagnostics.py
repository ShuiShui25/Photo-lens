from __future__ import annotations

import warnings

import numpy as np
from astropy.stats import mad_std, sigma_clip
from scipy.ndimage import binary_dilation, label as ndi_label
from scipy.signal import fftconvolve


def build_bright_source_mask(
    image: np.ndarray,
    nsigma: float,
    min_pixels: int = 5,
    dilate: int = 2,
) -> np.ndarray:
    finite = np.isfinite(image)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        clipped = sigma_clip(image[finite], sigma=3, maxiters=5)
    sky = clipped.data[~clipped.mask]
    level = float(np.median(sky))
    sigma = float(np.std(sky, ddof=1))
    detected = finite & (image > level + float(nsigma) * sigma)
    labels, count = ndi_label(detected)
    source_mask = np.zeros_like(detected, dtype=bool)
    if count:
        sizes = np.bincount(labels.ravel())
        keep = np.flatnonzero(sizes >= int(min_pixels))
        keep = keep[keep > 0]
        if keep.size:
            source_mask = np.isin(labels, keep)
    if dilate:
        source_mask = binary_dilation(source_mask, iterations=int(dilate))
    return source_mask | ~finite


def _crop_kernel(kernel: np.ndarray):
    ys, xs = np.nonzero(np.abs(kernel) > 0)
    if ys.size == 0:
        return None
    return (
        kernel[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
        int(ys.min()),
        int(xs.min()),
    )


def _kernel_map(image: np.ndarray, full_kernel: np.ndarray):
    cropped = _crop_kernel(full_kernel)
    if cropped is None:
        return None
    kernel, y_min, x_min = cropped
    if kernel.shape[0] >= image.shape[0] or kernel.shape[1] >= image.shape[1]:
        return None
    finite = np.isfinite(image)
    level = float(np.median(image[finite]))
    values = fftconvolve(
        np.where(finite, image - level, 0.0),
        kernel[::-1, ::-1],
        mode="valid",
    )
    footprint = (np.abs(kernel) > 0).astype(float)
    return values, footprint, y_min, x_min


def _sample_kernel_map(
    prepared,
    blocked: np.ndarray,
    lens_xy: tuple[float, float],
    local_max_radius_pix: float,
) -> dict | None:
    if prepared is None:
        return None
    values, footprint, y_min, x_min = prepared
    overlap = fftconvolve(
        blocked.astype(float), footprint[::-1, ::-1], mode="valid"
    )
    good = overlap < 0.5
    sample = values[good]
    sample = sample[np.isfinite(sample)]
    if sample.size < 2:
        return None

    oy, ox = np.mgrid[:values.shape[0], :values.shape[1]]
    translated_x = ox + (lens_xy[0] - x_min)
    translated_y = oy + (lens_xy[1] - y_min)
    radius = np.hypot(translated_x - lens_xy[0], translated_y - lens_xy[1])
    local_sample = values[good & (radius <= local_max_radius_pix)]
    local_sample = local_sample[np.isfinite(local_sample)]
    return {
        "std": float(np.std(sample, ddof=1)),
        "mad": float(mad_std(sample, ignore_nan=True)),
        "n": int(sample.size),
        "local_std": (
            float(np.std(local_sample, ddof=1)) if local_sample.size > 1 else np.nan
        ),
        "local_mad": (
            float(mad_std(local_sample, ignore_nan=True))
            if local_sample.size > 1 else np.nan
        ),
        "local_n": int(local_sample.size),
        "kernel_pixels": int(footprint.sum()),
        "bbox_y": int(footprint.shape[0]),
        "bbox_x": int(footprint.shape[1]),
    }


def differential_noise_diagnostics(
    image: np.ndarray,
    source_weights: np.ndarray,
    background_weights: np.ndarray,
    lens_xy: tuple[float, float],
    pixel_scale_arcsec: float,
) -> dict:
    """Measure blank-kernel scatter for diagnostics only."""
    finite = np.isfinite(image)
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    exclude_pix = 10.0 / max(pixel_scale_arcsec, 1e-9)
    local_pix = 18.0 / max(pixel_scale_arcsec, 1e-9)
    near_lens = np.hypot(xx - lens_xy[0], yy - lens_xy[1]) < exclude_pix

    kernel = source_weights - background_weights
    prepared = _kernel_map(image, kernel)
    output = {}
    samples = {}
    for nsigma in (2, 3, 5):
        source_mask = build_bright_source_mask(image, nsigma=nsigma)
        sample = _sample_kernel_map(
            prepared, source_mask | near_lens | ~finite, lens_xy, local_pix
        )
        samples[nsigma] = sample
        for metric in ("std", "mad"):
            output[f"sigma_{metric}_n{nsigma}"] = (
                sample[metric] if sample is not None else np.nan
            )

    adopted = samples[3]
    if adopted is None:
        output.update({
            "sigma_local18": np.nan,
            "sigma_full": np.nan,
            "n_valid": 0,
            "n_effective_approx": 0.0,
            "kernel_pixels": 0,
            "kernel_bbox_y": 0,
            "kernel_bbox_x": 0,
        })
    else:
        output.update({
            "sigma_local18": adopted["local_std"],
            "sigma_full": adopted["std"],
            "n_valid": adopted["n"],
            "n_effective_approx": adopted["n"] / max(adopted["kernel_pixels"], 1),
            "kernel_pixels": adopted["kernel_pixels"],
            "kernel_bbox_y": adopted["bbox_y"],
            "kernel_bbox_x": adopted["bbox_x"],
        })

    source_mask3 = build_bright_source_mask(image, nsigma=3)
    blocked3 = source_mask3 | near_lens | ~finite
    source_sample = _sample_kernel_map(
        _kernel_map(image, source_weights), blocked3, lens_xy, local_pix
    )
    background_sample = _sample_kernel_map(
        _kernel_map(image, background_weights), blocked3, lens_xy, local_pix
    )
    denominator = (
        np.hypot(source_sample["std"], background_sample["std"])
        if source_sample is not None and background_sample is not None else np.nan
    )
    output["diff_over_independent_empirical"] = (
        adopted["std"] / denominator
        if adopted is not None and np.isfinite(denominator) and denominator > 0
        else np.nan
    )

    source_area = float(source_weights.sum())
    background_area = float(background_weights.sum())
    zero_sum_kernel = (
        source_weights
        - (source_area / max(background_area, 1e-12)) * background_weights
    )
    zero_sample = _sample_kernel_map(
        _kernel_map(image, zero_sum_kernel), blocked3, lens_xy, local_pix
    )
    output["sigma_zero_sum"] = (
        zero_sample["std"] if zero_sample is not None else np.nan
    )
    output["sigma_raw"] = adopted["std"] if adopted is not None else np.nan
    output["unmasked_std"] = (
        _sample_kernel_map(
            prepared, near_lens | ~finite, lens_xy, local_pix
        ) or {"std": np.nan}
    )["std"]
    return output


def validate_empirical_kernel_math() -> dict:
    """Cheap deterministic white-noise and correlated-noise smoke tests."""
    rng = np.random.default_rng(3401)
    kernel = np.zeros((15, 17), dtype=float)
    kernel[2:6, 2:7] = 1.0
    kernel[9:13, 10:15] = -1.0
    expected = float(np.sqrt(np.sum(kernel ** 2)))

    white_ratios = []
    corr_ratios = []
    for _ in range(12):
        white = rng.normal(size=(160, 160))
        white_values = fftconvolve(white, kernel[::-1, ::-1], mode="valid")
        white_ratios.append(float(np.std(white_values, ddof=1) / expected))

        smooth = fftconvolve(white, np.ones((3, 3)) / 3.0, mode="same")
        corr_values = fftconvolve(smooth, kernel[::-1, ::-1], mode="valid")
        naive = float(np.std(smooth, ddof=1) * expected)
        corr_ratios.append(float(np.std(corr_values, ddof=1) / naive))

    white_ratio = float(np.mean(white_ratios))
    correlated_ratio = float(np.mean(corr_ratios))
    if not 0.95 < white_ratio < 1.05:
        raise AssertionError(f"White-noise kernel validation failed: {white_ratio}")
    if abs(correlated_ratio - 1.0) < 0.05:
        raise AssertionError("Correlated-noise test did not depart from naive variance")
    return {
        "white_noise_measured_over_expected": white_ratio,
        "correlated_noise_measured_over_naive": correlated_ratio,
    }
