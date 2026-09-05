from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-paus-wcs-region-selection")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, label as connected_labels
from skimage.draw import polygon
from skimage.feature import peak_local_max
from skimage.measure import approximate_polygon, find_contours


DS9_HEADER = (
    "# Region file format: DS9 version 4.1\n"
    "global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" "
    "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
    "fk5\n"
)


@dataclass
class Product:
    data: np.ndarray
    header: fits.Header
    wcs: WCS
    valid: np.ndarray
    x0: int
    y0: int
    centre_x: float
    centre_y: float
    background: float
    sigma: float

    @property
    def centre_crop_xy(self) -> tuple[float, float]:
        return self.centre_x - self.x0, self.centre_y - self.y0


@dataclass
class Candidate:
    rank: int
    original_label: int
    area_pix: int
    centroid_x: float
    centroid_y: float
    radius_from_lens_pix: float
    symmetric_overlap_fraction: float
    auto_score: float
    mean_signal: float
    reference_iou: float | None = None
    reference_precision: float | None = None
    reference_recall: float | None = None
    reference_centroid_distance_pix: float | None = None
    reference_true_positive_pix: int | None = None
    reference_false_positive_pix: int | None = None
    reference_false_negative_pix: int | None = None
    reference_excess_fraction: float | None = None
    reference_strict_match: float | None = None


def _finite_bbox(data: np.ndarray, padding: int = 2) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(np.isfinite(data))
    if len(xx) == 0:
        raise ValueError("WCS product contains no finite pixels")
    y0 = max(int(yy.min()) - padding, 0)
    y1 = min(int(yy.max()) + padding + 1, data.shape[0])
    x0 = max(int(xx.min()) - padding, 0)
    x1 = min(int(xx.max()) + padding + 1, data.shape[1])
    return x0, x1, y0, y1


def _robust_outer_background(
    data: np.ndarray, valid: np.ndarray, centre_xy: tuple[float, float]
) -> tuple[float, float]:
    yy, xx = np.indices(data.shape)
    radius = np.hypot(xx - centre_xy[0], yy - centre_xy[1])
    finite_r = radius[valid]
    lower = np.quantile(finite_r, 0.72)
    outer = data[valid & (radius >= lower)]
    background = float(np.median(outer))
    sigma = float(1.4826 * np.median(np.abs(outer - background)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(outer))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(abs(background) * 1.0e-6, 1.0e-6)
    return background, sigma


def load_product(path: Path, centre: dict, minimum_half_size_pix: int = 30) -> Product:
    data_full, header = fits.getdata(path, header=True)
    data_full = np.asarray(data_full, dtype=float)
    centre_x = float(centre["fits_x_pixel_zero_indexed"])
    centre_y = float(centre["fits_y_pixel_zero_indexed"])
    x0, x1, y0, y1 = _finite_bbox(data_full)
    centre_ix = int(round(centre_x))
    centre_iy = int(round(centre_y))
    x0 = max(min(x0, centre_ix - minimum_half_size_pix), 0)
    x1 = min(max(x1, centre_ix + minimum_half_size_pix + 1), data_full.shape[1])
    y0 = max(min(y0, centre_iy - minimum_half_size_pix), 0)
    y1 = min(max(y1, centre_iy + minimum_half_size_pix + 1), data_full.shape[0])
    data = data_full[y0:y1, x0:x1]
    valid = np.isfinite(data)
    centre_crop = (centre_x - x0, centre_y - y0)
    background, sigma = _robust_outer_background(data, valid, centre_crop)
    return Product(
        data=data,
        header=header,
        wcs=WCS(header),
        valid=valid,
        x0=x0,
        y0=y0,
        centre_x=centre_x,
        centre_y=centre_y,
        background=background,
        sigma=sigma,
    )


def find_sextractor(explicit: str | None = None) -> Path:
    candidates: list[Path | None] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(Path(p) if p else None for p in (
        shutil.which("source-extractor"),
        shutil.which("sextractor"),
        shutil.which("sex"),
    ))
    # The modeling and BCNz environments normally share one conda envs directory.
    executable = Path(sys.executable).resolve()
    if "envs" in executable.parts:
        envs_index = executable.parts.index("envs")
        envs_root = Path(*executable.parts[: envs_index + 1])
        candidates.extend([
            envs_root / "bcnz/bin/source-extractor",
            envs_root / "bcnz/bin/sex",
        ])
    for candidate in candidates:
        if candidate and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "Cannot find SExtractor. Activate the BCNz environment or pass --sextractor."
    )


def _crop_header(product: Product) -> fits.Header:
    header = product.header.copy()
    if "CRPIX1" in header:
        header["CRPIX1"] = float(header["CRPIX1"]) - product.x0
    if "CRPIX2" in header:
        header["CRPIX2"] = float(header["CRPIX2"]) - product.y0
    return header


def run_sextractor(
    product: Product,
    work_dir: Path,
    executable: Path,
    settings: dict,
) -> np.ndarray:
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "input.fits"
    segmentation_path = work_dir / "segmentation.fits"
    background_path = work_dir / "background.fits"
    weight_path = work_dir / "rms_weight.fits"
    catalog_path = work_dir / "sources.cat"
    parameter_path = work_dir / "default.param"
    convolution_path = work_dir / "default.conv"
    config_path = work_dir / "run.sex"

    input_mode = str(settings.get("input_mode", "raw"))
    if input_mode == "snr":
        signal = (product.data - product.background) / product.sigma
        filled = np.where(product.valid, signal, 0.0)
        peak_snr = float(np.max(signal[product.valid]))
        detect_threshold = max(
            float(settings["detect_snr_floor"]),
            float(settings["detect_peak_fraction"]) * peak_snr,
        )
        # Fix the noise scale to one in valid pixels. Very large RMS outside the
        # modeling mask prevents the filled canvas from becoming a detection.
        rms_map = np.where(product.valid, 1.0, 1.0e6)
        fits.writeto(weight_path, rms_map.astype(np.float32), _crop_header(product), overwrite=True)
        background_value = 0.0
        weight_config = f"WEIGHT_TYPE        MAP_RMS\nWEIGHT_IMAGE       {weight_path}\n"
    elif input_mode == "raw":
        filled = np.where(product.valid, product.data, product.background)
        detect_threshold = float(settings["detect_thresh"])
        background_value = product.background
        weight_config = "WEIGHT_TYPE        NONE\n"
    else:
        raise ValueError(f"Unknown SExtractor input_mode={input_mode!r}")
    fits.writeto(input_path, filled.astype(np.float32), _crop_header(product), overwrite=True)
    parameter_path.write_text(
        "NUMBER\nX_IMAGE\nY_IMAGE\nXWIN_IMAGE\nYWIN_IMAGE\nISOAREA_IMAGE\nFLUX_ISO\nFLAGS\n"
    )
    convolution_path.write_text("CONV NORM\n1 2 1\n2 4 2\n1 2 1\n")
    config_path.write_text(
        f"""CATALOG_NAME       {catalog_path}
CATALOG_TYPE       ASCII_HEAD
PARAMETERS_NAME    {parameter_path}
DETECT_TYPE        CCD
DETECT_MINAREA     {int(settings['detect_minarea'])}
DETECT_THRESH      {detect_threshold}
ANALYSIS_THRESH    {detect_threshold}
FILTER             Y
FILTER_NAME        {convolution_path}
DEBLEND_NTHRESH    {int(settings['deblend_nthresh'])}
DEBLEND_MINCONT    {float(settings['deblend_mincont'])}
CLEAN              Y
CLEAN_PARAM        1.0
BACK_TYPE          MANUAL
BACK_VALUE         {background_value:.12g}
BACK_SIZE          {int(settings['back_size'])}
BACK_FILTERSIZE    3
CHECKIMAGE_TYPE    SEGMENTATION,BACKGROUND
CHECKIMAGE_NAME    {segmentation_path},{background_path}
{weight_config}GAIN               1.0
PIXEL_SCALE        0
SATUR_LEVEL        1.0e9
VERBOSE_TYPE       NORMAL
WRITE_XML          N
"""
    )
    command = [str(executable), str(input_path), "-c", str(config_path)]
    result = subprocess.run(command, cwd=work_dir, text=True, capture_output=True, check=False)
    (work_dir / "sextractor.log").write_text(
        "COMMAND:\n" + " ".join(command) + "\n\nSTDOUT:\n" + result.stdout
        + "\n\nSTDERR:\n" + result.stderr
    )
    if result.returncode != 0:
        raise RuntimeError(f"SExtractor failed; inspect {work_dir / 'sextractor.log'}")
    segmentation = np.asarray(fits.getdata(segmentation_path), dtype=np.int32)
    segmentation[~product.valid] = 0
    return segmentation


def _component_template(
    signal: np.ndarray,
    peak_yx: tuple[int, int],
    all_peaks_yx: np.ndarray,
    half_size: int,
) -> np.ndarray:
    py, px = peak_yx
    size = 2 * half_size + 1
    y0, y1 = max(py - half_size, 0), min(py + half_size + 1, signal.shape[0])
    x0, x1 = max(px - half_size, 0), min(px + half_size + 1, signal.shape[1])
    patch = np.asarray(signal[y0:y1, x0:x1], dtype=float).copy()
    yy, xx = np.indices(patch.shape)
    yy_full, xx_full = yy + y0, xx + x0
    distances = np.stack(
        [(yy_full - qy) ** 2 + (xx_full - qx) ** 2 for qy, qx in all_peaks_yx], axis=0
    )
    owner = np.argmin(distances, axis=0)
    own_index = int(np.flatnonzero(np.all(all_peaks_yx == np.asarray(peak_yx), axis=1))[0])
    patch[owner != own_index] = 0.0
    full = np.zeros((size, size), dtype=float)
    oy = half_size - (py - y0)
    ox = half_size - (px - x0)
    full[oy : oy + patch.shape[0], ox : ox + patch.shape[1]] = np.clip(patch, 0, None)
    total = float(full.sum())
    if total <= 0:
        full[half_size, half_size] = 1.0
        total = 1.0
    return full / total


def run_scarlet2(product: Product, settings: dict, central_only: bool) -> np.ndarray:
    # Imported lazily so SExtractor-only diagnostics do not require JAX/Scarlet2.
    import jax.numpy as jnp
    import scarlet2
    from numpyro.distributions import constraints

    scarlet2.set_validation(False)
    signal = np.where(
        product.valid,
        np.clip((product.data - product.background) / product.sigma, 0.0, None),
        0.0,
    )
    centre_x, centre_y = product.centre_crop_xy
    if central_only:
        peaks = np.asarray([[int(round(centre_y)), int(round(centre_x))]], dtype=int)
    else:
        peaks = peak_local_max(
            gaussian_filter(signal, float(settings["peak_smoothing_sigma_pix"])),
            min_distance=int(settings["peak_min_distance_pix"]),
            threshold_abs=float(settings["peak_snr"]),
            exclude_border=False,
            num_peaks=int(settings["max_sources"]),
        )
        if len(peaks) == 0:
            peaks = np.asarray([np.unravel_index(np.argmax(signal), signal.shape)], dtype=int)

    channels = ["wcs_product"]
    obs = scarlet2.Observation(
        data=jnp.asarray(signal[None]),
        weights=jnp.asarray(product.valid[None].astype(float)),
        channels=channels,
        name="automated_wcs_region_selection",
    )
    model_frame = scarlet2.Frame(obs.frame.bbox, channels=channels)
    obs.match(model_frame)
    half_size = int(settings["morphology_half_size_pix"])
    with scarlet2.Scene(model_frame) as scene:
        for py, px in peaks:
            morphology = _component_template(signal, (int(py), int(px)), peaks, half_size)
            spectrum = scarlet2.StaticArraySpectrum(
                jnp.asarray([max(float(signal[py, px]), 1.0e-3)]),
                bands=channels,
                band_selector=lambda channel: channel,
            )
            scarlet2.Source(
                jnp.asarray([float(py), float(px)]),
                spectrum,
                jnp.asarray(morphology),
            )
    with scarlet2.Parameters(scene):
        for index, source in enumerate(scene.sources):
            scarlet2.Parameter(
                source.spectrum.data,
                name=f"source_{index}_amplitude",
                constraint=constraints.positive,
                stepsize=scarlet2.relative_step,
            )
    fitted = scene.fit(
        obs,
        max_iter=int(settings["max_iter"]),
        e_rel=float(settings["relative_tolerance"]),
        progress_bar=False,
    )
    component_models = np.stack(
        [np.asarray(obs.render(fitted.evaluate_source(source)))[0] for source in fitted.sources],
        axis=0,
    )
    threshold = np.maximum(
        float(settings["footprint_snr"]),
        float(settings["footprint_peak_fraction"]) * component_models.max(axis=(1, 2)),
    )
    scores = component_models / threshold[:, None, None]
    best = np.argmax(scores, axis=0)
    best_score = np.max(scores, axis=0)
    segmentation = np.zeros(signal.shape, dtype=np.int32)
    segmentation[best_score >= 1.0] = best[best_score >= 1.0] + 1
    segmentation[~product.valid] = 0
    return segmentation


def _split_components(segmentation: np.ndarray, min_area: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for source_label in (int(v) for v in np.unique(segmentation) if v > 0):
        components, count = connected_labels(segmentation == source_label)
        for component in range(1, count + 1):
            mask = components == component
            if int(mask.sum()) >= min_area:
                masks.append(mask)
    return masks


def _reflect_mask(mask: np.ndarray, centre_xy: tuple[float, float]) -> np.ndarray:
    reflected = np.zeros_like(mask, dtype=bool)
    yy, xx = np.nonzero(mask)
    xr = np.rint(2.0 * centre_xy[0] - xx).astype(int)
    yr = np.rint(2.0 * centre_xy[1] - yy).astype(int)
    inside = (xr >= 0) & (xr < mask.shape[1]) & (yr >= 0) & (yr < mask.shape[0])
    reflected[yr[inside], xr[inside]] = True
    return reflected


def _reference_metrics(mask: np.ndarray, reference: np.ndarray | None) -> dict:
    if reference is None:
        return {}
    intersection = int(np.sum(mask & reference))
    union = int(np.sum(mask | reference))
    predicted = int(mask.sum())
    reference_area = int(reference.sum())
    false_positive = predicted - intersection
    false_negative = reference_area - intersection
    yy, xx = np.nonzero(mask)
    ry, rx = np.nonzero(reference)
    return {
        "reference_iou": intersection / union if union else 0.0,
        "reference_precision": intersection / predicted if predicted else 0.0,
        "reference_recall": intersection / reference_area if reference_area else 0.0,
        "reference_centroid_distance_pix": float(
            np.hypot(xx.mean() - rx.mean(), yy.mean() - ry.mean())
        ) if len(xx) and len(rx) else math.nan,
        "reference_true_positive_pix": intersection,
        "reference_false_positive_pix": false_positive,
        "reference_false_negative_pix": false_negative,
        "reference_excess_fraction": false_positive / reference_area if reference_area else math.nan,
        # Missing and excess pixels receive exactly the same unit penalty.
        "reference_strict_match": 1.0 - (false_positive + false_negative) / reference_area
        if reference_area else math.nan,
    }


def rank_source_masks(
    segmentation: np.ndarray,
    product: Product,
    min_area: int,
    reference: np.ndarray | None,
) -> tuple[np.ndarray, list[Candidate], list[np.ndarray]]:
    masks = _split_components(segmentation, min_area=min_area)
    if not masks:
        raise RuntimeError("Detector produced no source-region candidate above min_area")
    union = np.logical_or.reduce(masks)
    centre_xy = product.centre_crop_xy
    rows = []
    for original_label, mask in enumerate(masks, start=1):
        yy, xx = np.nonzero(mask)
        reflected = _reflect_mask(mask, centre_xy)
        overlap = float(np.sum(reflected & union) / max(int(reflected.sum()), 1))
        area = int(mask.sum())
        mean_signal = float(np.mean(product.data[mask] - product.background))
        radius = float(np.hypot(xx.mean() - centre_xy[0], yy.mean() - centre_xy[1]))
        # A central/self-symmetric footprint receives a strong penalty. Among
        # cleanly reflected candidates, larger coherent regions remain preferred.
        auto_score = float(area * max(1.0 - overlap, 0.0) ** 2)
        metrics = _reference_metrics(mask, reference)
        rows.append((auto_score, area, mean_signal, original_label, mask, overlap, radius, xx, yy, metrics))
    rows.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    ranked_map = np.zeros_like(segmentation, dtype=np.int32)
    candidates: list[Candidate] = []
    ranked_masks: list[np.ndarray] = []
    for rank, row in enumerate(rows, start=1):
        score, area, mean_signal, original_label, mask, overlap, radius, xx, yy, metrics = row
        ranked_map[mask] = rank
        ranked_masks.append(mask)
        candidates.append(Candidate(
            rank=rank,
            original_label=original_label,
            area_pix=area,
            centroid_x=float(xx.mean() + product.x0),
            centroid_y=float(yy.mean() + product.y0),
            radius_from_lens_pix=radius,
            symmetric_overlap_fraction=overlap,
            auto_score=score,
            mean_signal=mean_signal,
            **metrics,
        ))
    return ranked_map, candidates, ranked_masks


def select_central_lens_mask(segmentation: np.ndarray, product: Product) -> np.ndarray:
    masks = _split_components(segmentation, min_area=1)
    if not masks:
        raise RuntimeError("Detector produced no central lens region")
    centre_x, centre_y = product.centre_crop_xy
    distances = []
    for mask in masks:
        yy, xx = np.nonzero(mask)
        distances.append(float(np.hypot(xx.mean() - centre_x, yy.mean() - centre_y)))
    return masks[int(np.argmin(distances))]


def parse_ds9_polygon(reference_path: Path | None, product: Product) -> np.ndarray | None:
    if reference_path is None or not reference_path.exists():
        return None
    text = reference_path.read_text()
    match = re.search(r"polygon\((.*?)\)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"No polygon found in reference region: {reference_path}")
    values = np.asarray([float(value) for value in match.group(1).split(",")]).reshape(-1, 2)
    px, py = product.wcs.world_to_pixel_values(values[:, 0], values[:, 1])
    px -= product.x0
    py -= product.y0
    mask = np.zeros(product.data.shape, dtype=bool)
    rr, cc = polygon(py, px, shape=mask.shape)
    mask[rr, cc] = True
    return mask


def _mask_contours_full_pixel(mask: np.ndarray, product: Product) -> Iterable[np.ndarray]:
    padded = np.pad(mask.astype(float), 1)
    for contour_yx in find_contours(padded, 0.5):
        contour_yx = approximate_polygon(contour_yx - 1.0, tolerance=0.35)
        if len(contour_yx) < 3:
            continue
        xy = np.column_stack((
            contour_yx[:, 1] + product.x0,
            contour_yx[:, 0] + product.y0,
        ))
        yield xy


def write_ds9_regions(
    path: Path,
    masks: list[np.ndarray],
    product: Product,
    labels: list[str],
    colors: list[str] | None = None,
) -> None:
    colors = colors or ["green"] * len(masks)
    lines = [DS9_HEADER]
    for mask, label, color in zip(masks, labels, colors, strict=True):
        for xy in _mask_contours_full_pixel(mask, product):
            ra, dec = product.wcs.pixel_to_world_values(xy[:, 0], xy[:, 1])
            coordinates = ",".join(f"{value:.8f}" for pair in zip(ra, dec, strict=True) for value in pair)
            lines.append(f"polygon({coordinates}) # color={color} text={{{label}}}\n")
    path.write_text("".join(lines))


def write_point_reflected_ds9_region(
    path: Path,
    mask: np.ndarray,
    product: Product,
    label: str = "background_point_reflection",
    color: str = "cyan",
    centre_full_xy: tuple[float, float] | None = None,
) -> None:
    """Write the exact 180-degree reflection of a mask contour in WCS.

    Reflection is applied to the continuous full-image pixel coordinates of
    each contour, not to a rounded boolean mask on the cropped analysis grid.
    Consequently, a reflected contour may extend outside that diagnostic crop
    without being truncated before downstream photometry rasterizes it on the
    original NB/BB images.
    """
    centre_x, centre_y = (
        centre_full_xy
        if centre_full_xy is not None
        else (product.centre_x, product.centre_y)
    )
    lines = [DS9_HEADER]
    for xy in _mask_contours_full_pixel(mask, product):
        reflected_xy = np.column_stack((
            2.0 * centre_x - xy[:, 0],
            2.0 * centre_y - xy[:, 1],
        ))
        ra, dec = product.wcs.pixel_to_world_values(
            reflected_xy[:, 0], reflected_xy[:, 1]
        )
        coordinates = ",".join(
            f"{value:.8f}"
            for pair in zip(ra, dec, strict=True)
            for value in pair
        )
        lines.append(
            f"polygon({coordinates}) # color={color} text={{{label}}}\n"
        )
    if len(lines) == 1:
        raise RuntimeError("Cannot write a reflected region from an empty mask")
    path.write_text("".join(lines))


def write_catalog(path: Path, candidates: list[Candidate]) -> None:
    fieldnames = list(asdict(candidates[0]).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(candidate) for candidate in candidates)


def _display_norm(product: Product) -> ImageNormalize:
    values = product.data[product.valid]
    return ImageNormalize(values, interval=PercentileInterval(99.5), stretch=AsinhStretch(0.06))


def save_method_preview(
    path: Path,
    method: str,
    lens: Product,
    source: Product,
    lens_mask: np.ndarray,
    ranked_map: np.ndarray,
    candidates: list[Candidate],
    ranked_masks: list[np.ndarray],
    reference: np.ndarray | None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].imshow(lens.data, origin="lower", cmap="gray", norm=_display_norm(lens))
    axes[0].contour(lens_mask, levels=[0.5], colors="magenta", linewidths=2, origin="lower")
    axes[0].plot(*lens.centre_crop_xy, "+", color="cyan", ms=13, mew=2)
    axes[0].set_title(f"{method}: central lens region")

    axes[1].imshow(source.data, origin="lower", cmap="gray", norm=_display_norm(source))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(candidates), 1)))
    for candidate, mask, color in zip(candidates, ranked_masks, colors, strict=True):
        axes[1].contour(mask, levels=[0.5], colors=[color], linewidths=2, origin="lower")
        axes[1].text(
            candidate.centroid_x - source.x0,
            candidate.centroid_y - source.y0,
            str(candidate.rank), color=color, fontsize=12, weight="bold",
        )
    if reference is not None:
        axes[1].contour(reference, levels=[0.5], colors="lime", linewidths=1.5, linestyles="--", origin="lower")
    axes[1].plot(*source.centre_crop_xy, "+", color="magenta", ms=13, mew=2)
    axes[1].set_title(f"{method}: numbered source candidates")

    selected = ranked_masks[0]
    reflected = _reflect_mask(selected, source.centre_crop_xy)
    axes[2].imshow(source.data, origin="lower", cmap="gray", norm=_display_norm(source))
    axes[2].contour(selected, levels=[0.5], colors="yellow", linewidths=2.2, origin="lower")
    axes[2].contour(reflected, levels=[0.5], colors="cyan", linewidths=1.8, linestyles="--", origin="lower")
    if reference is not None:
        axes[2].contour(reference, levels=[0.5], colors="lime", linewidths=1.5, linestyles=":", origin="lower")
    axes[2].plot(*source.centre_crop_xy, "+", color="magenta", ms=13, mew=2)
    axes[2].set_title(
        f"selected #1: area={candidates[0].area_pix}, "
        f"sym={candidates[0].symmetric_overlap_fraction:.3f}"
    )
    for ax in axes:
        ax.set_xlabel("cropped x pixel")
        ax.set_ylabel("cropped y pixel")
    fig.suptitle(
        "Contours: candidate/selected; cyan dashed: point reflection; green: manual reference",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_preview(
    path: Path,
    source: Product,
    results: dict,
    reference: np.ndarray | None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), squeeze=False)
    for ax, (method, result) in zip(axes.ravel(), results.items(), strict=True):
        ax.imshow(source.data, origin="lower", cmap="gray", norm=_display_norm(source))
        selected = result["ranked_masks"][0]
        reflected = _reflect_mask(selected, source.centre_crop_xy)
        ax.contour(selected, levels=[0.5], colors="yellow", linewidths=2.2, origin="lower")
        ax.contour(reflected, levels=[0.5], colors="cyan", linewidths=1.6, linestyles="--", origin="lower")
        if reference is not None:
            ax.contour(reference, levels=[0.5], colors="lime", linewidths=1.5, linestyles=":", origin="lower")
        candidate = result["candidates"][0]
        iou = "n/a" if candidate.reference_iou is None else f"{candidate.reference_iou:.3f}"
        precision = "n/a" if candidate.reference_precision is None else f"{candidate.reference_precision:.3f}"
        ax.set_title(
            f"{method}: selected #1\nIoU={iou}, precision={precision}, "
            f"sym={candidate.symmetric_overlap_fraction:.3f}"
        )
        ax.plot(*source.centre_crop_xy, "+", color="magenta", ms=13, mew=2)
        ax.set_xlabel("cropped x pixel")
        ax.set_ylabel("cropped y pixel")
    fig.suptitle("System source-region method comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_comparison(
    project_dir: Path,
    system_id: str,
    config_path: Path,
    output_dir: Path | None = None,
    reference_path: Path | None = None,
    sextractor_path: str | None = None,
) -> dict:
    project_dir = project_dir.resolve()
    config = json.loads(config_path.read_text())
    products_dir = project_dir / f"system_{system_id}" / "output/wcs_products"
    centre_path = products_dir / f"{system_id}_lens_light_centre_wcs.json"
    centre = json.loads(centre_path.read_text())
    analysis_half_size = int(config["selection"].get("analysis_half_size_pix", 30))
    lens = load_product(
        products_dir / f"{system_id}_lens_light_model_wcs.fits", centre, analysis_half_size
    )
    source = load_product(
        products_dir / f"{system_id}_lens_light_reduced_wcs.fits", centre, analysis_half_size
    )
    output_dir = (output_dir or products_dir / "automated_regions").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = parse_ds9_polygon(reference_path, source)
    sextractor = find_sextractor(sextractor_path)

    results = {}
    for method in ("sextractor", "scarlet2"):
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        if method == "sextractor":
            lens_raw = run_sextractor(lens, method_dir / "work_lens", sextractor, config[method]["lens"])
            source_raw = run_sextractor(source, method_dir / "work_source", sextractor, config[method]["source"])
        else:
            lens_raw = run_scarlet2(lens, config[method]["lens"], central_only=True)
            source_raw = run_scarlet2(source, config[method]["source"], central_only=False)

        lens_mask = select_central_lens_mask(lens_raw, lens)
        if bool(config["selection"].get("subtract_lens_region_from_source", False)):
            if lens_mask.shape != source_raw.shape:
                raise ValueError(
                    f"Lens/source analysis grids differ for {method}: "
                    f"{lens_mask.shape} versus {source_raw.shape}"
                )
            source_raw = np.asarray(source_raw).copy()
            source_raw[lens_mask] = 0
        ranked_map, candidates, ranked_masks = rank_source_masks(
            source_raw,
            source,
            min_area=int(config["selection"]["min_area_pix"]),
            reference=reference,
        )
        write_ds9_regions(
            method_dir / f"system{system_id}_{method}_lens_central.reg",
            [lens_mask], lens, ["lens central"], ["magenta"],
        )
        write_ds9_regions(
            method_dir / f"system{system_id}_{method}_source_candidates.reg",
            ranked_masks, source, [f"source {c.rank}" for c in candidates],
            ["yellow" if c.rank == 1 else "cyan" for c in candidates],
        )
        write_ds9_regions(
            method_dir / f"system{system_id}_{method}_source_selected.reg",
            [ranked_masks[0]], source, ["selected source 1"], ["yellow"],
        )
        write_catalog(method_dir / f"system{system_id}_{method}_source_candidates.csv", candidates)
        fits.writeto(
            method_dir / f"system{system_id}_{method}_source_ranked_segmentation.fits",
            ranked_map.astype(np.int16), _crop_header(source), overwrite=True,
        )
        save_method_preview(
            method_dir / f"system{system_id}_{method}_preview.png",
            method, lens, source, lens_mask, ranked_map, candidates, ranked_masks, reference,
        )
        results[method] = {
            "lens_mask": lens_mask,
            "ranked_map": ranked_map,
            "candidates": candidates,
            "ranked_masks": ranked_masks,
        }

    save_comparison_preview(
        output_dir / f"system{system_id}_scarlet2_vs_sextractor.png", source, results, reference
    )
    comparison_rows = []
    for method, result in results.items():
        selected = asdict(result["candidates"][0])
        selected["method"] = method
        comparison_rows.append(selected)
    fieldnames = ["method", *asdict(results["sextractor"]["candidates"][0]).keys()]
    with (output_dir / f"system{system_id}_method_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    if reference is not None:
        recommended = max(
            results,
            key=lambda name: results[name]["candidates"][0].reference_iou or -1.0,
        )
        recommendation_basis = "highest IoU against reference region (evaluation only)"
    else:
        recommended = max(
            results,
            key=lambda name: results[name]["candidates"][0].auto_score,
        )
        recommendation_basis = "highest automatic area/symmetry score"
    summary = {
        "system": str(system_id),
        "inputs": {
            "lens_model": str(products_dir / f"{system_id}_lens_light_model_wcs.fits"),
            "lens_reduced_source": str(products_dir / f"{system_id}_lens_light_reduced_wcs.fits"),
            "lens_light_centre": str(centre_path),
            "reference_evaluation_only": str(reference_path) if reference_path else None,
        },
        "reference_was_not_used_for_detection_or_ranking": True,
        "recommended_method_for_this_system": recommended,
        "recommendation_basis": recommendation_basis,
        "selected": {name: asdict(result["candidates"][0]) for name, result in results.items()},
        "settings": config,
    }
    (output_dir / f"system{system_id}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    sex = summary["selected"]["sextractor"]
    scarlet = summary["selected"]["scarlet2"]
    winner = "Scarlet2" if scarlet["reference_iou"] > sex["reference_iou"] else "SExtractor"
    loser = "SExtractor" if winner == "Scarlet2" else "Scarlet2"
    winner_iou = max(scarlet["reference_iou"], sex["reference_iou"])
    loser_iou = min(scarlet["reference_iou"], sex["reference_iou"])
    report = f"""# System {system_id}: Scarlet2 与 SExtractor 自动选区对比

## 结论

在 system {system_id} 的本次配置中，按人工参考区 IoU，**{winner} 略优于 {loser}**。
Scarlet2 的 IoU 为 {scarlet['reference_iou']:.3f}，SExtractor 为 {sex['reference_iou']:.3f}；
差值为 {winner_iou - loser_iou:.3f}，不能据此推断该方法对其他系统必然更好。

| 方法 | 面积 [pix] | 对称重叠率 | IoU | precision | recall | 中心距离 [pix] |
|---|---:|---:|---:|---:|---:|---:|
| SExtractor | {sex['area_pix']} | {sex['symmetric_overlap_fraction']:.3f} | {sex['reference_iou']:.3f} | {sex['reference_precision']:.3f} | {sex['reference_recall']:.3f} | {sex['reference_centroid_distance_pix']:.3f} |
| Scarlet2 | {scarlet['area_pix']} | {scarlet['symmetric_overlap_fraction']:.3f} | {scarlet['reference_iou']:.3f} | {scarlet['reference_precision']:.3f} | {scarlet['reference_recall']:.3f} | {scarlet['reference_centroid_distance_pix']:.3f} |

具体取舍应同时参考 precision、recall 与点对称重叠率，而不能只看面积或 IoU。本系统若以
复现当前参考 region 为唯一目标，推荐 {winner}。

## 防止信息泄漏

人工参考 region 只在检测和自动排序全部完成后用于计算 IoU、precision、recall 和中心距离；
它没有参与峰值初始化、分割、点对称评分或候选排序。自动排序只使用面积与绕 lens-light
centre 的点对称重叠率。

## 方法边界

这里调用的是项目环境中实际安装的 **Scarlet2**。输入 reduced WCS product 只有单通道，
因此没有多波段颜色信息帮助 Scarlet2 解除简并；结果主要由自动空间峰和非负形态模板决定。
这一点决定了当前对比是“单图空间选区”对比，而不是完整的多波段 Scarlet deblending。
在推广到其他系统前，应至少再检查几个具有多个 source 岛、明显噪声或遮罩边界的系统。
"""
    (output_dir / f"system{system_id}_comparison_report.md").write_text(report)
    return summary
