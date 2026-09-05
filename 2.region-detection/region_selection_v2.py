#!/usr/bin/env python3
"""Stable image-only SExtractor source-region selection, v2.

Differences against ``adaptive_region_selection.py`` (v1), all of which were
motivated by measuring the manual reference regions against v1's own gates:

* The detection threshold is an absolute signal-to-noise isophote rather than a
  fraction of the brightest pixel. A fraction-of-peak threshold ties the
  isophote to the sharpest knot in the frame, which is why one fixed value
  behaved completely differently across the five systems.
* The isophote is chosen by maximising the region's integrated signal-to-noise
  rather than by taking whichever admissible trial the scan reaches first. v1
  scanned bright-to-faint and kept the first pass, which has no optimum -- it
  simply returns the brightest gate-compatible isophote. Reversing the scan has
  the mirror problem of running to the scan floor. Integrated S/N turns over on
  its own, and ``SNR_TOLERANCE_FRACTION`` sets how much of it may be traded for
  completeness.
* ``MIN_EFFECTIVE_ISOPHOTE_SNR`` is gone. It constrained the isophote *level*,
  not the detection significance, and for the faintest systems it forced the
  threshold above half the peak. It is replaced by an integrated signal-to-noise
  gate plus a per-pixel isophote floor.
* ``MIN_CENTREWARD_CLEARANCE_RATIO`` is gone. ``r_min / r_centroid`` penalises
  radially extended arcs, which is precisely the morphology being selected; the
  manual regions score 0.618-0.784 against a 0.78 gate. It is replaced by the
  mirror-asymmetry contamination measure described at ``MIN_MIRROR_ASYMMETRY``.
* Exactly one detector component is retained. Candidates are considered by
  decreasing area and must have neither their source footprint nor its point
  reflection overlap any other detected source footprint. If every candidate
  conflicts, the largest one is retained after carving away both direct and
  reflected-other-source conflicts.
* The seed isophote is grown by hysteresis to a fainter level and optionally
  dilated, recovering the PSF wings that a single isophote cuts off.
* Stability is judged on integrated flux rather than on the pixel set, because
  flux is the quantity propagated to the photometry step.

This stable release intentionally contains only the SExtractor detector. The
manual reference region is read only after the configuration has been frozen,
and only to score the result.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    label as connected_labels,
    map_coordinates,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_ROOT = SCRIPT_DIR.parent / "1.lens-modeling"
if not MODELING_ROOT.is_dir():
    raise FileNotFoundError(f"Missing sibling modeling package: {MODELING_ROOT}")
sys.path.insert(0, str(MODELING_ROOT))

from region_selection.pipeline import (  # noqa: E402
    _reference_metrics,
    _display_norm,
    _reflect_mask,
    find_sextractor,
    load_product,
    parse_ds9_polygon,
    run_sextractor,
    select_central_lens_mask,
    write_ds9_regions,
    write_point_reflected_ds9_region,
)
from lens_region_methods_v2 import (  # noqa: E402
    DEFAULT_METHOD,
    build_lens_masks,
    evaluate_lens_mask,
    source_peak_anchor,
)


SYSTEMS = tuple(item["id"] for item in json.loads((Path(__file__).resolve().parents[1] / "config/systems.json").read_text()))

# Threshold scan, in units of the robust outer-region sigma, run descending so
# that adjacent trials are nested and the flux-stability metric is meaningful.
THRESHOLD_START_SIGMA = 8.0
THRESHOLD_STOP_SIGMA = 1.5
THRESHOLD_STEP_SIGMA = 0.25

# SExtractor takes max(snr_floor, peak_fraction * peak); forcing the peak
# fraction to zero makes the absolute sigma floor the only threshold in play.

# --- Assembly parameters (new in v2) ---------------------------------------
# Hysteresis: the seed isophote is grown down to this fraction of the detection
# threshold, inside its own connected component, but never below
# ``MIN_ISOPHOTE_SIGMA``. The level has to track the detection threshold; a
# fixed absolute growth level collapses every trial onto the same connected
# component and makes the scan meaningless. Set to None to disable growth.
GROW_SIGMA_RATIO = 0.5
# A grown region may not exceed this multiple of its seed area; runaway growth
# through a noise bridge is rejected rather than trusted.
MAX_GROW_AREA_RATIO = 2.5
# Final binary dilation, in pixels, to recover the outermost PSF wing.
DILATE_PIX = 0
# --- Gates (all image-only, fixed before the reference region is read) -------
# Every pixel in the region sits at or above this isophote by construction; the
# gate documents the floor of the scan.
MIN_ISOPHOTE_SIGMA = 1.5
# Detection significance of the region as a whole.  The original HSC sample
# passes the morphology/isolation gates. A 15-sigma floor remains highly
# conservative without turning the HSC-tuned value of 20 into a CFHT veto.
MIN_INTEGRATED_SNR = 15.0
# Contamination gate. (f_region - f_mirror) / f_region on the residual image,
# where the mirror is the point reflection about the lens centre and therefore
# contains lens light and background but no source. This is the same statistic
# the downstream PAUS photometry relies on, and it needs no lens-light model.
# The manual reference regions score 0.730-1.022, so 0.60 is a wide margin.
#
# Note: the obvious alternative -- lens-model flux inside the region, relative
# to the residual flux there -- was measured and discarded. The manual regions
# score 0.14-0.89 on it, because the arc lies at the Einstein radius where the
# deflector's own light is intrinsically bright. It says nothing about how much
# lens light survives into the residual, so it is reported as a diagnostic only.
MIN_MIRROR_ASYMMETRY = 0.60
# The region must not overlap its own point reflection about the lens centre;
# that is what makes the symmetric subtraction well posed.
MAX_SYMMETRIC_OVERLAP = 1.0e-12
MIN_AREA_PIX = 12
# Integrated flux may not change by more than this fraction between adjacent
# threshold steps, which replaces v1's pixel-set Jaccard stability gate.
MAX_NEIGHBOUR_FLUX_CHANGE = 0.25

# Completeness/purity knob. The region is taken at the faintest isophote whose
# integrated signal-to-noise is still within this fraction of the achievable
# maximum. 0.0 gives the strict optimum (smallest, purest region); larger values
# trade signal-to-noise for completeness.
SNR_TOLERANCE_FRACTION = 0.03
DEFAULT_EFFECTIVE_SNR_EPSILON = 0.015

MIN_COMPONENT_AREA_PIX = 5
ANALYSIS_HALF_SIZE_PIX = 30
# A modeled satellite is photometrically relevant where its PSF-convolved
# surface brightness is at least one robust residual-image sigma.  This makes
# the footprint adapt to both the fitted light distribution and image depth.
SATELLITE_MODEL_SNR_THRESHOLD = 1.0
SATELLITE_COMPACT_FLUX_RETENTION = 0.50
SATELLITE_COMPACT_THRESHOLD_STEP_SIGMA = 0.05


def positions_gui_centre_on_product(
    positions_path: Path, product
) -> tuple[tuple[float, float], dict]:
    """Map the saved GUI centre onto a WCS product grid.

    Photo-lens normally uses the saved subpixel antisymmetry minimum.  The
    some CFHT position files predate that saved quantity, so this CFHT adapter
    falls back explicitly to ``PIXEL_LENS_LIGHT_CENTRE`` when the floating
    centre is ``None``.  The fallback is recorded in the output summary.
    """
    text = positions_path.read_text()
    method_match = re.search(
        r"^LENS_LIGHT_CENTRE_METHOD\s*=\s*['\"]([^'\"]+)", text, re.M
    )
    float_pixel_match = re.search(
        r"^PIXEL_LENS_LIGHT_CENTRE_FLOAT\s*=\s*\(([^,]+),\s*([^\)]+)\)",
        text,
        re.M,
    )
    integer_pixel_match = re.search(
        r"^PIXEL_LENS_LIGHT_CENTRE\s*=\s*\(([^,]+),\s*([^\)]+)\)",
        text,
        re.M,
    )
    image_match = re.search(r"^# FITS image:\s*(.+?)\s*$", text, re.M)
    if not (method_match and image_match and (float_pixel_match or integer_pixel_match)):
        raise ValueError(
            f"Missing FITS image, method, or pixel centre in {positions_path}"
        )
    method = method_match.group(1)
    pixel_match = float_pixel_match or integer_pixel_match
    centre_value_source = (
        "PIXEL_LENS_LIGHT_CENTRE_FLOAT"
        if float_pixel_match is not None
        else "PIXEL_LENS_LIGHT_CENTRE_integer_fallback"
    )
    image_path = Path(image_match.group(1)).expanduser()
    if not image_path.is_absolute():
        image_path = positions_path.parent / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"positions_gui FITS image does not exist: {image_path}")
    row = float(pixel_match.group(1))
    col = float(pixel_match.group(2))
    # HSC cutouts keep the science plane (and its WCS) in extension 1 while the
    # primary HDU has NAXIS=0. ``getdata(..., header=True)`` follows the first
    # data-bearing HDU and therefore matches the image displayed by the GUI.
    _, source_header = fits.getdata(image_path, header=True)
    ra_deg, dec_deg = WCS(source_header).celestial.pixel_to_world_values(col, row)
    # Use scalar world values rather than the high-level SkyCoord adapter: the
    # exported primary header intentionally has NAXIS=0 while retaining celestial
    # WCS cards, which can make Astropy's high-level class matching ambiguous.
    full_x, full_y = product.wcs.celestial.world_to_pixel_values(ra_deg, dec_deg)
    full_x = float(full_x)
    full_y = float(full_y)
    if not (np.isfinite(full_x) and np.isfinite(full_y)):
        raise ValueError("positions_gui centre did not map to finite product pixels")
    crop_xy = (full_x - product.x0, full_y - product.y0)
    metadata = {
        "positions_file": str(positions_path.resolve()),
        "input_fits": str(image_path),
        "method": method,
        "centre_value_source": centre_value_source,
        "subpixel_antisymmetry_centre_available": bool(float_pixel_match is not None),
        "input_row_y_zero_indexed": row,
        "input_col_x_zero_indexed": col,
        "ra_deg": float(ra_deg),
        "dec_deg": float(dec_deg),
        "product_full_x_zero_indexed": full_x,
        "product_full_y_zero_indexed": full_y,
        "product_crop_x_zero_indexed": float(crop_xy[0]),
        "product_crop_y_zero_indexed": float(crop_xy[1]),
    }
    return crop_xy, metadata


def scale_mask_about_centre(
    mask: np.ndarray, centre_xy: tuple[float, float], scale: float
) -> np.ndarray:
    """Radially scale a binary aperture about a continuous detector centre."""
    if scale <= 0.0:
        raise ValueError("lens scale must be positive")
    if np.isclose(scale, 1.0):
        return np.asarray(mask, dtype=bool).copy()
    centre_x, centre_y = centre_xy
    yy, xx = np.indices(mask.shape, dtype=float)
    source_y = centre_y + (yy - centre_y) / scale
    source_x = centre_x + (xx - centre_x) / scale
    scaled = map_coordinates(
        np.asarray(mask, dtype=np.uint8),
        [source_y, source_x],
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool)
    return binary_fill_holes(scaled)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def is_clean(mask: np.ndarray, centre_xy: tuple[float, float]) -> bool:
    """Zero overlap with its own point reflection, and clear of the lens centre."""
    if not mask.any():
        return False
    centre_x, centre_y = centre_xy
    row = int(np.clip(round(centre_y), 0, mask.shape[0] - 1))
    col = int(np.clip(round(centre_x), 0, mask.shape[1] - 1))
    if mask[row, col]:
        return False
    return int(np.sum(mask & _reflect_mask(mask, centre_xy))) == 0


def symmetric_overlap(mask: np.ndarray, centre_xy: tuple[float, float]) -> float:
    if not mask.any():
        return 1.0
    return float(np.sum(mask & _reflect_mask(mask, centre_xy)) / mask.sum())


def remove_pixels_reflecting_into_blocker(
    source_mask: np.ndarray,
    blocker_mask: np.ndarray,
    centre_xy: tuple[float, float],
) -> tuple[np.ndarray, int]:
    """Remove source pixels whose discrete reflection lands in a blocker.

    The operation is performed on the source side so the subsequently written
    point-reflected background remains paired one-to-one with the retained
    source aperture.  This is preferable to carving the background alone,
    which would give the two photometric apertures different geometries.
    """
    source_mask = np.asarray(source_mask, dtype=bool)
    blocker_mask = np.asarray(blocker_mask, dtype=bool)
    if source_mask.shape != blocker_mask.shape:
        raise ValueError("source and blocker masks must share one detector grid")
    kept = source_mask.copy()
    yy, xx = np.nonzero(source_mask)
    blocker_y, blocker_x = np.nonzero(blocker_mask)
    if len(blocker_x) == 0:
        return kept, 0
    # Reflect pixel *cells* about the continuous centre. Two unit detector
    # pixels have positive-area overlap exactly when both centre separations
    # are < 1 pixel. Testing the continuous cells (rather than rounded pixel
    # centres) is necessary for a generic subpixel reflection centre.
    xr = 2.0 * centre_xy[0] - xx
    yr = 2.0 * centre_xy[1] - yy
    epsilon = 1.0e-10
    conflicts = np.any(
        (np.abs(xr[:, None] - blocker_x[None, :]) < 1.0 - epsilon)
        & (np.abs(yr[:, None] - blocker_y[None, :]) < 1.0 - epsilon),
        axis=1,
    )
    kept[yy[conflicts], xx[conflicts]] = False
    return kept, int(np.sum(conflicts))


def symmetric_closure(
    mask: np.ndarray, centre_xy: tuple[float, float], valid: np.ndarray
) -> np.ndarray:
    """Close a discrete mask under rounded point reflection.

    For a generic subpixel centre rounded reflection is not exactly involutive:
    multiple input pixels can map to one output pixel.  Iterating the monotone
    union to a fixed point makes the production exclusion mask invariant under
    the exact discrete operator used by source/background construction.
    """
    closed = np.asarray(mask, dtype=bool) & valid
    for _ in range(mask.size + 1):
        updated = (closed | _reflect_mask(closed, centre_xy)) & valid
        if np.array_equal(updated, closed):
            return closed
        closed = updated
    raise RuntimeError("Point-reflection closure did not converge")


def ordered_components(
    components: list[np.ndarray],
    centre_xy: tuple[float, float],
    einstein_radius_pix: float | None,
) -> list[np.ndarray]:
    """Return every clean detector component, largest first."""
    del einstein_radius_pix  # v2 selection is deliberately independent of radius.
    return sorted(
        (mask for mask in components if is_clean(mask, centre_xy)),
        key=lambda mask: -int(mask.sum()),
    )


def largest_connected_piece(mask: np.ndarray) -> np.ndarray:
    labels, count = connected_labels(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = [int(np.sum(labels == index)) for index in range(1, count + 1)]
    return labels == (int(np.argmax(areas)) + 1)


def modeled_satellite_masks(products: Path, system_id: str, source):
    """Build the smallest compact satellite mask retaining 50% of 1-sigma flux."""
    manifest_path = products / f"{system_id}_satellite_light_manifest.json"
    if not manifest_path.is_file():
        return [], {
            "applied": False,
            "reason": "no exported satellite-light manifest",
            "manifest": str(manifest_path),
        }
    manifest = json.loads(manifest_path.read_text())
    masks, current_masks, records = [], [], []
    for item in manifest.get("satellites", []):
        model_path = Path(item["model"])
        model_path = products / (model_path.name if model_path.is_absolute() else model_path)
        full = np.asarray(fits.getdata(model_path), dtype=float)
        model = full[
            source.y0 : source.y0 + source.data.shape[0],
            source.x0 : source.x0 + source.data.shape[1],
        ]
        model = np.nan_to_num(model, nan=0.0)
        centre_x = float(item["fits_x_pixel_zero_indexed"]) - source.x0
        centre_y = float(item["fits_y_pixel_zero_indexed"]) - source.y0

        def centre_component(threshold_sigma: float) -> np.ndarray:
            raw = (
                model >= threshold_sigma * source.sigma
            ) & source.valid
            labels, count = connected_labels(raw)
            row = int(np.clip(round(centre_y), 0, raw.shape[0] - 1))
            col = int(np.clip(round(centre_x), 0, raw.shape[1] - 1))
            component_id = int(labels[row, col])
            if component_id:
                return binary_fill_holes(labels == component_id) & source.valid
            components = [labels == index for index in range(1, count + 1)]
            if not components:
                return np.zeros_like(source.valid, dtype=bool)
            nearest = int(np.argmin([
                np.hypot(
                    np.nonzero(mask)[1].mean() - centre_x,
                    np.nonzero(mask)[0].mean() - centre_y,
                )
                for mask in components
            ]))
            return binary_fill_holes(components[nearest]) & source.valid

        current = centre_component(SATELLITE_MODEL_SNR_THRESHOLD)
        if not current.any():
            records.append({
                "index": int(item["index"]),
                "selected": False,
                "reason": "no pixel exceeds the model-S/N threshold",
            })
            continue
        positive = np.clip(model, 0.0, None)
        current_flux = float(positive[current].sum())
        peak_sigma = float(np.max(model[source.valid]) / source.sigma)
        candidates = []
        for threshold_sigma in np.arange(
            SATELLITE_MODEL_SNR_THRESHOLD,
            peak_sigma + SATELLITE_COMPACT_THRESHOLD_STEP_SIGMA,
            SATELLITE_COMPACT_THRESHOLD_STEP_SIGMA,
        ):
            candidate = centre_component(float(threshold_sigma))
            candidate_flux = float(positive[candidate].sum())
            retained = candidate_flux / max(current_flux, 1.0e-30)
            if (
                int(candidate.sum()) >= MIN_COMPONENT_AREA_PIX
                and retained >= SATELLITE_COMPACT_FLUX_RETENTION
            ):
                candidates.append(
                    (float(threshold_sigma), candidate, candidate_flux, retained)
                )
        if candidates:
            selected_threshold, mask, selected_flux, retained = candidates[-1]
        else:
            selected_threshold = SATELLITE_MODEL_SNR_THRESHOLD
            mask, selected_flux, retained = current, current_flux, 1.0
        if np.any(mask & ~current):
            raise RuntimeError("Compact satellite mask is not inside its 1-sigma mask")
        current_masks.append(current)
        masks.append(mask)
        records.append({
            "index": int(item["index"]),
            "selected": True,
            "centre_crop_x": centre_x,
            "centre_crop_y": centre_y,
            "area_pix": int(mask.sum()),
            "current_1sigma_area_pix": int(current.sum()),
            "area_reduction_pix": int(current.sum() - mask.sum()),
            "selected_threshold_model_snr": selected_threshold,
            "threshold_model_flux": float(
                selected_threshold * source.sigma
            ),
            "current_1sigma_model_flux": current_flux,
            "compact_model_flux": selected_flux,
            "current_1sigma_model_flux_retention": retained,
            "enclosed_positive_model_flux_fraction": float(
                positive[mask].sum() / max(positive[source.valid].sum(), 1.0e-30)
            ),
            "model": str(model_path),
        })
    current_union = np.logical_or.reduce(current_masks) if current_masks else np.zeros_like(
        source.valid, dtype=bool
    )
    compact_union = np.logical_or.reduce(masks) if masks else np.zeros_like(
        source.valid, dtype=bool
    )
    return masks, {
        "applied": bool(masks),
        "manifest": str(manifest_path.resolve()),
        "selection_rule": (
            "highest connected satellite-only model isophote containing or "
            "nearest the fitted centre, with >=5 pixels and retaining >=50% "
            "of the positive model flux inside the original 1-sigma mask"
        ),
        "threshold_model_snr": SATELLITE_MODEL_SNR_THRESHOLD,
        "compact_current_mask_flux_retention": SATELLITE_COMPACT_FLUX_RETENTION,
        "compact_threshold_step_sigma": SATELLITE_COMPACT_THRESHOLD_STEP_SIGMA,
        "satellite_count_in_manifest": int(manifest.get("satellite_count", 0)),
        "selected_satellite_count": len(masks),
        "current_1sigma_union_area_pix": int(current_union.sum()),
        "compact_union_area_pix": int(compact_union.sum()),
        "compact_union_area_reduction_pix": int(
            current_union.sum() - compact_union.sum()
        ),
        "compact_union_area_reduction_fraction": float(
            1.0 - compact_union.sum() / current_union.sum()
        ) if current_union.any() else 0.0,
        "satellites": records,
    }


def refresh_selected_measurements(
    selected: dict,
    mask: np.ndarray,
    source,
    lens_product,
    lens_mask: np.ndarray,
    effective_snr_epsilon: float | None,
) -> None:
    """Recompute every flux/geometry statistic after exclusion-mask subtraction."""
    centre_xy = source.centre_crop_xy
    signal = (source.data - source.background) / source.sigma
    mirror = _reflect_mask(mask, centre_xy) & source.valid
    region_flux = float(np.sum(source.data[mask] - source.background))
    mirror_flux = (
        float(np.sum(source.data[mirror] - source.background))
        * int(mask.sum()) / int(mirror.sum())
        if mirror.any() else 0.0
    )
    lens_flux = float(np.sum(lens_product.data[mask] - lens_product.background))
    difference_flux = region_flux - mirror_flux
    statistical_flux_error = float(
        source.sigma * np.sqrt(int(mask.sum()) + int(mirror.sum()))
    )
    effective_snr = (
        float(
            difference_flux
            / np.hypot(
                statistical_flux_error,
                float(effective_snr_epsilon) * abs(lens_flux),
            )
        )
        if effective_snr_epsilon is not None else 0.0
    )
    row = int(np.clip(round(centre_xy[1]), 0, mask.shape[0] - 1))
    col = int(np.clip(round(centre_xy[0]), 0, mask.shape[1] - 1))
    selected.update({
        "mask": mask,
        "area_pix": int(mask.sum()),
        "hole_pixel_count": int((binary_fill_holes(mask) & ~mask).sum()),
        "contains_lens_centre": bool(mask[row, col]),
        "symmetric_overlap_fraction": symmetric_overlap(mask, centre_xy),
        "integrated_snr": float(
            np.sum(signal[mask]) / np.sqrt(max(int(mask.sum()), 1))
        ),
        "difference_flux": difference_flux,
        "statistical_flux_error": statistical_flux_error,
        "lens_model_flux": lens_flux,
        "effective_snr": effective_snr,
        "region_flux": region_flux,
        "mirror_asymmetry": (
            difference_flux / region_flux if region_flux > 0 else 0.0
        ),
        "lens_model_flux_fraction": (
            abs(lens_flux) / region_flux if region_flux > 0 else 1.0
        ),
        "final_lens_overlap_pix": int(np.sum(mask & lens_mask)),
    })


def grow_level_sigma(detect_sigma: float) -> float | None:
    if GROW_SIGMA_RATIO is None:
        return None
    return max(MIN_ISOPHOTE_SIGMA, GROW_SIGMA_RATIO * float(detect_sigma))


def grow_by_hysteresis(
    seed: np.ndarray,
    signal: np.ndarray,
    valid: np.ndarray,
    lens_mask: np.ndarray,
    centre_xy: tuple[float, float],
    detect_sigma: float,
) -> tuple[np.ndarray, bool]:
    """Extend the seed down to the hysteresis level inside its own component."""
    level = grow_level_sigma(detect_sigma)
    if level is None:
        return seed, False
    faint = (signal >= level) & valid & ~lens_mask
    labels, _ = connected_labels(faint)
    touched = np.unique(labels[seed])
    touched = touched[touched > 0]
    if len(touched) == 0:
        return seed, False
    grown = np.isin(labels, touched) | seed
    if grown.sum() > MAX_GROW_AREA_RATIO * seed.sum():
        return seed, False
    if not is_clean(grown, centre_xy):
        return seed, False
    return grown, True


def dilate_if_clean(
    mask: np.ndarray, centre_xy: tuple[float, float], valid: np.ndarray, lens_mask: np.ndarray
) -> tuple[np.ndarray, bool]:
    if DILATE_PIX <= 0:
        return mask, False
    grown = binary_dilation(mask, iterations=int(DILATE_PIX)) & valid & ~lens_mask
    if not is_clean(grown, centre_xy):
        return mask, False
    return grown, True


# ---------------------------------------------------------------------------
# Trial construction and scoring
# ---------------------------------------------------------------------------

def detect_components(source, settings, sextractor, work_dir, lens_mask):
    segmentation = run_sextractor(source, work_dir, sextractor, settings)
    segmentation = np.asarray(segmentation).copy()
    raw_area = int(np.sum(segmentation > 0))
    removed = int(np.sum((segmentation > 0) & lens_mask))
    segmentation[lens_mask] = 0
    components = []
    for source_label in (int(v) for v in np.unique(segmentation) if v > 0):
        labels, count = connected_labels(segmentation == source_label)
        for index in range(1, count + 1):
            component = labels == index
            if int(component.sum()) >= MIN_COMPONENT_AREA_PIX:
                components.append(component)
    return components, raw_area, removed


def build_trial(source, lens_product, base_settings, sigma, sextractor, work_dir,
                lens_mask, einstein_radius_pix, effective_snr_epsilon=None):
    """One threshold step: detect, assemble, grow, dilate, then measure."""
    settings = deepcopy(base_settings)
    settings["detect_snr_floor"] = float(sigma)
    settings["detect_peak_fraction"] = 0.0

    trial = {
        "threshold_sigma": float(sigma),
        "n_components": 0,
        "n_components_merged": 0,
        "n_isolated_components": 0,
        "raw_detected_area_pix": 0,
        "lens_overlap_removed_pix": 0,
        "final_lens_overlap_pix": 0,
        "other_source_direct_overlap_pix": 0,
        "other_source_mirror_overlap_pix": 0,
        "isolation_fallback_used": False,
        "pixels_carved_for_source_isolation": 0,
        "seed_area_pix": 0,
        "grown": False,
        "dilated": False,
        "area_pix": 0,
        "hole_pixel_count": -1,
        "contains_lens_centre": True,
        "symmetric_overlap_fraction": 1.0,
        "integrated_snr": 0.0,
        "difference_flux": 0.0,
        "statistical_flux_error": 0.0,
        "lens_model_flux": 0.0,
        "effective_snr": 0.0,
        "region_flux": 0.0,
        "mirror_asymmetry": 0.0,
        "lens_model_flux_fraction": 1.0,
        "mask": None,
        "settings": settings,
        "error": "",
    }

    signal = (source.data - source.background) / source.sigma
    centre_xy = source.centre_crop_xy
    try:
        components, raw_area, removed = detect_components(
            source, settings, sextractor, work_dir, lens_mask
        )
        trial["n_components"] = len(components)
        trial["raw_detected_area_pix"] = raw_area
        trial["lens_overlap_removed_pix"] = removed
        if not components:
            trial["error"] = "no component above min area"
            return trial

        seeds = ordered_components(components, centre_xy, einstein_radius_pix)
        if not seeds:
            trial["error"] = "no point-asymmetric component"
            return trial

        # Build each source independently. No components are merged in v2.
        # Keeping the preliminary footprints for every source lets isolation be
        # tested against what the detector actually regards as other sources.
        prepared = []
        for seed in seeds:
            candidate, grown = grow_by_hysteresis(
                seed, signal, source.valid, lens_mask, centre_xy, sigma
            )
            candidate, dilated = dilate_if_clean(
                candidate, centre_xy, source.valid, lens_mask
            )
            filled = binary_fill_holes(candidate) & source.valid
            if (
                not np.any(filled & lens_mask)
                and is_clean(filled, centre_xy)
            ):
                candidate = filled
            # Mandatory final lens subtraction: unlike v11, no later operation
            # is allowed to fill the lens footprint back into the source mask.
            candidate = candidate & source.valid & ~lens_mask
            if candidate.any() and is_clean(candidate, centre_xy):
                prepared.append({
                    "seed": seed,
                    "mask": candidate,
                    "grown": grown,
                    "dilated": dilated,
                })
        if not prepared:
            trial["error"] = "no valid component after growth and lens subtraction"
            return trial

        isolated = []
        for index, item in enumerate(prepared):
            other_union = np.zeros_like(item["mask"], dtype=bool)
            # All raw detector components are blockers, including components
            # that are unsuitable as the final aperture themselves.
            for component in components:
                if component is not item["seed"]:
                    other_union |= component
            # Also include the grown footprints of the other admissible sources.
            for other_index, other in enumerate(prepared):
                if other_index != index:
                    other_union |= other["mask"]
            direct = int(np.sum(item["mask"] & other_union))
            mirror = _reflect_mask(item["mask"], centre_xy)
            mirror_overlap = int(np.sum(mirror & other_union))
            item["other_union"] = other_union
            item["direct_overlap"] = direct
            item["mirror_overlap"] = mirror_overlap
            if direct == 0 and mirror_overlap == 0:
                isolated.append(item)

        trial["n_isolated_components"] = len(isolated)
        if isolated:
            chosen = max(isolated, key=lambda item: int(item["mask"].sum()))
            mask = chosen["mask"]
        else:
            # Required fallback: take the largest source, then remove pixels
            # belonging to another source or whose reflected locations belong
            # to another source. Retain only the largest connected remainder so
            # the final result is still exactly one region.
            chosen = max(prepared, key=lambda item: int(item["mask"].sum()))
            conflict = chosen["other_union"] | _reflect_mask(
                chosen["other_union"], centre_xy
            )
            carved = chosen["mask"] & ~conflict & ~lens_mask
            mask = largest_connected_piece(carved)
            trial["isolation_fallback_used"] = True
            trial["pixels_carved_for_source_isolation"] = int(
                chosen["mask"].sum() - mask.sum()
            )

        if not mask.any():
            trial["error"] = "source-isolation carving removed the full component"
            return trial
        trial["seed_area_pix"] = int(chosen["seed"].sum())
        trial["n_components_merged"] = 1
        trial["grown"] = bool(chosen["grown"])
        trial["dilated"] = bool(chosen["dilated"])
        trial["other_source_direct_overlap_pix"] = int(
            np.sum(mask & chosen["other_union"])
        )
        trial["other_source_mirror_overlap_pix"] = int(
            np.sum(_reflect_mask(mask, centre_xy) & chosen["other_union"])
        )
        trial["final_lens_overlap_pix"] = int(np.sum(mask & lens_mask))

        region_flux = float(np.sum(source.data[mask] - source.background))
        lens_flux = float(np.sum(lens_product.data[mask] - lens_product.background))
        # The mirror carries lens light and background but, by construction, no
        # source. Rescaled by area because reflection can clip at the frame edge.
        mirror = _reflect_mask(mask, centre_xy) & source.valid
        mirror_flux = (
            float(np.sum(source.data[mirror] - source.background))
            * int(mask.sum()) / int(mirror.sum())
            if mirror.any() else 0.0
        )
        difference_flux = region_flux - mirror_flux
        # The detection product carries one robust per-pixel sigma.  Source and
        # reflected control apertures have independent statistical noise to
        # first order, hence sqrt(N_source + N_control).  R1 adds a fractional
        # lens-model leakage floor in quadrature; it changes candidate ranking
        # only and is never propagated into downstream photometry.
        statistical_flux_error = float(
            source.sigma * np.sqrt(int(mask.sum()) + int(mirror.sum()))
        )
        effective_snr = (
            float(
                difference_flux
                / np.hypot(
                    statistical_flux_error,
                    float(effective_snr_epsilon) * abs(lens_flux),
                )
            )
            if effective_snr_epsilon is not None else 0.0
        )
        trial.update({
            "mask": mask,
            "area_pix": int(mask.sum()),
            "hole_pixel_count": int((binary_fill_holes(mask) & ~mask).sum()),
            "contains_lens_centre": not is_clean(mask, centre_xy) and bool(
                mask[
                    int(np.clip(round(centre_xy[1]), 0, mask.shape[0] - 1)),
                    int(np.clip(round(centre_xy[0]), 0, mask.shape[1] - 1)),
                ]
            ),
            "symmetric_overlap_fraction": symmetric_overlap(mask, centre_xy),
            "integrated_snr": float(np.sum(signal[mask]) / np.sqrt(max(int(mask.sum()), 1))),
            "difference_flux": difference_flux,
            "statistical_flux_error": statistical_flux_error,
            "lens_model_flux": lens_flux,
            "effective_snr": effective_snr,
            "region_flux": region_flux,
            "mirror_asymmetry": (
                (region_flux - mirror_flux) / region_flux if region_flux > 0 else 0.0
            ),
            "lens_model_flux_fraction": (
                abs(lens_flux) / region_flux if region_flux > 0 else 1.0
            ),
        })
    except Exception as exc:  # noqa: BLE001 - a failed threshold is a data point
        trial["error"] = repr(exc)
    return trial


def apply_gates(trials: list[dict]) -> None:
    """Fill in the neighbour-flux stability metric and the pass/fail flag."""
    for index, trial in enumerate(trials):
        ratios = []
        if trial["mask"] is not None and trial["region_flux"] > 0:
            for neighbour_index in (index - 1, index + 1):
                if not 0 <= neighbour_index < len(trials):
                    continue
                neighbour = trials[neighbour_index]
                if neighbour["mask"] is None or neighbour["region_flux"] <= 0:
                    continue
                ratios.append(
                    abs(neighbour["region_flux"] - trial["region_flux"]) / trial["region_flux"]
                )
        trial["neighbour_flux_change"] = max(ratios) if ratios else 1.0

    for trial in trials:
        trial["passes_image_gates"] = bool(
            trial["mask"] is not None
            and trial["threshold_sigma"] >= MIN_ISOPHOTE_SIGMA
            and trial["area_pix"] >= MIN_AREA_PIX
            and trial["hole_pixel_count"] == 0
            and trial["final_lens_overlap_pix"] == 0
            and trial["other_source_direct_overlap_pix"] == 0
            and trial["other_source_mirror_overlap_pix"] == 0
            and not trial["contains_lens_centre"]
            and trial["symmetric_overlap_fraction"] <= MAX_SYMMETRIC_OVERLAP
            and trial["integrated_snr"] >= MIN_INTEGRATED_SNR
            and trial["mirror_asymmetry"] >= MIN_MIRROR_ASYMMETRY
            and trial["neighbour_flux_change"] <= MAX_NEIGHBOUR_FLUX_CHANGE
        )


def adapt_region(source, lens_product, base_settings, sextractor, work_root,
                 lens_mask, einstein_radius_pix, effective_snr_epsilon=None):
    thresholds = np.round(
        np.arange(
            THRESHOLD_START_SIGMA,
            THRESHOLD_STOP_SIGMA - 0.5 * THRESHOLD_STEP_SIGMA,
            -THRESHOLD_STEP_SIGMA,
        ),
        3,
    )
    trials = [
        build_trial(
            source, lens_product, base_settings, sigma, sextractor,
            work_root / f"sigma_{sigma:.2f}", lens_mask, einstein_radius_pix,
            effective_snr_epsilon,
        )
        for sigma in thresholds
    ]
    apply_gates(trials)

    accepted = [trial for trial in trials if trial["passes_image_gates"]]
    if not accepted:
        diagnostic_path = work_root / "failed_trials.csv"
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        write_trace(diagnostic_path, trials)
        raise RuntimeError(
            "No SExtractor trial passed all image-only gates; "
            f"diagnostics written to {diagnostic_path}"
        )
    # Among the admissible isophotes, keep the one maximising the integrated
    # signal-to-noise of the region. Unlike "brightest passing" (v1) or
    # "faintest passing", this criterion has an interior optimum: lowering the
    # threshold adds area faster than it adds signal once the boundary drops
    # below roughly half the mean surface brightness, so the statistic turns
    # over on its own instead of running to whichever end the scan stops at.
    ranking_field = (
        "effective_snr" if effective_snr_epsilon is not None else "integrated_snr"
    )
    peak_snr = max(trial[ranking_field] for trial in accepted)
    # The strict optimum is conservative: a hand-drawn region is deliberately
    # generous and scores slightly below the achievable maximum. Accepting a
    # small, explicit loss of integrated signal-to-noise buys back the faint
    # outskirts, so take the faintest trial still within the tolerance band.
    within = [
        trial for trial in accepted
        if trial[ranking_field] >= (1.0 - SNR_TOLERANCE_FRACTION) * peak_snr
    ]
    best = within[-1]
    best_index = trials.index(best)
    # One universal one-step safety margin towards the brighter side, so a
    # single boundary pixel cannot flip a nominally clean region. Applied to
    # every system and method, never using reference-region metrics.
    selected = trials[max(best_index - 1, 0)]
    if not selected["passes_image_gates"]:
        selected = best
    selected["optimal_snr_sigma"] = best["threshold_sigma"]
    selected["faintest_passing_sigma"] = accepted[-1]["threshold_sigma"]
    selected["safety_margin_steps"] = int(selected is not best)
    selected["ranking_field"] = ranking_field
    selected["effective_snr_epsilon"] = effective_snr_epsilon
    return selected, trials


TRACE_FIELDS = [
    "method", "threshold_sigma", "n_components", "n_components_merged",
    "n_isolated_components", "raw_detected_area_pix", "lens_overlap_removed_pix",
    "final_lens_overlap_pix", "other_source_direct_overlap_pix",
    "other_source_mirror_overlap_pix", "isolation_fallback_used",
    "pixels_carved_for_source_isolation", "seed_area_pix",
    "grown", "dilated", "area_pix", "hole_pixel_count", "contains_lens_centre",
    "symmetric_overlap_fraction", "integrated_snr", "difference_flux",
    "statistical_flux_error", "lens_model_flux", "effective_snr", "region_flux", "mirror_asymmetry",
    "lens_model_flux_fraction", "neighbour_flux_change", "passes_image_gates", "error",
]


def write_trace(path: Path, trials: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        for trial in trials:
            writer.writerow(
                {key: "sextractor" if key == "method" else trial.get(key) for key in TRACE_FIELDS}
            )


def write_lens_trace(path: Path, trials: list[dict]) -> None:
    preferred = [
        "method", "layer_number_from_centre", "sigma_arcsec",
        "cumulative_gaussian_count", "enclosed_flux_fraction", "aperture_threshold",
        "equivalent_radius_pix", "source_overlap_pix",
        "layer_factor", "threshold_model_snr", "area_pix", "contains_source_peak",
        "hole_pixel_count", "model_flux_fraction", "admissible",
    ]
    fields = [key for key in preferred if any(key in row for row in trials)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in trials)


def save_lens_method_comparison(path: Path, lens_product, source, methods: dict,
                                manual_source, manual_peak_yx, anchor_yx) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].imshow(lens_product.data, origin="lower", cmap="gray", norm=_display_norm(lens_product))
    axes[0].plot(lens_product.centre_crop_xy[0], lens_product.centre_crop_xy[1],
                 "+", color="cyan", ms=13, mew=2)
    axes[0].plot(anchor_yx[1], anchor_yx[0], "*", color="red", ms=11)
    axes[0].set_title("summed MGE lens-light model")
    for ax, method in zip(axes[1:], ("sextractor", "mge_isophote"), strict=True):
        item = methods[method]
        ax.imshow(source.data, origin="lower", cmap="gray", norm=_display_norm(source))
        ax.contour(item["mask"], levels=[0.5], colors="magenta", linewidths=2, origin="lower")
        if manual_source is not None:
            ax.contour(manual_source, levels=[0.5], colors="lime", linewidths=1.5,
                       linestyles="--", origin="lower")
        ax.plot(anchor_yx[1], anchor_yx[0], "*", color="red", ms=11)
        if manual_peak_yx is not None and manual_peak_yx != anchor_yx:
            ax.plot(manual_peak_yx[1], manual_peak_yx[0], "x", color="yellow", ms=9)
        meta = item["metadata"]
        evaluation = item["evaluation"]
        detail = (
            f"layer={meta['layer_number_from_centre']}, "
            f"sigma={meta['sigma_arcsec']:.3f} arcsec"
            if "layer_number_from_centre" in meta else
            (
                f"flux={meta['enclosed_flux_fraction']:.3f}, "
                f"R_eq={meta['equivalent_radius_pix']:.2f} pix"
                if "equivalent_radius_pix" in meta else
                f"threshold factor={meta['layer_factor']:.3f}"
            )
        )
        ax.set_title(
            f"{method}: area={meta['area_pix']}\n{detail}, ref overlap="
            f"{evaluation.get('manual_source_overlap_fraction', float('nan')):.3f}"
        )
    for ax in axes:
        ax.set_xlabel("cropped x pixel")
        ax.set_ylabel("cropped y pixel")
    fig.suptitle(
        "Magenta: automatic lens; green dashed: manual source (evaluation only); "
        "red star: image-only protected source peak"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def einstein_radius_pixels(system_dir: Path, source) -> float | None:
    config_path = system_dir / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    radius_arcsec = config.get("einstein_radius_mean")
    if radius_arcsec is None:
        return None
    scale = float(np.sqrt(np.abs(np.linalg.det(source.wcs.pixel_scale_matrix))) * 3600.0)
    return float(radius_arcsec) / scale


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Stable SExtractor region selection, v2")
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--sextractor", default="source-extractor")
    parser.add_argument(
        "--products-dir",
        type=Path,
        default=None,
        help="Stage-specific WCS-products directory; defaults to stage-3 wcs_products.",
    )
    parser.add_argument(
        "--lens-method",
        choices=("sextractor", "mge_isophote", "mge_fixed_flux90"),
        default=DEFAULT_METHOD,
        help="Wide lens-aperture method used for the final source extraction.",
    )
    parser.add_argument(
        "--lens-geometry",
        choices=("legacy", "symmetric"),
        default="symmetric",
        help="Use L directly or enforce L_sym = L union its point reflection.",
    )
    parser.add_argument(
        "--lens-scale",
        type=float,
        default=0.85,
        help=(
            "CFHT radial scale about the continuous lens centre, applied before "
            "symmetry (production default: 0.85)."
        ),
    )
    parser.add_argument(
        "--effective-snr-epsilon",
        type=float,
        default=None,
        help=(
            "Enable R1 candidate ranking with the fractional lens-model leakage "
            f"floor (recommended v31 value: {DEFAULT_EFFECTIVE_SNR_EPSILON})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Versioned output directory; defaults to outputs/system_ID.",
    )
    parser.add_argument(
        "--pair-guard-pix",
        type=int,
        default=0,
        help=(
            "Image-only safety gap between a source mask and its point reflection. "
            "Zero preserves stable-v2; positive values carve only source pixels "
            "within this many detector pixels of the reflected aperture."
        ),
    )
    parser.add_argument(
        "--background-centre",
        choices=("positions-gui", "model"),
        default="positions-gui",
        help=(
            "Centre used only for the final point-reflected background aperture. "
            "Source/lens selection always remains tied to the WCS-product model centre."
        ),
    )
    parser.add_argument(
        "--positions-file",
        type=Path,
        default=None,
        help="positions_gui.txt used when --background-centre=positions-gui.",
    )
    parser.add_argument(
        "--manual-source-region",
        type=Path,
        default=None,
        help=(
            "Optional manual source region used only for post-selection metrics "
            "and report overlays; it never enters parameter selection."
        ),
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the large transient SExtractor parameter-search products.",
    )
    args = parser.parse_args()
    if args.pair_guard_pix < 0:
        parser.error("--pair-guard-pix must be non-negative")
    if args.effective_snr_epsilon is not None and args.effective_snr_epsilon <= 0:
        parser.error("--effective-snr-epsilon must be positive")

    system_id = args.system
    base_config = json.loads(
        (MODELING_ROOT / "region_selection/config_adaptive_v1_base.json").read_text()
    )
    config = {
        "selection": deepcopy(base_config["selection"]),
        "sextractor": deepcopy(base_config["sextractor"]),
    }
    config["selection"]["analysis_half_size_pix"] = ANALYSIS_HALF_SIZE_PIX
    products = (
        args.products_dir.resolve()
        if args.products_dir is not None
        else MODELING_ROOT / f"{system_id}/output/wcs_products"
    )
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else here / "outputs" / f"{system_id}"
    )
    output.mkdir(parents=True, exist_ok=True)

    centre = json.loads((products / f"{system_id}_lens_light_centre_wcs.json").read_text())
    source_path = products / f"{system_id}_autolens_lens_light_subtracted_wcs.fits"
    if not source_path.is_file():
        raise FileNotFoundError(
            "AutoLens lens-plane-subtracted WCS product is required for source "
            f"selection: {source_path}"
        )
    lens_path = products / f"{system_id}_lens_light_model_wcs.fits"
    for exact_path in (source_path, lens_path):
        header = fits.getheader(exact_path)
        if not bool(header.get("FITEXACT", False)) or header.get("MAPSCHEM") != "FITNATIVE1":
            raise RuntimeError(
                "Region selection refuses a re-rendered or stale model product; "
                f"expected strict FitImaging native export: {exact_path}"
            )
    source = load_product(source_path, centre, ANALYSIS_HALF_SIZE_PIX)
    lens_product = load_product(
        lens_path, centre, ANALYSIS_HALF_SIZE_PIX
    )
    if lens_product.data.shape != source.data.shape:
        raise ValueError("Lens and source WCS products do not share the same analysis grid")

    if args.background_centre == "positions-gui":
        positions_path = (
            args.positions_file.resolve()
            if args.positions_file is not None
            else SCRIPT_DIR.parent
            / "data"
            / "bb"
            / system_id
            / "positions_gui.txt"
        )
        background_centre_xy, background_centre_metadata = (
            positions_gui_centre_on_product(positions_path, source)
        )
    else:
        background_centre_xy = source.centre_crop_xy
        background_centre_metadata = {
            "method": "latest_modeled_lens_light_wcs_centre"
        }

    sextractor = find_sextractor(args.sextractor)
    radius_pix = einstein_radius_pixels(MODELING_ROOT / f"{system_id}", source)
    # Pass 1 deliberately uses the old conservative lens core only to locate an
    # automatic source candidate. Its brightest residual pixel is the protected
    # image-only anchor for both wide-lens methods; no manual region enters here.
    preliminary_lens_segmentation = run_sextractor(
        lens_product,
        output / "parameter_search/sextractor/lens_preliminary",
        sextractor,
        config["sextractor"]["lens"],
    )
    preliminary_lens_mask = select_central_lens_mask(
        preliminary_lens_segmentation, lens_product
    )
    preliminary_source, preliminary_trials = adapt_region(
        source,
        lens_product,
        config["sextractor"]["source"],
        sextractor,
        output / "parameter_search/source_preliminary",
        preliminary_lens_mask,
        radius_pix,
        args.effective_snr_epsilon,
    )
    anchor_y, anchor_x, anchor_metadata = source_peak_anchor(
        source, preliminary_source["mask"]
    )
    lens_methods = build_lens_masks(
        lens_product,
        (anchor_y, anchor_x),
        sextractor,
        config["sextractor"]["lens"],
        output / "parameter_search/lens_wide",
        products / f"{system_id}_lens_mge_components_wcs.fits",
    )
    lens_mask_raw = lens_methods[args.lens_method]["mask"]
    lens_mask_scaled = (
        scale_mask_about_centre(
            lens_mask_raw, source.centre_crop_xy, args.lens_scale
        )
        & source.valid
    )
    if args.lens_geometry == "symmetric":
        lens_mask = symmetric_closure(
            lens_mask_scaled, source.centre_crop_xy, source.valid
        )
    else:
        lens_mask = lens_mask_scaled
    reflected_lens_mask = _reflect_mask(lens_mask, source.centre_crop_xy) & source.valid
    lens_symmetry_mismatch_pix = int(np.sum(reflected_lens_mask & ~lens_mask))
    if args.lens_geometry == "symmetric" and lens_symmetry_mismatch_pix != 0:
        raise RuntimeError(
            "Symmetric lens aperture is not closed under point reflection"
        )

    satellite_masks, satellite_metadata = modeled_satellite_masks(
        products, system_id, source
    )
    satellite_union_mask = np.zeros_like(lens_mask, dtype=bool)
    for satellite_mask in satellite_masks:
        satellite_union_mask |= satellite_mask
    if satellite_masks:
        write_ds9_regions(
            output / f"{system_id}_satellites_v2.reg",
            satellite_masks,
            source,
            [f"satellite_{index}_v2" for index in range(len(satellite_masks))],
            ["red"] * len(satellite_masks),
        )
        write_ds9_regions(
            output / f"{system_id}_satellite_union_v2.reg",
            [satellite_union_mask],
            source,
            ["satellite_union_v2"],
            ["red"],
        )

    # Pass 2 is the production source extraction after subtracting the selected
    # wide lens aperture. This is essential: merely replacing the lens .reg
    # would leave a source aperture inconsistent with it.
    selected, trials = adapt_region(
        source,
        lens_product,
        config["sextractor"]["source"],
        sextractor,
        output / f"parameter_search/source_final_{args.lens_method}",
        lens_mask,
        radius_pix,
        args.effective_snr_epsilon,
    )
    config["sextractor"]["source"] = selected["settings"]
    source_mask_before_satellite_exclusion = np.asarray(
        selected["mask"], dtype=bool
    ).copy()
    selected_mask = source_mask_before_satellite_exclusion & ~satellite_union_mask
    satellite_source_direct_overlap_pix = int(
        np.sum(source_mask_before_satellite_exclusion & satellite_union_mask)
    )
    source_area_after_direct_satellite_exclusion = int(selected_mask.sum())
    selected_mask, satellite_background_paired_removed_pix = (
        remove_pixels_reflecting_into_blocker(
            selected_mask,
            satellite_union_mask,
            background_centre_xy,
        )
    )
    if not selected_mask.any():
        raise RuntimeError(
            "Modeled-satellite exclusion removed the entire automatic source aperture"
        )
    refresh_selected_measurements(
        selected,
        selected_mask,
        source,
        lens_product,
        lens_mask,
        args.effective_snr_epsilon,
    )
    pair_guard_removed_pix = 0
    if args.pair_guard_pix:
        reflected_before_guard = (
            _reflect_mask(selected_mask, source.centre_crop_xy) & source.valid
        )
        reflected_guard = binary_dilation(
            reflected_before_guard,
            structure=np.ones((3, 3), dtype=bool),
            iterations=args.pair_guard_pix,
        )
        guarded_mask = selected_mask & ~reflected_guard & source.valid & ~lens_mask
        pair_guard_removed_pix = int(selected_mask.sum() - guarded_mask.sum())
        if not guarded_mask.any():
            raise RuntimeError("Pair-separation guard removed the entire source aperture")
        selected_mask = guarded_mask

        signal = (source.data - source.background) / source.sigma
        region_flux = float(np.sum(source.data[selected_mask] - source.background))
        lens_flux = float(
            np.sum(lens_product.data[selected_mask] - lens_product.background)
        )
        reflected = _reflect_mask(selected_mask, source.centre_crop_xy) & source.valid
        mirror_flux = (
            float(np.sum(source.data[reflected] - source.background))
            * int(selected_mask.sum()) / int(reflected.sum())
            if reflected.any() else 0.0
        )
        selected.update({
            "mask": selected_mask,
            "area_pix": int(selected_mask.sum()),
            "hole_pixel_count": int(
                (binary_fill_holes(selected_mask) & ~selected_mask).sum()
            ),
            "contains_lens_centre": False,
            "symmetric_overlap_fraction": symmetric_overlap(
                selected_mask, source.centre_crop_xy
            ),
            "integrated_snr": float(
                np.sum(signal[selected_mask])
                / np.sqrt(max(int(selected_mask.sum()), 1))
            ),
            "region_flux": region_flux,
            "mirror_asymmetry": (
                (region_flux - mirror_flux) / region_flux if region_flux > 0 else 0.0
            ),
            "lens_model_flux_fraction": (
                abs(lens_flux) / region_flux if region_flux > 0 else 1.0
            ),
            "final_lens_overlap_pix": int(np.sum(selected_mask & lens_mask)),
        })
    refresh_selected_measurements(
        selected,
        selected_mask,
        source,
        lens_product,
        lens_mask,
        args.effective_snr_epsilon,
    )
    satellite_labels, satellite_remaining_component_count = connected_labels(
        selected_mask
    )
    del satellite_labels
    satellite_metadata.update({
        "union_area_pix": int(satellite_union_mask.sum()),
        "source_area_before_exclusion_pix": int(
            source_mask_before_satellite_exclusion.sum()
        ),
        "source_satellite_overlap_removed_pix": satellite_source_direct_overlap_pix,
        "source_area_after_direct_satellite_exclusion_pix": (
            source_area_after_direct_satellite_exclusion
        ),
        "source_paired_for_background_satellite_removed_pix": (
            satellite_background_paired_removed_pix
        ),
        "source_area_after_exclusion_pix": int(selected_mask.sum()),
        "source_connected_components_after_exclusion": int(
            satellite_remaining_component_count
        ),
        "satellite_region_file": (
            f"{system_id}_satellites_v2.reg" if satellite_masks else None
        ),
        "satellite_union_region_file": (
            f"{system_id}_satellite_union_v2.reg" if satellite_masks else None
        ),
    })
    selected_metadata = {
        key: value for key, value in selected.items()
        if key not in {"mask", "settings"}
    }
    write_trace(output / f"{system_id}_sextractor_v2_trace.csv", trials)
    write_trace(
        output / f"{system_id}_source_preliminary_trace.csv", preliminary_trials
    )
    for method, item in lens_methods.items():
        write_lens_trace(
            output / f"{system_id}_lens_{method}_trace.csv", item["trials"]
        )

    selected_config_path = output / f"{system_id}_v2_selected_config.json"
    selected_config_path.write_text(json.dumps(config, indent=2) + "\n")

    adaptation = {
        "version": "stable_v2",
        "system": system_id,
        "manual_region_used_during_parameter_selection": False,
        "lens_selection": {
            "production_method": args.lens_method,
            "available_methods": list(lens_methods),
            "source_anchor": anchor_metadata,
            "manual_source_region_used": False,
            "sextractor": lens_methods["sextractor"]["metadata"],
            "mge_isophote": lens_methods["mge_isophote"]["metadata"],
            "mge_fixed_flux90": lens_methods["mge_fixed_flux90"]["metadata"],
            "geometry": args.lens_geometry,
            "radial_scale": float(args.lens_scale),
            "raw_area_pix": int(lens_mask_raw.sum()),
            "scaled_area_pix": int(lens_mask_scaled.sum()),
            "production_area_pix": int(lens_mask.sum()),
            "point_reflection_mismatch_pix": lens_symmetry_mismatch_pix,
        },
        "satellite_exclusion": satellite_metadata,
        "threshold_scan_sigma": {
            "start": THRESHOLD_START_SIGMA,
            "stop": THRESHOLD_STOP_SIGMA,
            "step": THRESHOLD_STEP_SIGMA,
            "direction": "descending; the faintest passing isophote is kept",
        },
        "assembly": {
            "merge_components": False,
            "selection_rule": (
                "largest single component whose source and reflected footprint "
                "avoid all other source components; otherwise carve conflicts "
                "from the largest component"
            ),
            "einstein_radius_pix": radius_pix,
            "grow_sigma_ratio": GROW_SIGMA_RATIO,
            "max_grow_area_ratio": MAX_GROW_AREA_RATIO,
            "dilate_pix": DILATE_PIX,
            "pair_guard_pix": int(args.pair_guard_pix),
            "pair_guard_removed_pix": pair_guard_removed_pix,
            "source_candidate_ranking": selected.get("ranking_field", "integrated_snr"),
            "effective_snr_epsilon": args.effective_snr_epsilon,
        },
        "fixed_image_only_gates": {
            "minimum_isophote_sigma": MIN_ISOPHOTE_SIGMA,
            "minimum_integrated_snr": MIN_INTEGRATED_SNR,
            "minimum_mirror_asymmetry": MIN_MIRROR_ASYMMETRY,
            "maximum_symmetric_overlap": MAX_SYMMETRIC_OVERLAP,
            "maximum_neighbour_flux_change": MAX_NEIGHBOUR_FLUX_CHANGE,
            "minimum_area_pix": MIN_AREA_PIX,
            "maximum_hole_pixel_count": 0,
            "contains_lens_centre": False,
            "maximum_final_lens_overlap_pix": 0,
            "maximum_other_source_direct_overlap_pix": 0,
            "maximum_other_source_mirror_overlap_pix": 0,
        },
        "selected": {"sextractor": selected_metadata},
    }
    (output / f"{system_id}_v2_adaptation.json").write_text(
        json.dumps(adaptation, indent=2) + "\n"
    )

    # An optional manual region enters only here, after configuration is frozen.
    # Production Photo-lens runs omit it completely.
    reference = (
        parse_ds9_polygon(args.manual_source_region.resolve(), source)
        if args.manual_source_region is not None
        else None
    )
    if reference is not None:
        reference = reference & source.valid

    source_signal = (source.data - source.background) / source.sigma
    manual_peak_yx = None
    if reference is not None and reference.any():
        manual_peak_yx = tuple(int(value) for value in np.unravel_index(
            np.argmax(np.where(reference, source_signal, -np.inf)), source_signal.shape
        ))
    for method, item in lens_methods.items():
        item["evaluation"] = evaluate_lens_mask(
            item["mask"], reference, manual_peak_yx
        )
        write_ds9_regions(
            output / f"{system_id}_lens_{method}_v2.reg",
            [item["mask"]],
            lens_product,
            [f"lens_{method}_v2"],
            ["magenta"],
        )
    save_lens_method_comparison(
        output / f"{system_id}_lens_methods_v2_comparison.png",
        lens_product,
        source,
        lens_methods,
        reference,
        manual_peak_yx,
        (anchor_y, anchor_x),
    )

    metrics = {
        "threshold_sigma": selected_metadata["threshold_sigma"],
        "area_pix": int(selected_mask.sum()),
        "integrated_snr": selected_metadata["integrated_snr"],
        "effective_snr": selected_metadata["effective_snr"],
        "effective_snr_epsilon": selected_metadata["effective_snr_epsilon"],
        "mirror_asymmetry": selected_metadata["mirror_asymmetry"],
        "lens_model_flux_fraction": selected_metadata["lens_model_flux_fraction"],
        "symmetric_overlap_fraction": selected_metadata["symmetric_overlap_fraction"],
        "hole_pixel_count": selected_metadata["hole_pixel_count"],
        "final_lens_overlap_pix": selected_metadata["final_lens_overlap_pix"],
        "other_source_direct_overlap_pix": selected_metadata["other_source_direct_overlap_pix"],
        "other_source_mirror_overlap_pix": selected_metadata["other_source_mirror_overlap_pix"],
        "isolation_fallback_used": selected_metadata["isolation_fallback_used"],
        "pixels_carved_for_source_isolation": selected_metadata["pixels_carved_for_source_isolation"],
        **_reference_metrics(selected_mask, reference),
    }
    metrics["passes_all_hard_gates"] = bool(
        metrics["area_pix"] >= MIN_AREA_PIX
        and metrics["integrated_snr"] >= MIN_INTEGRATED_SNR
        and metrics["symmetric_overlap_fraction"] <= MAX_SYMMETRIC_OVERLAP
        and metrics["hole_pixel_count"] == 0
        and metrics["final_lens_overlap_pix"] == 0
        and metrics["other_source_direct_overlap_pix"] == 0
        and metrics["other_source_mirror_overlap_pix"] == 0
        and metrics["mirror_asymmetry"] >= MIN_MIRROR_ASYMMETRY
    )
    summary = {
        "system": system_id,
        "version": "stable_v2",
        "wcs_products_dir": str(products),
        "source_detection_product": str(source_path),
        "source_detection_definition": (
            "literal maximum-likelihood FitImaging "
            "subtracted_images_of_planes_list[-1], placed back on its original "
            "detector pixels with strict array-equality validation"
        ),
        "method": "sextractor",
        "reference_was_used_for_parameter_selection": False,
        "reference_was_used_only_for_post_selection_evaluation": bool(
            args.manual_source_region is not None
        ),
        "reference_area_pix": int(reference.sum()) if reference is not None else None,
        "lens_method": args.lens_method,
        "lens_geometry": args.lens_geometry,
        "lens_scale": float(args.lens_scale),
        "pair_guard_pix": int(args.pair_guard_pix),
        "pair_guard_removed_pix": pair_guard_removed_pix,
        "source_candidate_ranking": selected_metadata["ranking_field"],
        "effective_snr_epsilon": args.effective_snr_epsilon,
        "lens_area_pix": int(lens_mask.sum()),
        "lens_point_reflection_mismatch_pix": lens_symmetry_mismatch_pix,
        "satellite_exclusion": satellite_metadata,
        "lens_methods": {
            method: {
                **item["metadata"],
                "post_selection_manual_evaluation": item["evaluation"],
            }
            for method, item in lens_methods.items()
        },
        "automatic_source_peak_matches_manual_source_peak": bool(
            manual_peak_yx == (anchor_y, anchor_x)
        ) if manual_peak_yx is not None else None,
        "selected": {"sextractor": metrics},
    }
    write_ds9_regions(
        output / f"{system_id}_source_v2.reg",
        [selected_mask],
        source,
        ["source_v2"],
    )
    # Source/lens selection above deliberately uses the latest model centre.
    # Only the final differential background is reflected about the independent
    # antisymmetry centre saved by positions_gui, as requested for production.
    # Build the diagnostic reflection on the complete detector, not on the
    # finite/modeling-mask footprint of the WCS product.  The DS9 writer below
    # has always reflected the continuous contour in full-image coordinates;
    # matching the QA calculation to that geometry prevents a valid background
    # aperture outside the modeling mask from being truncated or rejected.
    full_shape = (
        int(source.header.get("NAXIS2", source.y0 + source.data.shape[0])),
        int(source.header.get("NAXIS1", source.x0 + source.data.shape[1])),
    )
    crop_y = slice(source.y0, source.y0 + source.data.shape[0])
    crop_x = slice(source.x0, source.x0 + source.data.shape[1])
    source_mask_full = np.zeros(full_shape, dtype=bool)
    source_mask_full[crop_y, crop_x] = selected_mask
    valid_model_full = np.zeros(full_shape, dtype=bool)
    valid_model_full[crop_y, crop_x] = source.valid
    lens_mask_full = np.zeros(full_shape, dtype=bool)
    lens_mask_full[crop_y, crop_x] = lens_mask
    satellite_mask_full = np.zeros(full_shape, dtype=bool)
    satellite_mask_full[crop_y, crop_x] = satellite_union_mask
    background_centre_full_xy = (
        background_centre_xy[0] + source.x0,
        background_centre_xy[1] + source.y0,
    )
    background_mask_full = _reflect_mask(
        source_mask_full, background_centre_full_xy
    )
    background_mask = background_mask_full[crop_y, crop_x]
    if not background_mask_full.any():
        raise RuntimeError("Point-reflected automatic background aperture is empty")
    if np.any(background_mask_full & source_mask_full):
        raise RuntimeError("Automatic source and point-reflected background overlap")
    background_lens_overlap_pix = int(
        np.sum(background_mask_full & lens_mask_full)
    )
    background_satellite_overlap_pix = int(
        np.sum(background_mask_full & satellite_mask_full)
    )
    if background_satellite_overlap_pix:
        raise RuntimeError(
            "Paired satellite exclusion failed: point-reflected background "
            f"still overlaps satellite mask by {background_satellite_overlap_pix} pixels"
        )
    background_outside_model_mask_pix = int(
        np.sum(background_mask_full & ~valid_model_full)
    )
    background_outside_analysis_crop_pix = int(
        background_mask_full.sum() - background_mask.sum()
    )
    if background_lens_overlap_pix:
        print(
            "WARNING: positions_gui-reflected background overlaps the model-centred "
            f"lens mask by {background_lens_overlap_pix} diagnostic crop pixels; "
            "preserving the requested direct point reflection."
        )
    discrete_paired_area_difference_pix = int(
        background_mask_full.sum() - selected_mask.sum()
    )
    write_point_reflected_ds9_region(
        output / f"{system_id}_background_v2.reg",
        selected_mask,
        source,
        "background_v2_point_reflection",
        "cyan",
        centre_full_xy=background_centre_full_xy,
    )
    write_ds9_regions(
        output / f"{system_id}_lens_v2.reg",
        [lens_mask],
        lens_product,
        [f"lens_v2_{args.lens_method}"],
    )
    summary["background_region"] = {
        "definition": (
            "180-degree point reflection of the final automatic source mask "
            f"about the {args.background_centre} lens-light centre"
        ),
        "source_selection_centre": "latest_modeled_lens_light_wcs_centre",
        "reflection_centre_source": args.background_centre,
        "reflection_centre_metadata": background_centre_metadata,
        "centre_crop_x": float(background_centre_xy[0]),
        "centre_crop_y": float(background_centre_xy[1]),
        "model_to_background_centre_distance_pix": float(np.hypot(
            background_centre_xy[0] - source.centre_crop_xy[0],
            background_centre_xy[1] - source.centre_crop_xy[1],
        )),
        "source_area_pix": int(selected_mask.sum()),
        "background_area_pix_full_detector": int(background_mask_full.sum()),
        "background_area_pix_in_analysis_crop": int(background_mask.sum()),
        "discrete_paired_area_difference_pix": discrete_paired_area_difference_pix,
        "area_acceptance_definition": (
            "continuous DS9 aperture weights at 81x81 and 161x161; integer-pixel "
            "areas are diagnostic because rounded reflection is non-bijective for "
            "a generic subpixel centre"
        ),
        "background_pixels_outside_analysis_crop": background_outside_analysis_crop_pix,
        "background_pixels_outside_model_mask": background_outside_model_mask_pix,
        "final_ds9_region_clipped_by_model_mask": False,
        "geometry_note": (
            "QA reflection is evaluated on the full detector; the final DS9 "
            "polygon is a continuous full-image reflection and may extend beyond "
            "the modeling mask without truncation"
        ),
        "source_background_overlap_pix": int(
            np.sum(source_mask_full & background_mask_full)
        ),
        "background_lens_overlap_pix": background_lens_overlap_pix,
        "background_satellite_overlap_pix": background_satellite_overlap_pix,
        "region_file": f"{system_id}_background_v2.reg",
    }
    (output / f"{system_id}_v2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    if not args.keep_intermediates:
        shutil.rmtree(output / "parameter_search", ignore_errors=True)

    recall = metrics.get("reference_recall")
    precision = metrics.get("reference_precision")
    print(
        f"{system_id} sextractor_v2: sigma={metrics['threshold_sigma']:.2f} "
        f"area={metrics['area_pix']} "
        f"recall={recall:.3f} precision={precision:.3f} "
        f"intSNR={metrics['integrated_snr']:.1f} "
        f"asym={metrics['mirror_asymmetry']:.3f}"
        if recall is not None else
        f"{system_id} sextractor_v2: sigma={metrics['threshold_sigma']:.2f} "
        f"area={metrics['area_pix']}"
    )


if __name__ == "__main__":
    main()
