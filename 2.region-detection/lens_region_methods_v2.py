#!/usr/bin/env python3
"""Wide, image-only lens apertures for automatic region selection.

The source reference polygons are deliberately not inputs to either method.
Both methods stop at an image-derived anchor: the brightest positive pixel in
the preliminary source aperture on the lens-light-reduced image.  The manual
source polygon is accepted only by :func:`evaluate_lens_mask` after selection.

``sextractor`` scans a lens-specific, non-deblended detection threshold and
keeps the largest central segmentation that does not contain the source anchor.
``mge_isophote`` scans nested positive-flux envelopes of the summed,
PSF-convolved 30-Gaussian lens model and selects the envelope whose
equivalent-area radius matches the fitted Einstein radius.  The source anchor
is only a hard no-cover guard, not the scale-setting quantity.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from region_selection.pipeline import run_sextractor, select_central_lens_mask


METHOD_NAMES = ("sextractor", "mge_isophote", "mge_fixed_flux90")
DEFAULT_METHOD = "mge_isophote"

# SExtractor threshold scan settings.
LAYER_FACTOR_MIN = 0.50
LAYER_FACTOR_MAX = 3.00
LAYER_FACTOR_STEP = 0.025
# Legacy fitted-component-layer diagnostics retained in the trace and metadata.
SOURCE_PEAK_SAFETY_FACTOR = 1.025
MGE_INWARD_LAYER_STEPS = 2
MIN_MODEL_ISOPHOTE_SNR = 0.50
# Each cumulative radial MGE model is converted to a finite aperture using the
# smallest central isophote containing this fraction of its positive flux.
MGE_ENCLOSED_FLUX_FRACTION = 0.90
MGE_ENVELOPE_FLUX_START = 0.30
MGE_ENVELOPE_FLUX_STOP = 0.99
MGE_ENVELOPE_FLUX_STEP = 0.005


def as_anchor_list(anchors) -> list[tuple[int, int]]:
    """Normalise a single ``(row, col)`` anchor or a sequence of them.

    A bare anchor is a 2-tuple of integers; an anchor list is a sequence of
    such tuples.  Accepting both keeps every existing single-anchor call site
    valid while the lens layers are protected against all source components.
    """
    if anchors is None:
        return []
    items = list(anchors)
    if not items:
        return []
    if len(items) == 2 and all(np.isscalar(value) for value in items):
        return [(int(items[0]), int(items[1]))]
    return [(int(row), int(col)) for row, col in items]


def anchor_min_distance(mask: np.ndarray, anchors: list[tuple[int, int]],
                        post_process=None) -> float:
    """Euclidean distance (pixel centres) from the nearest anchor to the
    production aperture; 0 when an anchor is covered, inf without anchors."""
    if post_process is not None:
        mask = post_process(mask)
    if not anchors:
        return float("inf")
    if not mask.any():
        return float("inf")
    distance = distance_transform_edt(~mask)
    return float(min(distance[row, col] for row, col in anchors))


def covers_any_anchor(mask: np.ndarray, anchors: list[tuple[int, int]],
                      post_process=None, contact_distance: float = 0.0) -> bool:
    """Does the *production* aperture cover any protected anchor?

    ``post_process`` must be the same scaling/symmetrisation the caller applies
    before the mask becomes the production lens aperture.  Testing the raw mask
    is not sufficient: the symmetric closure unions a mask with its own point
    reflection, so an aperture that covers an anchor's mirror pixel acquires the
    anchor itself and silently swallows a source component that the raw-mask
    test declared safe.
    """
    if contact_distance > 0.0:
        # Anchor "contact": the production aperture covers an anchor or
        # comes within ``contact_distance`` pixels (0.5 PSF FWHM) of one.
        return anchor_min_distance(mask, anchors, post_process) <= contact_distance
    if post_process is not None:
        mask = post_process(mask)
    return any(bool(mask[row, col]) for row, col in anchors)


def anchor_signal_level(signal: np.ndarray, anchors: list[tuple[int, int]]) -> float:
    """Brightest anchor sets the isophote safety threshold."""
    if not anchors:
        return MIN_MODEL_ISOPHOTE_SNR
    return max(float(signal[row, col]) for row, col in anchors)


def source_peak_anchor(source, source_mask: np.ndarray) -> tuple[int, int, dict]:
    """Return (row, column) of the brightest residual pixel in a source mask."""
    usable = np.asarray(source_mask, dtype=bool) & source.valid
    if not usable.any():
        raise ValueError("Cannot define a source anchor from an empty aperture")
    signal = (source.data - source.background) / source.sigma
    row, col = np.unravel_index(np.argmax(np.where(usable, signal, -np.inf)), signal.shape)
    return int(row), int(col), {
        "definition": "brightest residual-SNR pixel inside preliminary automatic source mask",
        "crop_x": int(col),
        "crop_y": int(row),
        "full_x": int(col + source.x0),
        "full_y": int(row + source.y0),
        "residual_snr": float(signal[row, col]),
    }


def _central_mask_from_segmentation(segmentation, lens_product) -> np.ndarray | None:
    try:
        mask = select_central_lens_mask(segmentation, lens_product)
    except RuntimeError:
        return None
    mask = binary_fill_holes(mask) & lens_product.valid
    return mask if mask.any() else None


def _central_isophote(lens_product, threshold_snr: float) -> np.ndarray | None:
    signal = (lens_product.data - lens_product.background) / lens_product.sigma
    # Reusing the central-component selector makes the centre definition exactly
    # the same as in the SExtractor branch.
    segmentation = ((signal >= threshold_snr) & lens_product.valid).astype(np.int16)
    return _central_mask_from_segmentation(segmentation, lens_product)


def _trial_row(method: str, factor: float, threshold: float,
               mask: np.ndarray | None, anchors,
               lens_product, post_process=None, contact_distance: float = 0.0) -> dict:
    anchors = as_anchor_list(anchors)
    signal = (lens_product.data - lens_product.background) / lens_product.sigma
    if mask is None:
        return {
            "method": method, "layer_factor": float(factor),
            "threshold_model_snr": float(threshold), "area_pix": 0,
            "contains_source_peak": False, "hole_pixel_count": 0,
            "model_flux_fraction": 0.0, "admissible": False,
            "protected_anchor_count": len(anchors),
        }
    holes = int((binary_fill_holes(mask) & lens_product.valid & ~mask).sum())
    positive = np.clip(signal, 0.0, None)
    denominator = float(positive[lens_product.valid].sum())
    covered = covers_any_anchor(mask, anchors, post_process, contact_distance)
    return {
        "method": method, "layer_factor": float(factor),
        "threshold_model_snr": float(threshold), "area_pix": int(mask.sum()),
        "contains_source_peak": covered,
        "hole_pixel_count": holes,
        "model_flux_fraction": float(positive[mask].sum() / denominator) if denominator > 0 else 0.0,
        "admissible": bool(
            not covered
            and holes == 0
            and threshold >= SOURCE_PEAK_SAFETY_FACTOR
            * max(anchor_signal_level(signal, anchors), MIN_MODEL_ISOPHOTE_SNR)
        ),
        "protected_anchor_count": len(anchors),
    }


def _choose_largest_admissible(trials: list[dict], masks: list[np.ndarray | None]):
    choices = [(trial, mask) for trial, mask in zip(trials, masks, strict=True)
               if trial["admissible"] and mask is not None]
    if not choices:
        raise RuntimeError("No lens layer can protect the automatic source peak")
    # Area is the primary requirement. A higher threshold wins an exact-area
    # tie, avoiding reliance on an isolated boundary pixel.
    return max(choices, key=lambda item: (item[0]["area_pix"], item[0]["threshold_model_snr"]))


def _enclosed_flux_mask(image: np.ndarray, lens_product) -> tuple[np.ndarray, float]:
    positive = np.clip(np.where(lens_product.valid, image, 0.0), 0.0, None)
    values = positive[lens_product.valid]
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(values[order])
    target = MGE_ENCLOSED_FLUX_FRACTION * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target)), len(order) - 1)
    threshold = float(values[order[index]])
    raw = (positive >= threshold) & lens_product.valid
    mask = _central_mask_from_segmentation(raw.astype(np.int16), lens_product)
    if mask is None:
        raise RuntimeError("A cumulative MGE layer produced no central flux envelope")
    return mask, threshold


def mge_component_lens_mask(lens_product, anchors, component_path: Path,
                            post_process=None, contact_distance: float = 0.0):
    """Select a total-MGE envelope at the fitted Einstein scale.

    The aperture scale is set by the fitted lens mass.  Every preliminary
    source component contributes its brightest reduced-image pixel as a hard
    safety anchor, so no component can be absorbed by the lens aperture.
    """
    anchors = as_anchor_list(anchors)
    with fits.open(component_path) as hdul:
        cube_full = np.asarray(hdul[0].data, dtype=float)
        table = hdul["COMPONENTS"].data
        sigmas = np.asarray(table["SIGMA_ARCSEC"], dtype=float)
        reconstruction_relative_l2 = float(hdul[0].header["RELL2"])
        fitted_einstein_radius_arcsec = float(hdul[0].header["EINRAD"])
    if cube_full.ndim != 3 or cube_full.shape[0] != 30 or len(sigmas) != 30:
        raise ValueError(f"Expected a 30-component MGE cube: {component_path}")
    cube = cube_full[
        :,
        lens_product.y0 : lens_product.y0 + lens_product.data.shape[0],
        lens_product.x0 : lens_product.x0 + lens_product.data.shape[1],
    ]
    radial_sigmas = np.unique(np.round(sigmas, decimals=12))
    if len(radial_sigmas) != 15:
        raise ValueError(
            f"Expected 15 unique radial scales (two bases each), found {len(radial_sigmas)}"
        )
    masks, radial_trials = [], []
    full_positive_flux = float(np.nansum(np.clip(cube, 0.0, None)))
    for layer_number, sigma in enumerate(radial_sigmas, start=1):
        included = sigmas <= sigma + 1.0e-10
        cumulative_model = np.nansum(cube[included], axis=0)
        mask, threshold = _enclosed_flux_mask(cumulative_model, lens_product)
        holes = int((binary_fill_holes(mask) & lens_product.valid & ~mask).sum())
        trial = {
            "method": "mge_components",
            "layer_number_from_centre": int(layer_number),
            "sigma_arcsec": float(sigma),
            "cumulative_gaussian_count": int(np.sum(included)),
            "enclosed_flux_fraction": MGE_ENCLOSED_FLUX_FRACTION,
            "aperture_threshold": threshold,
            "area_pix": int(mask.sum()),
            "contains_source_peak": covers_any_anchor(
                mask, anchors, post_process, contact_distance
            ),
            "hole_pixel_count": holes,
            "model_flux_fraction": float(
                np.nansum(np.clip(cumulative_model[mask], 0.0, None))
                / max(full_positive_flux, 1.0e-30)
            ),
            "admissible": bool(
                not covers_any_anchor(mask, anchors, post_process, contact_distance)
                and holes == 0
            ),
        }
        masks.append(mask)
        radial_trials.append(trial)

    safe_indices = [index for index, trial in enumerate(radial_trials) if trial["admissible"]]
    fixed_layer_available = bool(safe_indices)
    outer_index = max(safe_indices) if safe_indices else None
    selected_index = (
        max(outer_index - MGE_INWARD_LAYER_STEPS, 0)
        if outer_index is not None
        else 0
    )
    fixed_layer_selected = dict(radial_trials[selected_index])
    fixed_layer_mask = masks[selected_index]
    if fixed_layer_available and not fixed_layer_selected["admissible"]:
        raise RuntimeError("The requested inward fitted-MGE layer is not admissible")
    fixed_layer_selected.update({
        "available": fixed_layer_available,
        "unavailable_reason": (
            None if fixed_layer_available
            else "every 90%-flux cumulative MGE layer contains the automatic source anchor"
        ),
        "outer_safe_layer_number_from_centre": (
            int(outer_index + 1) if outer_index is not None else None
        ),
        "outer_safe_sigma_arcsec": (
            float(radial_trials[outer_index]["sigma_arcsec"])
            if outer_index is not None else None
        ),
        "inward_layer_steps": MGE_INWARD_LAYER_STEPS,
        "total_radial_layers": int(len(radial_sigmas)),
        "total_fitted_gaussians": int(len(sigmas)),
        "component_cube": str(component_path),
        "component_sum_relative_l2": reconstruction_relative_l2,
        "selection_rule": (
            "outermost safe cumulative fitted-Gaussian radial layer, then two "
            "unique-sigma layers inward; if no such layer exists, retain the "
            "innermost layer only as an explicitly unavailable diagnostic"
        ),
    })

    # Nested total-flux envelopes choose a smooth boundary without depending on
    # the arbitrary ordering of the two fitted Gaussian bases at each sigma.
    # Use the exact FitImaging main-lens image here.  The component cube remains
    # useful for the optional radial-layer diagnostic, but it was independently
    # rendered on the full detector and can inherit a half-pixel parity offset.
    total_model = np.where(lens_product.valid, lens_product.data, 0.0)
    positive = np.clip(np.where(lens_product.valid, total_model, 0.0), 0.0, None)
    values = positive[lens_product.valid]
    order = np.argsort(values)[::-1]
    cumulative = np.cumsum(values[order])
    envelope_trials = []
    fractions = np.arange(
        MGE_ENVELOPE_FLUX_START,
        MGE_ENVELOPE_FLUX_STOP + 0.5 * MGE_ENVELOPE_FLUX_STEP,
        MGE_ENVELOPE_FLUX_STEP,
    )
    pixel_scale_arcsec = float(
        np.sqrt(np.abs(np.linalg.det(lens_product.wcs.pixel_scale_matrix))) * 3600.0
    )
    einstein_radius_pix = fitted_einstein_radius_arcsec / pixel_scale_arcsec
    for fraction in fractions:
        index = min(
            int(np.searchsorted(cumulative, float(fraction) * cumulative[-1])),
            len(order) - 1,
        )
        threshold = float(values[order[index]])
        mask = _central_mask_from_segmentation(
            ((positive >= threshold) & lens_product.valid).astype(np.int16), lens_product
        )
        if mask is None:
            continue
        distance = anchor_min_distance(mask, anchors, post_process)
        envelope_trials.append({
            "enclosed_flux_fraction": float(fraction),
            "mask": mask,
            "area_pix": int(mask.sum()),
            "equivalent_radius_pix": float(np.sqrt(mask.sum() / np.pi)),
            "anchor_min_distance_pix": distance,
            "contains_source_peak": bool(
                distance <= contact_distance if contact_distance > 0.0 else distance == 0.0
            ),
        })
    safe_envelopes = [trial for trial in envelope_trials if not trial["contains_source_peak"]]
    contact_fallback = not safe_envelopes
    if contact_fallback:
        # No envelope keeps the required distance; fall back to
        # the smallest envelope and flag the system for the HTML warning.
        selected_trial = min(envelope_trials, key=lambda trial: trial["area_pix"])
    else:
        selected_trial = min(
            safe_envelopes,
            key=lambda trial: abs(trial["equivalent_radius_pix"] - einstein_radius_pix),
        )
    unconstrained_einstein_trial = min(
        envelope_trials,
        key=lambda trial: abs(trial["equivalent_radius_pix"] - einstein_radius_pix),
    )
    mask = selected_trial["mask"]
    selected = {
        "method": "mge_components_einstein_envelope",
        "area_pix": int(mask.sum()),
        "contains_source_peak": bool(selected_trial["contains_source_peak"]),
        "covers_anchor": covers_any_anchor(mask, anchors, post_process),
        "hole_pixel_count": int((binary_fill_holes(mask) & ~mask).sum()),
        "model_flux_fraction": float(selected_trial["enclosed_flux_fraction"]),
        "admissible": bool(not selected_trial["contains_source_peak"]),
        "contact_distance_pix": float(contact_distance),
        "contact_definition": (
            "production aperture covers an anchor or lies within contact_distance_pix "
            "(Euclidean, pixel centres) of one"
        ),
        "anchor_min_distance_pix": float(selected_trial["anchor_min_distance_pix"]),
        "contact_fallback_smallest_envelope": bool(contact_fallback),
        "warning": (
            "no envelope keeps the required distance from every anchor; the smallest "
            "envelope was used" if contact_fallback else None
        ),
        "anchor_guard_applied_to": (
            "production geometry" if post_process is not None else "raw mask"
        ),
        "protected_anchor_count": len(anchors),
        "enclosed_flux_fraction": float(selected_trial["enclosed_flux_fraction"]),
        "equivalent_radius_pix": float(selected_trial["equivalent_radius_pix"]),
        "fitted_einstein_radius_arcsec": fitted_einstein_radius_arcsec,
        "fitted_einstein_radius_pix": einstein_radius_pix,
        "unconstrained_einstein_enclosed_flux_fraction": float(
            unconstrained_einstein_trial["enclosed_flux_fraction"]
        ),
        "unconstrained_einstein_equivalent_radius_pix": float(
            unconstrained_einstein_trial["equivalent_radius_pix"]
        ),
        "unconstrained_einstein_area_pix": int(unconstrained_einstein_trial["area_pix"]),
        "unconstrained_einstein_anchor_min_distance_pix": float(
            unconstrained_einstein_trial["anchor_min_distance_pix"]
        ),
        "source_anchor_forced_inward": bool(
            unconstrained_einstein_trial["contains_source_peak"]
        ),
        "fixed_layer_baseline": {
            key: value for key, value in fixed_layer_selected.items()
            if key not in {"component_cube"}
        },
        "component_cube": str(component_path),
        "component_sum_relative_l2": reconstruction_relative_l2,
        "total_radial_layers": 15,
        "total_fitted_gaussians": 30,
        "selection_rule": (
            "exact FitImaging main-lens positive-flux envelope whose equivalent-area radius is "
            "closest to the fitted Einstein radius; move inward only when needed "
            "to protect the automatic reduced-image source anchor"
        ),
    }
    trace = [
        {key: value for key, value in trial.items() if key != "mask"}
        for trial in envelope_trials
    ]
    fixed_layer_metadata = dict(fixed_layer_selected)
    fixed_layer_metadata.update({
        "method": "mge_components_fixed_flux90",
        "selection_rule": (
            "R3: outermost source-anchor-safe cumulative fitted-Gaussian layer, "
            "then two unique-sigma layers inward; retain the central envelope "
            "containing 90% of that cumulative model's positive flux"
        ),
    })
    return (
        mask,
        selected,
        trace,
        fixed_layer_mask,
        fixed_layer_metadata,
        radial_trials,
    )


def sextractor_lens_mask(lens_product, anchors, executable: Path,
                         base_settings: dict, work_root: Path,
                         post_process=None, contact_distance: float = 0.0):
    anchors = as_anchor_list(anchors)
    signal = (lens_product.data - lens_product.background) / lens_product.sigma
    anchor_level = max(anchor_signal_level(signal, anchors), MIN_MODEL_ISOPHOTE_SNR)
    settings = deepcopy(base_settings)
    # Lens extraction is intentionally distinct from source extraction: use an
    # absolute model-S/N threshold, suppress deblending, and allow small seeds.
    settings.update({
        "input_mode": "snr",
        "detect_peak_fraction": 0.0,
        "detect_minarea": 3,
        "deblend_nthresh": 1,
        "deblend_mincont": 1.0,
    })
    factors = np.arange(
        LAYER_FACTOR_MIN,
        LAYER_FACTOR_MAX + 0.5 * LAYER_FACTOR_STEP,
        LAYER_FACTOR_STEP,
    )
    masks, trials = [], []
    for factor in factors:
        threshold = max(MIN_MODEL_ISOPHOTE_SNR, anchor_level * float(factor))
        trial_settings = deepcopy(settings)
        trial_settings["detect_snr_floor"] = float(threshold)
        segmentation = run_sextractor(
            lens_product,
            work_root / f"factor_{factor:.3f}",
            executable,
            trial_settings,
        )
        mask = _central_mask_from_segmentation(segmentation, lens_product)
        masks.append(mask)
        trials.append(_trial_row(
            "sextractor", factor, threshold, mask, anchors, lens_product,
            post_process, contact_distance
        ))
    selected, mask = _choose_largest_admissible(trials, masks)
    selected = dict(selected)
    selected["anchor_model_snr"] = anchor_signal_level(signal, anchors)
    selected["protected_anchor_count"] = len(anchors)
    selected["selection_rule"] = (
        "largest non-deblended central SExtractor footprint excluding the "
        "brightest pixel of every preliminary source component"
    )
    selected["settings"] = settings
    return mask, selected, trials


def build_lens_masks(lens_product, anchors, executable: Path,
                     base_settings: dict, work_root: Path, component_path: Path,
                     post_process=None, contact_distance: float = 0.0) -> dict:
    """Build the candidate lens apertures.

    ``post_process`` is the caller's raw-mask -> production-aperture transform
    (radial scaling and, when requested, point-reflection closure).  It is
    applied inside every admissibility test so a layer is accepted only if the
    aperture that actually gets used protects every source anchor.
    """
    anchors = as_anchor_list(anchors)
    try:
        sex_mask, sex_meta, sex_trials = sextractor_lens_mask(
            lens_product, anchors, executable, base_settings, work_root / "sextractor",
            post_process, contact_distance,
        )
    except RuntimeError as exc:
        # Diagnostic method only: an unreachable contact distance must not stop
        # the production (mge_isophote) aperture.
        sex_mask, sex_meta, sex_trials = None, {"available": False, "error": str(exc)}, []
    (
        mge_mask,
        mge_meta,
        mge_trials,
        fixed_mask,
        fixed_meta,
        fixed_trials,
    ) = mge_component_lens_mask(
        lens_product, anchors, component_path, post_process, contact_distance
    )
    return {
        "sextractor": {"mask": sex_mask, "metadata": sex_meta, "trials": sex_trials},
        "mge_isophote": {"mask": mge_mask, "metadata": mge_meta, "trials": mge_trials},
        "mge_fixed_flux90": {
            "mask": fixed_mask,
            "metadata": fixed_meta,
            "trials": fixed_trials,
        },
    }


def evaluate_lens_mask(mask: np.ndarray, manual_source: np.ndarray | None,
                       manual_peak_yx: tuple[int, int] | None) -> dict:
    """Post-selection diagnostics only; never used to choose a layer."""
    if manual_source is None:
        return {}
    overlap = int(np.sum(mask & manual_source))
    distance = distance_transform_edt(~mask)
    gap = float(np.min(distance[manual_source])) if manual_source.any() else float("nan")
    peak_covered = bool(mask[manual_peak_yx]) if manual_peak_yx is not None else None
    return {
        "manual_source_area_pix": int(manual_source.sum()),
        "manual_source_overlap_pix": overlap,
        "manual_source_overlap_fraction": float(overlap / max(int(manual_source.sum()), 1)),
        "minimum_gap_to_manual_source_pix": gap,
        "touches_or_overlaps_manual_source": bool(gap == 0.0),
        "covers_brightest_manual_source_pixel": peak_covered,
    }
