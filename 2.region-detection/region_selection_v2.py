#!/usr/bin/env python3
"""Single-pass lens, source and point-reflected background selection.

The source-selection core shares one 4-sigma segmentation between protected
anchors and source candidates. MGE lens envelopes respect every accepted
anchor. Candidates are grown, classified by the point-reflection source
fraction, carved against contaminants and their own reflections, checked
against hard gates, then ranked by effective S/N. Only this final selection
is exported for downstream photometry.

All apertures share the exact-fit lens-light centre and detector grid.
Diagnostic threshold scans never replace the production selection.
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
from regions import Regions  # noqa: E402
from skimage.draw import polygon as draw_polygon  # noqa: E402

import source_selection as selection_core  # noqa: E402
from lens_region_methods_v2 import (  # noqa: E402
    DEFAULT_METHOD,
    build_lens_masks,
    evaluate_lens_mask,
    source_peak_anchor,
)


sys.path.insert(0, str(SCRIPT_DIR.parent))
from photo_lens.workflow import available_systems, model_products, REGION_METHOD
SYSTEMS = tuple(item["id"] for item in available_systems())

# Image diagnostics (reported, not selection gates).
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
# Integrated flux may not change by more than this fraction between adjacent
# threshold steps, which replaces v1's pixel-set Jaccard stability gate.
MAX_NEIGHBOUR_FLUX_CHANGE = 0.25

DEFAULT_EFFECTIVE_SNR_EPSILON = 0.015

MIN_COMPONENT_AREA_PIX = 5
# A component may constrain the lens aperture only if it is a credible source.
# The detector floor (5 px) admits lens-subtraction residual knots; protecting
# one of those needlessly shrinks the lens aperture.  Measured on the current
# systems: genuine protected components have 58-91 px, while the rejected knots
# are 7 and 13 px and sit at 0.6-1.0 Einstein radii.
ANCHOR_MIN_COMPONENT_AREA_PIX = 20
ANALYSIS_HALF_SIZE_PIX = 30
# A modeled satellite is photometrically relevant where its PSF-convolved
# surface brightness is at least one robust residual-image sigma.  This makes
# the footprint adapt to both the fitted light distribution and image depth.
SATELLITE_MODEL_SNR_THRESHOLD = 1.0
SATELLITE_COMPACT_FLUX_RETENTION = 0.50
SATELLITE_COMPACT_THRESHOLD_STEP_SIGMA = 0.05
# Degeneracy gate.  A satellite Gaussian that has run away to a large sigma
# absorbs diffuse flux instead of a companion; its 1-sigma isophote is then
# enormous and the flux-retention compaction cannot shrink it, because a nearly
# flat profile loses more than half its flux at the first threshold step.  Such
# a component is not a satellite and must not produce an exclusion aperture.
# Measured on the current systems: genuine satellites have peak model S/N
# 2.9-25.9 and half-light radii 2.4-3.6 px; the two degenerate ones have peak
# S/N 1.6 and 1.9 with half-light radii 39.6 and 8.1 px.
SATELLITE_MIN_PEAK_MODEL_SNR = 2.5
SATELLITE_MAX_HALF_LIGHT_RADIUS_PIX = 6.0


def positions_gui_centre_on_product(
    positions_path: Path, product
) -> tuple[tuple[float, float], dict]:
    """Map the saved GUI centre onto a WCS product grid.

    Position files may contain a saved subpixel antisymmetry minimum. The
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





def symmetric_overlap(mask: np.ndarray, centre_xy: tuple[float, float]) -> float:
    if not mask.any():
        return 1.0
    return float(np.sum(mask & _reflect_mask(mask, centre_xy)) / mask.sum())




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






def modeled_satellite_masks(products: Path, system_id: str, source):
    """Build the smallest compact satellite mask meeting the configured retention."""
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
        model_path = (products / item["model"]).resolve()
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

        # Reject a degenerate satellite component before it becomes an aperture.
        peak_model_snr = float(positive.max() / source.sigma)
        total_positive = float(positive.sum())
        yy, xx = np.indices(model.shape)
        radius = np.hypot(xx - centre_x, yy - centre_y)
        order = np.argsort(radius.ravel())
        cumulative = np.cumsum(positive.ravel()[order])
        half_light_radius_pix = (
            float(radius.ravel()[order][int(np.searchsorted(cumulative, 0.5 * total_positive))])
            if total_positive > 0.0 else float("inf")
        )
        degenerate = (
            peak_model_snr < SATELLITE_MIN_PEAK_MODEL_SNR
            or half_light_radius_pix > SATELLITE_MAX_HALF_LIGHT_RADIUS_PIX
        )
        if degenerate:
            records.append({
                "index": int(item["index"]),
                "selected": False,
                "reason": (
                    "degenerate satellite light model: peak model S/N "
                    f"{peak_model_snr:.2f} (min {SATELLITE_MIN_PEAK_MODEL_SNR}) and "
                    f"half-light radius {half_light_radius_pix:.2f} px "
                    f"(max {SATELLITE_MAX_HALF_LIGHT_RADIUS_PIX}); the fitted "
                    "component is diffuse rather than a compact companion, so no "
                    "exclusion aperture is emitted"
                ),
                "peak_model_snr": peak_model_snr,
                "half_light_radius_pix": half_light_radius_pix,
                "one_sigma_area_pix": int(current.sum()),
                "model": str(model_path),
            })
            continue
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
            "peak_model_snr": peak_model_snr,
            "half_light_radius_pix": half_light_radius_pix,
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
            "nearest the fitted centre, with >=5 pixels and retaining "
            f">={100.0 * SATELLITE_COMPACT_FLUX_RETENTION:g}% "
            "of the positive model flux inside the original 1-sigma mask"
        ),
        "threshold_model_snr": SATELLITE_MODEL_SNR_THRESHOLD,
        "degeneracy_gate": {
            "min_peak_model_snr": SATELLITE_MIN_PEAK_MODEL_SNR,
            "max_half_light_radius_pix": SATELLITE_MAX_HALF_LIGHT_RADIUS_PIX,
            "rejected_count": int(sum(
                1 for record in records
                if not record["selected"]
                and str(record.get("reason", "")).startswith("degenerate")
            )),
        },
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








# ---------------------------------------------------------------------------
# Trial construction and scoring
# ---------------------------------------------------------------------------

def _split_segmentation(segmentation: np.ndarray) -> list[np.ndarray]:
    components = []
    for source_label in (int(v) for v in np.unique(segmentation) if v > 0):
        labels, count = connected_labels(segmentation == source_label)
        for index in range(1, count + 1):
            component = labels == index
            if int(component.sum()) >= MIN_COMPONENT_AREA_PIX:
                components.append(component)
    return components










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




def write_lens_trace(path: Path, trials: list[dict]) -> None:
    preferred = [
        "method", "layer_number_from_centre", "sigma_arcsec",
        "cumulative_gaussian_count", "enclosed_flux_fraction", "aperture_threshold",
        "equivalent_radius_pix", "source_overlap_pix",
        "layer_factor", "threshold_model_snr", "area_pix", "contains_source_peak",
        "hole_pixel_count", "model_flux_fraction", "admissible",
        "anchor_min_distance_pix",
    ]
    fields = [key for key in preferred if any(key in row for row in trials)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in trials)


def save_lens_method_comparison(path: Path, lens_product, source, methods: dict,
                                manual_source, manual_peak_yx, anchors) -> None:
    """Overlay every protected source anchor, not just the primary one."""
    anchors = list(anchors)
    anchor_yx = anchors[0] if anchors else None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].imshow(lens_product.data, origin="lower", cmap="gray", norm=_display_norm(lens_product))
    axes[0].plot(lens_product.centre_crop_xy[0], lens_product.centre_crop_xy[1],
                 "+", color="cyan", ms=13, mew=2)
    for row, col in anchors:
        axes[0].plot(col, row, "*", color="red", ms=11)
    axes[0].set_title("summed MGE lens-light model")
    for ax, method in zip(axes[1:], ("sextractor", "mge_isophote"), strict=True):
        item = methods[method]
        ax.imshow(source.data, origin="lower", cmap="gray", norm=_display_norm(source))
        if item["mask"] is None:
            ax.set_title(f"{method}: unavailable\n{item['metadata'].get('error', '')[:60]}")
            for row, col in anchors:
                ax.plot(col, row, "*", color="red", ms=11)
            continue
        ax.contour(item["mask"], levels=[0.5], colors="magenta", linewidths=2, origin="lower")
        if manual_source is not None:
            ax.contour(manual_source, levels=[0.5], colors="lime", linewidths=1.5,
                       linestyles="--", origin="lower")
        for row, col in anchors:
            ax.plot(col, row, "*", color="red", ms=11)
        if manual_peak_yx is not None and manual_peak_yx != anchor_yx:
            ax.plot(manual_peak_yx[1], manual_peak_yx[0], "x", color="yellow", ms=9)
        meta = item["metadata"]
        evaluation = item.get("evaluation") or {}
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
    parser = argparse.ArgumentParser(description="Automatic lens, source and reflected-background selection")
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--sextractor", default=None)
    parser.add_argument(
        "--products-dir",
        type=Path,
        default=None,
        help="Stage-specific WCS-products directory; defaults to the selected model stage.",
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
        default=None,
        help=(
            "Radial scale before symmetry (default: HSC 1.0, CFHT 0.85)."
        ),
    )
    parser.add_argument(
        "--effective-snr-epsilon",
        type=float,
        default=None,
        help=(
            "Enable R1 candidate ranking with the fractional lens-model leakage "
            f"floor (production value: {DEFAULT_EFFECTIVE_SNR_EPSILON})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Versioned output directory; defaults to outputs/<name>.",
    )
    parser.add_argument(
        "--pair-guard-pix",
        type=int,
        default=0,
        help=(
            "Image-only safety gap between a source mask and its point reflection. "
            "Zero uses the standard selection; positive values carve only source pixels "
            "within this many detector pixels of the reflected aperture."
        ),
    )
    parser.add_argument(
        "--background-centre",
        choices=("model",),
        default="model",
        help=(
            "Retained for command-line compatibility; the only accepted value. "
            "Source, lens and reflected background all share the WCS-product "
            "model centre. Reflecting the background about a separate centre "
            "displaces it by twice the centre discrepancy, so the previous "
            "'positions-gui' option has been removed."
        ),
    )
    parser.add_argument(
        "--positions-file",
        type=Path,
        default=None,
        help="Optional positions_gui.txt for centre diagnostics; reflection always uses the model centre.",
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
    if args.pair_guard_pix != 0:
        parser.error("Extra pair guards are not part of the single-pass selection; use 0")
    if args.effective_snr_epsilon is not None and not np.isclose(args.effective_snr_epsilon, selection_core.EFFECTIVE_SNR_EPSILON):
        parser.error(f"Production ranking uses epsilon={selection_core.EFFECTIVE_SNR_EPSILON}")

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
        else model_products(system_id)
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

    # One centre for everything.  The reflected background is the mirror of the
    # source aperture, and the source aperture is built against a lens aperture
    # centred on the model centre; reflecting about any other point breaks that
    # pairing and displaces the background by *twice* the centre discrepancy.
    # The positions_gui centre is still mapped, but only as a recorded
    # comparison -- it never defines the reflection.
    background_centre_xy = source.centre_crop_xy
    background_centre_metadata = {
        "method": "latest_modeled_lens_light_wcs_centre",
        "rule": (
            "source, lens and reflected background share one centre; the "
            "reflection centre is never taken from a separate definition"
        ),
    }
    positions_path = (
        args.positions_file.resolve()
        if args.positions_file is not None
        else SCRIPT_DIR.parent / "data" / "bb" / system_id / "positions_gui.txt"
    )
    if positions_path.is_file():
        try:
            gui_centre_xy, gui_metadata = positions_gui_centre_on_product(
                positions_path, source
            )
        except (ValueError, FileNotFoundError) as exc:
            background_centre_metadata["positions_gui_comparison"] = {
                "available": False, "reason": repr(exc)
            }
        else:
            background_centre_metadata["positions_gui_comparison"] = {
                "available": True,
                "centre_crop_x": float(gui_centre_xy[0]),
                "centre_crop_y": float(gui_centre_xy[1]),
                "distance_from_model_centre_pix": float(np.hypot(
                    gui_centre_xy[0] - source.centre_crop_xy[0],
                    gui_centre_xy[1] - source.centre_crop_xy[1],
                )),
                "metadata": gui_metadata,
            }

    sextractor = find_sextractor(args.sextractor)
    radius_pix = einstein_radius_pixels(MODELING_ROOT / f"{system_id}", source)
    system_root = MODELING_ROOT / f"{system_id}"
    system_config = json.loads((system_root / "config.json").read_text())
    if args.lens_scale is None:
        args.lens_scale = 1.0 if system_config["bb_survey"] == "HSC" else 0.85
    product_pixel_scale = float(
        np.sqrt(np.abs(np.linalg.det(source.wcs.pixel_scale_matrix))) * 3600.0
    )
    component_path = products / f"{system_id}_lens_mge_components_wcs.fits"
    fitted_einstein_radius_pix = (
        float(fits.getheader(component_path)["EINRAD"]) / product_pixel_scale
    )
    fwhm_pix = selection_core.psf_fwhm_pix(
        (system_root / system_config["psf"]).resolve(), float(system_config["pixel_scale"]), product_pixel_scale
    )
    contact_distance = selection_core.LENS_CONTACT_DISTANCE_FWHM * fwhm_pix
    point_config = json.loads((SCRIPT_DIR / "point_fraction_config.json").read_text())
    override = point_config.get("source_sextractor_overrides", {}).get(system_id)
    if override is not None and (
        not np.isclose(float(override["fixed_detect_sigma"]), selection_core.ANCHOR_DETECT_SIGMA)
        or not np.isclose(float(override["grow_sigma_ratio"]), selection_core.SOURCE_GROW_SIGMA_RATIO)
    ):
        raise RuntimeError(
            "point_fraction_config.json overrides disagree with the single production "
            f"segmentation ({selection_core.ANCHOR_DETECT_SIGMA} sigma, grow "
            f"{selection_core.SOURCE_GROW_SIGMA_RATIO}): {override}"
        )
    merge_group = point_config.get("source_component_merge_groups_1based", {}).get(system_id)

    # One segmentation, shared by the anchors and the source selection.
    cache = selection_core.SegmentationCache(
        sys.modules[__name__], source, config["sextractor"]["source"], sextractor,
        output / "parameter_search/segmentation",
    )
    components = cache.components(selection_core.ANCHOR_DETECT_SIGMA)
    if not components:
        raise RuntimeError(f"No source component at {selection_core.ANCHOR_DETECT_SIGMA} sigma")
    anchors, component_records = selection_core.select_anchors(
        source, components, fwhm_pix, fitted_einstein_radius_pix
    )
    anchor_metadata = {
        "definition": (
            f"brightest residual-S/N pixel of every {selection_core.ANCHOR_DETECT_SIGMA}-sigma "
            "SExtractor component of the lens-plane-subtracted image"
        ),
        "detect_sigma": selection_core.ANCHOR_DETECT_SIGMA,
        "core_radius_pix": selection_core.ANCHOR_CORE_RADIUS_FWHM * fwhm_pix,
        "core_radius_fwhm": selection_core.ANCHOR_CORE_RADIUS_FWHM,
        "anchor_min_component_area_pix": selection_core.ANCHOR_MIN_COMPONENT_AREA_PIX,
        "fitted_einstein_radius_pix": fitted_einstein_radius_pix,
        "einstein_radius_definition": (
            "PowerLaw einstein_radius parameter (EINRAD), not a mean-convergence radius"
        ),
        "psf_fwhm_pix": fwhm_pix,
        "protected_anchor_count": len(anchors),
        "components": component_records,
        "components_label_image": f"{system_id}_components_{selection_core.ANCHOR_DETECT_SIGMA:.1f}sigma.fits",
    }

    def lens_post_process(mask: np.ndarray) -> np.ndarray:
        scaled = (
            scale_mask_about_centre(mask, source.centre_crop_xy, args.lens_scale)
            & source.valid
        )
        if args.lens_geometry == "symmetric":
            return symmetric_closure(scaled, source.centre_crop_xy, source.valid)
        return scaled

    lens_methods = build_lens_masks(
        lens_product,
        anchors,
        sextractor,
        config["sextractor"]["lens"],
        output / "parameter_search/lens_wide",
        component_path,
        lens_post_process,
        contact_distance,
    )
    lens_mask_raw = lens_methods[args.lens_method]["mask"]
    lens_mask_scaled = (
        scale_mask_about_centre(lens_mask_raw, source.centre_crop_xy, args.lens_scale)
        & source.valid
    )
    if args.lens_geometry == "symmetric":
        lens_mask = symmetric_closure(lens_mask_scaled, source.centre_crop_xy, source.valid)
    else:
        lens_mask = lens_mask_scaled
    reflected_lens_mask = _reflect_mask(lens_mask, source.centre_crop_xy) & source.valid
    lens_symmetry_mismatch_pix = int(np.sum(reflected_lens_mask & ~lens_mask))
    if args.lens_geometry == "symmetric" and lens_symmetry_mismatch_pix != 0:
        raise RuntimeError("Symmetric lens aperture is not closed under point reflection")
    covered_anchors = [(row, col) for row, col in anchors if lens_mask[row, col]]
    if covered_anchors:
        raise RuntimeError(
            "The production lens aperture covers the brightest pixel of a source "
            f"component at (row, col) {covered_anchors}."
        )
    lens_meta = lens_methods[args.lens_method]["metadata"]
    lens_contact = {
        "contact_distance_pix": contact_distance,
        "contact_distance_fwhm": selection_core.LENS_CONTACT_DISTANCE_FWHM,
        "anchor_min_distance_pix": lens_meta.get("anchor_min_distance_pix"),
        "fallback_smallest_envelope": bool(
            lens_meta.get("contact_fallback_smallest_envelope", False)
        ),
        "warning": lens_meta.get("warning"),
        "unconstrained_einstein_area_pix": lens_meta.get("unconstrained_einstein_area_pix"),
    }

    satellite_masks, satellite_metadata = modeled_satellite_masks(products, system_id, source)
    satellite_union_mask = np.zeros_like(lens_mask, dtype=bool)
    for satellite_mask in satellite_masks:
        satellite_union_mask |= satellite_mask
    if satellite_masks:
        write_ds9_regions(
            output / f"{system_id}_satellites_v2.reg", satellite_masks, source,
            [f"satellite_{index}_v2" for index in range(len(satellite_masks))],
            ["red"] * len(satellite_masks),
        )
        write_ds9_regions(
            output / f"{system_id}_satellite_union_v2.reg", [satellite_union_mask],
            source, ["satellite_union_v2"], ["red"],
        )

    # Single source selection: grow, r_s, carve, hard gates, effective S/N.
    raw_crop = selection_core.science_crop(system_config, system_root, source)
    fits.writeto(
        output / anchor_metadata["components_label_image"],
        selection_core.label_image(components, source.data.shape), overwrite=True,
    )
    try:
        selection = selection_core.select_source(
            sys.modules[__name__], source, lens_product, raw_crop, components,
            component_records, lens_mask, satellite_union_mask, merge_group,
        )
    except selection_core.SourceSelectionError as exc:
        # Failure branch: keep every decision for the HTML report, then stop.
        failure = {
            "version": REGION_METHOD,
            "system": system_id,
            "status": "failed",
            "error": str(exc),
            "lens_selection": {
                "production_method": args.lens_method,
                "source_anchor": anchor_metadata,
                "contact": lens_contact,
                "production_area_pix": int(lens_mask.sum()),
            },
            "satellite_exclusion": satellite_metadata,
            "source_selection": {"candidates": exc.candidates, "merge_group_1based": merge_group},
        }
        (output / f"{system_id}_v2_adaptation.json").write_text(
            json.dumps(failure, indent=2, default=float) + "\n"
        )
        (output / f"{system_id}_v2_summary.json").write_text(
            json.dumps({"system": system_id, "version": REGION_METHOD, "status": "failed",
                        "error": str(exc), "wcs_products_dir": str(products)},
                       indent=2, default=float) + "\n"
        )
        write_ds9_regions(output / f"{system_id}_lens_initial_v2.reg", [lens_mask],
                          lens_product, [f"lens_initial_v2_{args.lens_method}"])
        raise
    selected_mask = selection["source_mask"]
    lens_final = selection["lens_final"]
    neighbour = selection_core.neighbour_flux_diagnostic(
        sys.modules[__name__], cache, source, lens_product, raw_crop, lens_mask,
        satellite_union_mask, selection["winner_members_1based"], components,
    )
    winner_flux = float(np.sum(source.data[selected_mask] - source.background))
    for item in neighbour.values():
        if "region_flux" in item and winner_flux > 0:
            item["relative_flux_change"] = abs(item["region_flux"] - winner_flux) / winner_flux
    neighbour_change = max(
        (item.get("relative_flux_change", 0.0) for item in neighbour.values()), default=0.0
    )

    measured: dict = {}
    refresh_selected_measurements(
        measured, selected_mask, source, lens_product, lens_final,
        selection_core.EFFECTIVE_SNR_EPSILON,
    )
    measured.pop("mask", None)
    winner = selection["winner"]
    metrics = {
        "threshold_sigma": selection_core.ANCHOR_DETECT_SIGMA,
        "area_pix": int(selected_mask.sum()),
        "integrated_snr": float(measured["integrated_snr"]),
        "effective_snr": float(measured["effective_snr"]),
        "effective_snr_epsilon": selection_core.EFFECTIVE_SNR_EPSILON,
        "mirror_asymmetry": float(measured["mirror_asymmetry"]),
        "lens_model_flux_fraction": float(measured["lens_model_flux_fraction"]),
        "symmetric_overlap_fraction": float(measured["symmetric_overlap_fraction"]),
        "hole_pixel_count": int(selection["holes"].sum()),
        "final_lens_overlap_pix": int(np.sum(selected_mask & lens_final)),
        "contains_lens_centre": bool(measured["contains_lens_centre"]),
        "neighbour_flux_change_diagnostic": float(neighbour_change),
        "selected_candidate": winner["candidate"],
        "selected_members_1based": selection["winner_members_1based"],
        "area_before_carving_pix": winner["area_before_carving_pix"],
        "snr_retention": winner["snr_retention"],
    }
    metrics["passes_all_hard_gates"] = bool(winner["passes_hard_gates"])
    warnings = []
    if lens_contact["fallback_smallest_envelope"]:
        warnings.append(
            "lens: no MGE envelope keeps 0.5 FWHM from every anchor; smallest envelope used "
            f"(min anchor distance {lens_contact['anchor_min_distance_pix']:.2f} px < "
            f"{contact_distance:.2f} px)"
        )
    if metrics["hole_pixel_count"]:
        warnings.append(
            f"source has {metrics['hole_pixel_count']} hole pixels; DS9 polygons fill holes "
            "on read-back"
        )
    if metrics["mirror_asymmetry"] < MIN_MIRROR_ASYMMETRY:
        warnings.append(
            f"mirror asymmetry {metrics['mirror_asymmetry']:.2f} < {MIN_MIRROR_ASYMMETRY} "
            "(display only)"
        )
    if neighbour_change > MAX_NEIGHBOUR_FLUX_CHANGE:
        warnings.append(
            f"neighbour-threshold flux change {neighbour_change:.2f} > "
            f"{MAX_NEIGHBOUR_FLUX_CHANGE} (diagnostic only)"
        )

    # ---------------- background, region files and read-back checks ------------
    full_shape = (
        int(source.header.get("NAXIS2", source.y0 + source.data.shape[0])),
        int(source.header.get("NAXIS1", source.x0 + source.data.shape[1])),
    )
    crop_y = slice(source.y0, source.y0 + source.data.shape[0])
    crop_x = slice(source.x0, source.x0 + source.data.shape[1])

    def to_full(mask):
        full = np.zeros(full_shape, dtype=bool)
        full[crop_y, crop_x] = mask
        return full

    source_mask_full = to_full(selected_mask)
    valid_model_full = to_full(source.valid)
    lens_mask_full = to_full(lens_final)
    satellite_mask_full = to_full(satellite_union_mask)
    secondary_full = to_full(selection["secondary"])
    background_centre_full_xy = (
        background_centre_xy[0] + source.x0, background_centre_xy[1] + source.y0,
    )
    background_mask_full = _reflect_mask(source_mask_full, background_centre_full_xy)
    background_mask = background_mask_full[crop_y, crop_x]
    if not background_mask_full.any():
        raise RuntimeError("Point-reflected automatic background aperture is empty")
    if np.any(background_mask_full & source_mask_full):
        raise RuntimeError("Automatic source and point-reflected background overlap")
    background_satellite_overlap_pix = int(np.sum(background_mask_full & satellite_mask_full))
    background_secondary_overlap_pix = int(np.sum(background_mask_full & secondary_full))
    if background_satellite_overlap_pix or background_secondary_overlap_pix:
        raise RuntimeError(
            "Reflected background touches a satellite or another source component "
            f"({background_satellite_overlap_pix}/{background_secondary_overlap_pix} px)"
        )
    background_lens_overlap_pix = int(np.sum(background_mask_full & lens_mask_full))

    # Exact pixel-cell outlines: the written background is then exactly the
    # reflected source cells, so carving's no-overlap guarantee survives.
    selection_core.write_exact_cell_region(output / f"{system_id}_source_v2.reg", selected_mask,
                                source, "source_v2", "green")
    selection_core.write_exact_reflected_region(
        output / f"{system_id}_background_v2.reg", selected_mask, source,
        background_centre_full_xy, "background_v2_point_reflection", "cyan",
    )
    write_ds9_regions(output / f"{system_id}_lens_v2.reg", [lens_final], lens_product,
                      [f"lens_v2_{args.lens_method}_rs"])
    write_ds9_regions(output / f"{system_id}_lens_initial_v2.reg", [lens_mask],
                      lens_product, [f"lens_initial_v2_{args.lens_method}"])
    write_ds9_regions(
        output / f"{system_id}_all_components_v2.reg", components, source,
        [f"component_{i + 1}" for i in range(len(components))],
        ["green" if (i + 1) in selection["winner_members_1based"] else "blue"
         for i in range(len(components))],
    )
    if selection["secondary"].any():
        write_ds9_regions(output / f"{system_id}_secondary_sources_v2.reg",
                          [selection["secondary"]], source, ["secondary_sources"], ["blue"])

    # Component label image (crop WCS) and the exact masks used downstream.
    header = source.wcs.celestial.to_header()
    header["CRPIX1"] -= source.x0
    header["CRPIX2"] -= source.y0
    header["LTV1"] = -source.x0
    header["LTV2"] = -source.y0
    header["DETSIG"] = (selection_core.ANCHOR_DETECT_SIGMA, "SExtractor detection threshold (sigma)")
    header["COMMENT"] = "Pixel value = 1-based component index (0 = no component)"
    fits.writeto(output / anchor_metadata["components_label_image"],
                 selection_core.label_image(components, source.data.shape), header, overwrite=True)
    np.savez_compressed(
        output / f"{system_id}_selection_masks.npz",
        source=selected_mask, background=background_mask, lens_initial=lens_mask,
        lens_final=lens_final, satellite=satellite_union_mask,
        secondary=selection["secondary"], accepted=selection["accepted"],
        holes=selection["holes"], rs_ratio=selection["rs_ratio"],
        components=selection_core.label_image(components, source.data.shape),
        x0=source.x0, y0=source.y0, centre_crop_xy=np.asarray(source.centre_crop_xy),
    )

    def read_back(path, product):
        mask = np.zeros(product.data.shape, dtype=bool)
        for region in Regions.read(path, format="ds9"):
            pixel = region.to_pixel(product.wcs)
            rr, cc = draw_polygon(pixel.vertices.y - product.y0, pixel.vertices.x - product.x0,
                                  shape=mask.shape)
            mask[rr, cc] = True
        return mask

    roundtrip = {}
    for name, mask, product in (("source", selected_mask, source), ("lens", lens_final, lens_product)):
        back = read_back(output / f"{system_id}_{name}_v2.reg", product)
        roundtrip[name] = {
            "missing_after_readback_pix": int(np.sum(mask & ~back)),
            "extra_after_readback_pix": int(np.sum(back & ~mask)),
        }
        if roundtrip[name]["extra_after_readback_pix"] or roundtrip[name]["missing_after_readback_pix"]:
            warnings.append(
                f"{name} region read-back differs from the selection mask "
                f"(+{roundtrip[name]['extra_after_readback_pix']} / "
                f"-{roundtrip[name]['missing_after_readback_pix']} px)"
            )

    for method, item in lens_methods.items():
        if item["mask"] is not None:
            write_ds9_regions(
                output / f"{system_id}_lens_{method}_v2.reg", [item["mask"]],
                lens_product, [f"lens_{method}_v2"], ["magenta"],
            )
        item["evaluation"] = evaluate_lens_mask(item["mask"], None, None) if item["mask"] is not None else {}
    save_lens_method_comparison(
        output / f"{system_id}_lens_methods_v2_comparison.png",
        lens_product, source, lens_methods, None, None, anchors,
    )
    for method, item in lens_methods.items():
        write_lens_trace(output / f"{system_id}_lens_{method}_trace.csv", item["trials"])

    adaptation = {
        "version": REGION_METHOD,
        "system": system_id,
        "manual_region_used_during_parameter_selection": False,
        "lens_selection": {
            "production_method": args.lens_method,
            "available_methods": list(lens_methods),
            "source_anchor": anchor_metadata,
            "contact": lens_contact,
            "sextractor": lens_methods["sextractor"]["metadata"],
            "mge_isophote": lens_methods["mge_isophote"]["metadata"],
            "mge_fixed_flux90": lens_methods["mge_fixed_flux90"]["metadata"],
            "geometry": args.lens_geometry,
            "radial_scale": float(args.lens_scale),
            "raw_area_pix": int(lens_mask_raw.sum()),
            "scaled_area_pix": int(lens_mask_scaled.sum()),
            "production_area_pix": int(lens_mask.sum()),
            "final_area_after_rs_pix": int(lens_final.sum()),
            "point_reflection_mismatch_pix": lens_symmetry_mismatch_pix,
            "rs_symmetrisation_iterations": selection["lens_symmetrisation_iterations"],
        },
        "satellite_exclusion": satellite_metadata,
        "source_selection": {
            "definition": (
                "single selection: every component (or configured merge group) grown "
                "by hysteresis with an empty lens mask (growth creating self-reflection "
                "overlap rejected), r_s-classified against the lens aperture, carved "
                "against other components (uncarved 4-sigma segmentation) "
                "and satellites (direct and reflected, continuous unit cells) and against "
                "its own reflection, largest 4-connected piece, hard gates, maximum "
                "effective S/N F_diff / sqrt(sigma_stat^2 + (eps * |M_lens|)^2)"
            ),
            "ranking_metric": "effective_snr",
            "effective_snr_epsilon": selection_core.EFFECTIVE_SNR_EPSILON,
            "detect_sigma": selection_core.ANCHOR_DETECT_SIGMA,
            "grow_sigma_ratio": selection_core.SOURCE_GROW_SIGMA_RATIO,
            "max_grow_area_ratio": None,
            "rs_threshold": selection_core.RS_THRESHOLD,
            "merge_group_1based": merge_group,
            "hard_gates": {
                "minimum_area_pix": selection_core.MIN_SOURCE_AREA_PIX,
                "minimum_integrated_snr": selection_core.MIN_SOURCE_INTEGRATED_SNR,
                "minimum_carving_snr_retention": selection_core.MIN_CARVE_SNR_RETENTION,
                "contains_lens_centre": False,
            },
            "display_only": {
                "mirror_asymmetry_reference": MIN_MIRROR_ASYMMETRY,
                "neighbour_flux_change_reference": MAX_NEIGHBOUR_FLUX_CHANGE,
                "holes": "visualised and warned, not gated",
            },
            "candidates": selection["candidates"],
            "neighbour_threshold_diagnostic": neighbour,
        },
        "selected": {"sextractor": metrics},
    }
    (output / f"{system_id}_v2_adaptation.json").write_text(
        json.dumps(adaptation, indent=2, default=float) + "\n"
    )
    summary = {
        "system": system_id,
        "version": REGION_METHOD,
        "wcs_products_dir": str(products),
        "source_detection_product": str(source_path),
        "source_detection_definition": (
            "literal maximum-likelihood FitImaging subtracted_images_of_planes_list[-1] "
            "(lens plane, including modeled satellites, removed)"
        ),
        "method": "sextractor_fixed_4sigma_single_selection",
        "lens_method": args.lens_method,
        "lens_geometry": args.lens_geometry,
        "lens_scale": float(args.lens_scale),
        "lens_area_pix": int(lens_final.sum()),
        "lens_initial_area_pix": int(lens_mask.sum()),
        "lens_point_reflection_mismatch_pix": lens_symmetry_mismatch_pix,
        "lens_contact": lens_contact,
        "anchors": {
            "protected": len(anchors),
            "components": len(components),
        },
        "satellite_exclusion": satellite_metadata,
        "selected": {"sextractor": metrics},
        "readback": roundtrip,
        "warnings": warnings,
        "background_region": {
            "definition": (
                "180-degree point reflection of the final source mask about the same "
                "model lens-light centre used for source and lens selection"
            ),
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
            "background_pixels_outside_analysis_crop": int(
                background_mask_full.sum() - background_mask.sum()
            ),
            "background_pixels_outside_model_mask": int(
                np.sum(background_mask_full & ~valid_model_full)
            ),
            "final_ds9_region_clipped_by_model_mask": False,
            "source_background_overlap_pix": int(np.sum(source_mask_full & background_mask_full)),
            "background_lens_overlap_pix": background_lens_overlap_pix,
            "background_satellite_overlap_pix": background_satellite_overlap_pix,
            "background_secondary_overlap_pix": background_secondary_overlap_pix,
            "region_file": f"{system_id}_background_v2.reg",
        },
    }
    (output / f"{system_id}_v2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n"
    )
    if not args.keep_intermediates:
        shutil.rmtree(output / "parameter_search", ignore_errors=True)
    print(
        f"{system_id} Photo-lens: anchors={len(anchors)}/{len(components)} "
        f"lens={int(lens_mask.sum())}->{int(lens_final.sum())} "
        f"source={metrics['area_pix']} ({winner['candidate']}, intSNR="
        f"{metrics['integrated_snr']:.1f}) warnings={len(warnings)}"
    )


if __name__ == "__main__":
    main()
