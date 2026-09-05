from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.wcs.utils import proj_plane_pixel_scales
from regions import PixCoord, Regions
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

from .apertures import annulus_measurement, region_weight_image
from .config import PipelineConfig
from .diagnostics import differential_noise_diagnostics
from .io_utils import load_array, one_bb_file, ordered_nb_files, read_2d_celestial_image


@dataclass(frozen=True)
class ImageJob:
    band_type: str
    band: str
    path: Path
    exposure_time: float | None
    zero_point: float
    zero_point_error: float
    weight_path: Path | None = None


@dataclass
class PhotometryProducts:
    source: pd.DataFrame
    lens: pd.DataFrame
    exposure_diagnostics: pd.DataFrame
    mask_convergence: pd.DataFrame
    noise_diagnostics: pd.DataFrame


def _load_nb_aperture_centres(config: PipelineConfig) -> dict[str, dict]:
    path = config.nb_aperture_centre_path
    if path is None:
        return {}
    table = pd.read_csv(path)
    required = {
        "band", "ra_deg", "dec_deg", "bb_ra_deg", "bb_dec_deg",
        "minimum_at_search_edge",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"NB centre table lacks columns {sorted(missing)}: {path}")
    if table.band.duplicated().any():
        raise ValueError(f"Duplicate bands in NB centre table: {path}")
    expected = set(config.nb_names)
    actual = set(table.band.astype(str))
    if actual != expected:
        raise ValueError(
            f"NB centre table coverage mismatch; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    if table.minimum_at_search_edge.astype(bool).any():
        bad = table.loc[table.minimum_at_search_edge.astype(bool), "band"].tolist()
        raise ValueError(f"NB centre minima on search edge: {bad}")
    numeric = table[["ra_deg", "dec_deg", "bb_ra_deg", "bb_dec_deg"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite coordinate in NB centre table: {path}")
    if "apply_coordinate_correction" not in table.columns:
        # Backward compatibility for the unfiltered v1 experiment.
        table["apply_coordinate_correction"] = True
    return {str(row.band): row._asdict() for row in table.itertuples(index=False)}


def _translated_pixel_region(region, dx: float, dy: float):
    shifted = deepcopy(region)
    if hasattr(shifted, "vertices"):
        shifted.vertices = PixCoord(
            x=np.asarray(shifted.vertices.x, dtype=float) + dx,
            y=np.asarray(shifted.vertices.y, dtype=float) + dy,
        )
    elif hasattr(shifted, "center"):
        shifted.center = PixCoord(
            x=float(shifted.center.x) + dx,
            y=float(shifted.center.y) + dy,
        )
    else:
        raise TypeError(
            f"Cannot translate pixel region of type {shifted.__class__.__name__}"
        )
    return shifted


def _region_union_weights(regions_pix, image_shape, subpixels, mode="subpixels"):
    """Rasterize every DS9 component and return their clipped union."""
    union = np.zeros(image_shape, dtype=float)
    for region_pix in regions_pix:
        weights = region_weight_image(
            region_pix, image_shape, subpixels, mode=mode
        )
        if weights is None:
            return None
        union = np.maximum(union, weights)
    return union


def _nb_pixel_shift(image_wcs, centre: dict) -> tuple[float, float]:
    bb_x, bb_y = image_wcs.world_to_pixel_values(
        float(centre["bb_ra_deg"]), float(centre["bb_dec_deg"])
    )
    nb_x, nb_y = image_wcs.world_to_pixel_values(
        float(centre["ra_deg"]), float(centre["dec_deg"])
    )
    return float(nb_x - bb_x), float(nb_y - bb_y)


def _north_up_payload(payload: dict) -> dict:
    """Reproject a plotting payload to North-up, East-left coordinates.

    This is intentionally visualization-only.  Photometry is always evaluated
    on each exposure's native pixels before this function is called.
    """
    source_wcs = payload["image_wcs"]
    target_wcs, shape_out = find_optimal_celestial_wcs(
        [(payload["image"], source_wcs)], auto_rotate=False
    )
    oriented = dict(payload)
    oriented["image"], _ = reproject_interp(
        (payload["image"], source_wcs), target_wcs, shape_out=shape_out
    )
    for key in ("source_weights", "background_weights", "lens_weights"):
        values, _ = reproject_interp(
            (payload[key], source_wcs), target_wcs,
            shape_out=shape_out, order="bilinear",
        )
        oriented[key] = np.nan_to_num(values, nan=0.0)
    annulus, _ = reproject_interp(
        (payload["annulus_mask"].astype(float), source_wcs),
        target_wcs,
        shape_out=shape_out,
        order="nearest-neighbor",
    )
    oriented["annulus_mask"] = np.nan_to_num(annulus, nan=0.0) > 0.5
    lens_world = source_wcs.pixel_to_world_values(*payload["lens_xy"])
    oriented["lens_xy"] = tuple(
        float(value) for value in target_wcs.world_to_pixel_values(*lens_world)
    )
    oriented["image_wcs"] = target_wcs
    return oriented


def _draw_selection(axis, band: str, payload: dict | None) -> None:
    """Draw one representative exposure without changing photometry masks."""
    if payload is None:
        axis.text(
            0.5, 0.5, "no usable image", ha="center", va="center",
            transform=axis.transAxes,
        )
        axis.set_title(band)
        axis.set_axis_off()
        return

    image = payload["image"]
    finite = image[np.isfinite(image)]
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [40, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = np.nanmin(finite), np.nanmax(finite)
        axis.imshow(image, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)
    else:
        axis.imshow(np.zeros_like(image), cmap="bone", origin="lower")
    for weights, color in (
        (payload["source_weights"], "cyan"),
        (payload["background_weights"], "orange"),
        (payload["lens_weights"], "magenta"),
        (payload["annulus_mask"].astype(float), "yellow"),
    ):
        if np.any(weights > 0.5):
            axis.contour(weights, levels=[0.5], colors=[color], linewidths=0.8)
    axis.plot(
        payload["lens_xy"][0], payload["lens_xy"][1], marker="+",
        color="red", markersize=7, markeredgewidth=1.2,
    )
    suffix = "" if payload["photometry_valid"] else " (geometry only)"
    axis.set_title(f"{band}{suffix}  [N up, E left]", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def _save_selection_figures(config: PipelineConfig, payloads: dict[str, dict]) -> None:
    """Save one image per band plus fixed-size NB and BB overview figures."""
    payloads = {
        band: _north_up_payload(payload)
        for band, payload in payloads.items()
    }
    selection_root = config.result_dir / "aperture_selection"
    directories = {
        "nb": selection_root / "nb",
        "bb": selection_root / "bb",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    legend_handles = [
        Line2D([0], [0], color="cyan", label="source aperture"),
        Line2D([0], [0], color="orange", label="background aperture"),
        Line2D([0], [0], color="magenta", label="lens aperture"),
        Line2D([0], [0], color="yellow", label="shared annulus"),
        Line2D(
            [0], [0], marker="+", color="red", linestyle="none",
            label="lens-region centre",
        ),
    ]
    for band in config.ordered_bands:
        kind = "nb" if band.startswith("NB") else "bb"
        fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
        _draw_selection(axis, band, payloads.get(band))
        if payloads.get(band) is not None:
            axis.legend(handles=legend_handles, loc="best", fontsize=7)
        fig.savefig(directories[kind] / f"{band}.png", dpi=160)
        plt.close(fig)

    overview_specs = [
        ("NB", config.nb_names, 8, 5, (15, 20)),
        ("BB", config.bb_names, 2, 3, (14, 8)),
    ]
    for label, bands, nrows, ncols, figsize in overview_specs:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        flat_axes = np.asarray(axes).ravel()
        for axis, band in zip(flat_axes, bands):
            _draw_selection(axis, band, payloads.get(band))
        for axis in flat_axes[len(bands):]:
            axis.set_axis_off()
        fig.suptitle(
            f"System {config.lens_id}: {label} aperture selections "
            "(one representative exposure per band)",
            fontsize=15,
        )
        fig.legend(
            handles=legend_handles, loc="lower center", ncol=5,
            bbox_to_anchor=(0.5, 0.005), fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.025, 1, 0.98))
        fig.savefig(
            config.result_dir / f"aperture_selection_{label}_{config.lens_id}.png",
            dpi=160,
        )
        plt.close(fig)


def _jobs(config: PipelineConfig) -> list[ImageJob]:
    jobs = []
    for band in config.bb_names:
        jobs.append(ImageJob(
            band_type="bb",
            band=band,
            path=one_bb_file(config.bb_system_dir, band),
            exposure_time=None,
            zero_point=1.0,
            zero_point_error=0.0,
        ))

    for band in config.nb_names:
        wave = band.removeprefix("NB")
        load_dir = config.nb_cutout_dir / wave / str(config.lens_id)
        if load_dir.is_dir():
            paths = ordered_nb_files(load_dir)
            if paths:
                exposure_times = load_array(load_dir / "exp_time.txt", len(paths), "exp_time")
                zero_points = load_array(load_dir / "zero_points.txt", len(paths), "zero_points")
                zero_point_errors = load_array(
                    load_dir / "zero_points_err.txt", len(paths), "zero_points_err"
                )
                if zero_points is None or zero_point_errors is None:
                    raise FileNotFoundError(f"NB calibration arrays missing: {load_dir}")
                for index, path in enumerate(paths):
                    jobs.append(ImageJob(
                        band_type="nb",
                        band=band,
                        path=path,
                        exposure_time=(
                            None if exposure_times is None else float(exposure_times[index])
                        ),
                        zero_point=float(zero_points[index]),
                        zero_point_error=float(zero_point_errors[index]),
                    ))
                continue

        # fixed042-coadd fallback: calibrated science plus inverse-variance map.
        coadd_dir = config.nb_coadd_dir / band
        science = coadd_dir / f"coadd_cutout_{config.lens_id}.fits"
        weight = coadd_dir / f"coadd_cutout_{config.lens_id}.weight.fits"
        if science.is_file() and weight.is_file():
            jobs.append(ImageJob(
                band_type="nb", band=band, path=science, exposure_time=None,
                zero_point=1.0, zero_point_error=0.0, weight_path=weight,
            ))
    return jobs


def _calibrate(flux: float, variance: float, job: ImageJob) -> tuple[float, float]:
    calibrated_flux = job.zero_point * flux
    calibrated_variance = (
        job.zero_point ** 2 * variance
        + flux ** 2 * job.zero_point_error ** 2
    )
    return calibrated_flux, calibrated_variance


def _valid_nb_joint(
    image: np.ndarray,
    source_weights: np.ndarray,
    background_weights: np.ndarray,
    annulus: dict,
) -> bool:
    source = source_weights > 0
    background = background_weights > 0
    annulus_mask = annulus["mask"]
    return bool(
        len(np.unique(image[source])) >= 0.9 * source.sum()
        and len(np.unique(image[background])) >= 0.9 * background.sum()
        and len(np.unique(image[annulus_mask])) >= 0.8 * annulus_mask.sum()
        and annulus_mask.sum() >= 0.9 * annulus["area"]
    )


def _valid_nb_lens(
    image: np.ndarray,
    lens_center_weights: np.ndarray,
    annulus: dict,
) -> bool:
    aperture = lens_center_weights > 0
    annulus_mask = annulus["mask"]
    return bool(
        len(np.unique(image[aperture])) >= 0.9 * aperture.sum()
        and len(np.unique(image[annulus_mask])) >= 0.8 * annulus_mask.sum()
        and annulus_mask.sum() >= 0.9 * annulus["area"]
    )


def _combine_exposures(rows: list[dict], target: str) -> pd.DataFrame:
    records = []
    for band, group in pd.DataFrame(rows).groupby("band", sort=False):
        flux = group[f"{target}_calibrated_flux"].to_numpy(dtype=float)
        variance = group[f"{target}_calibrated_variance"].to_numpy(dtype=float)
        good = np.isfinite(flux) & np.isfinite(variance) & (variance > 0)
        if not good.any():
            continue
        inverse = 1.0 / variance[good]
        records.append({
            "ref_id": str(group.lens_id.iloc[0]),
            "band": band,
            "flux": float(np.average(flux[good], weights=inverse)),
            "flux_error": float(np.sqrt(1.0 / inverse.sum())),
            "n_exposure": int(good.sum()),
        })
    return pd.DataFrame(records)


def _write_wide(long_table: pd.DataFrame, path: Path, config: PipelineConfig) -> None:
    row = {"lens_id": config.lens_id}
    for item in long_table.itertuples(index=False):
        row[item.band] = item.flux
        suffix = "err" if str(item.band).startswith("NB") else "_err"
        row[f"{item.band}{suffix}"] = item.flux_error
    pd.DataFrame([row]).set_index("lens_id").to_csv(path)


def run_photometry(config: PipelineConfig) -> PhotometryProducts:
    catalog = pd.read_csv(config.catalog_path)
    selected = catalog[catalog.lens_id == config.lens_id]
    if selected.empty:
        raise ValueError(f"Lens {config.lens_id} is absent from the catalogue")
    source_sky = list(Regions.read(config.source_region_path, format="ds9"))
    background_sky = list(Regions.read(config.background_region_path, format="ds9"))
    lens_sky = list(Regions.read(config.lens_region_path, format="ds9"))
    if not source_sky or not background_sky or not lens_sky:
        raise ValueError("Source, background, and lens region files must be non-empty")
    nb_aperture_centres = _load_nb_aperture_centres(config)

    source_rows: list[dict] = []
    lens_rows: list[dict] = []
    exposure_rows: list[dict] = []
    convergence_rows: list[dict] = []
    noise_rows: list[dict] = []
    selection_payloads: dict[str, dict] = {}
    config.result_dir.mkdir(parents=True, exist_ok=True)

    for job in _jobs(config):
        try:
            image, image_wcs, hdu_index = read_2d_celestial_image(job.path)
            variance_image = None
            if job.weight_path is not None:
                weight, weight_wcs, _ = read_2d_celestial_image(job.weight_path)
                if weight.shape != image.shape:
                    raise ValueError(
                        f"Weight/image shape mismatch: {job.weight_path} vs {job.path}"
                    )
                if not np.allclose(
                    weight_wcs.pixel_to_world_values(0.0, 0.0),
                    image_wcs.pixel_to_world_values(0.0, 0.0),
                    rtol=0.0, atol=1e-8,
                ):
                    raise ValueError(f"Weight/image WCS mismatch: {job.weight_path}")
                variance_image = np.full(weight.shape, np.nan, dtype=float)
                valid_weight = np.isfinite(weight) & (weight > 0.0)
                variance_image[valid_weight] = 1.0 / weight[valid_weight]
            if job.band_type == "bb":
                image = image * 10 ** (
                    (config.paus_zero_point - config.bb_zero_point) / 2.5
                )

            source_pix = [region.to_pixel(image_wcs) for region in source_sky]
            background_pix = [region.to_pixel(image_wcs) for region in background_sky]
            lens_pix = [region.to_pixel(image_wcs) for region in lens_sky]
            shift_dx = 0.0
            shift_dy = 0.0
            centre_row = nb_aperture_centres.get(job.band)
            apply_nb_shift = bool(
                centre_row is not None
                and centre_row["apply_coordinate_correction"]
            )
            if apply_nb_shift:
                shift_dx, shift_dy = _nb_pixel_shift(image_wcs, centre_row)
                source_pix = [
                    _translated_pixel_region(region, shift_dx, shift_dy)
                    for region in source_pix
                ]
                background_pix = [
                    _translated_pixel_region(region, shift_dx, shift_dy)
                    for region in background_pix
                ]
            # Use the lens aperture itself as the positional source of truth.
            # Circle/ellipse regions expose ``center``; the automated v2 lens
            # aperture is a polygon, for which the vertex centroid supplies the
            # shared annulus centre and display marker. Photometry always uses
            # the complete region, not this representative point.
            lens_reference = lens_pix[0]
            if hasattr(lens_reference, "center"):
                lens_xy = (
                    float(lens_reference.center.x), float(lens_reference.center.y)
                )
            elif hasattr(lens_reference, "vertices"):
                lens_xy = (
                    float(np.mean(lens_reference.vertices.x)),
                    float(np.mean(lens_reference.vertices.y)),
                )
            else:
                raise TypeError(
                    "Lens region must define a centre or polygon vertices: "
                    f"{config.lens_region_path}"
                )

            source_weights = _region_union_weights(
                source_pix, image.shape, config.aperture_subpixels
            )
            background_weights = _region_union_weights(
                background_pix, image.shape, config.aperture_subpixels
            )
            lens_weights = _region_union_weights(
                lens_pix, image.shape, config.aperture_subpixels
            )
            source_center = _region_union_weights(
                source_pix, image.shape, config.aperture_subpixels, mode="center"
            )
            background_center = _region_union_weights(
                background_pix, image.shape, config.aperture_subpixels, mode="center"
            )
            lens_center = _region_union_weights(
                lens_pix, image.shape, config.aperture_subpixels, mode="center"
            )
            source_convergence = _region_union_weights(
                source_pix, image.shape, config.convergence_subpixels
            )
            background_convergence = _region_union_weights(
                background_pix, image.shape, config.convergence_subpixels
            )
            if any(x is None for x in (
                source_weights, background_weights, lens_weights,
                source_center, background_center, lens_center,
                source_convergence, background_convergence,
            )):
                warnings.warn(f"Region outside image; skipping {job.path.name}")
                continue

            overlap = float(np.minimum(source_weights, background_weights).sum())
            if overlap > 1e-6:
                raise ValueError(
                    f"Source/background overlap by {overlap:.6g} pixels: {job.path}"
                )
            annulus = annulus_measurement(image, lens_xy)
            valid_source = (
                job.band_type != "nb"
                or _valid_nb_joint(image, source_weights, background_weights, annulus)
            )
            valid_lens = (
                job.band_type != "nb"
                or _valid_nb_lens(image, lens_center, annulus)
            )

            # Keep the first geometrically valid image, but replace it with the
            # first exposure that is also admitted to both photometry streams.
            photometry_valid = bool(valid_source and valid_lens)
            current = selection_payloads.get(job.band)
            if current is None or (
                photometry_valid and not current["photometry_valid"]
            ):
                selection_payloads[job.band] = {
                    "image": image.copy(),
                    "image_wcs": image_wcs,
                    "source_weights": source_weights.copy(),
                    "background_weights": background_weights.copy(),
                    "lens_weights": lens_weights.copy(),
                    "annulus_mask": annulus["mask"].copy(),
                    "lens_xy": lens_xy,
                    "photometry_valid": photometry_valid,
                    "image_file": job.path.name,
                }

            pixel_variance = annulus["pixel_sigma"] ** 2
            background_level_variance = (
                (np.pi / 2.0) * pixel_variance / max(annulus["samples"], 1)
            )

            if valid_source:
                source_area = float(source_weights.sum())
                background_area = float(background_weights.sum())
                area_difference = source_area - background_area
                source_sum = float(np.sum(image * source_weights))
                background_sum = float(np.sum(image * background_weights))
                residual_flux = (
                    source_sum - background_sum
                    - area_difference * annulus["level"]
                )
                combined_weight2 = source_weights ** 2 + background_weights ** 2
                if variance_image is None:
                    aperture_variance = pixel_variance * float(combined_weight2.sum())
                else:
                    valid_variance = np.isfinite(variance_image) & (combined_weight2 > 0)
                    if not np.any(valid_variance):
                        continue
                    aperture_variance = float(np.sum(
                        variance_image[valid_variance] * combined_weight2[valid_variance]
                    ))
                shared_variance = area_difference ** 2 * background_level_variance
                poisson_variance = 0.0
                if job.exposure_time is not None:
                    source_net = max(
                        source_sum - source_area * annulus["level"], 0.0
                    )
                    background_net = max(
                        background_sum - background_area * annulus["level"], 0.0
                    )
                    poisson_variance = (
                        source_net + background_net
                    ) / max(job.exposure_time, 1e-12)
                residual_variance = (
                    aperture_variance + shared_variance + poisson_variance
                )
                if job.exposure_time is None:
                    residual_variance *= config.coadd_error_factor ** 2
                calibrated_flux, calibrated_variance = _calibrate(
                    residual_flux, residual_variance, job
                )
                source_rows.append({
                    "lens_id": config.lens_id,
                    "band": job.band,
                    "source_calibrated_flux": calibrated_flux,
                    "source_calibrated_variance": calibrated_variance,
                })

                def difference_flux(sw, bw):
                    sa, ba = float(sw.sum()), float(bw.sum())
                    return float(
                        np.sum(image * sw) - np.sum(image * bw)
                        - (sa - ba) * annulus["level"]
                    )

                center_flux = difference_flux(source_center, background_center)
                convergence_flux = difference_flux(
                    source_convergence, background_convergence
                )
                center_flux_cal = job.zero_point * center_flux
                convergence_flux_cal = job.zero_point * convergence_flux
                convergence_rows.append({
                    "lens_id": config.lens_id,
                    "band": job.band,
                    "band_type": job.band_type,
                    "image_file": job.path.name,
                    "aperture_subpixels": config.aperture_subpixels,
                    "convergence_subpixels": config.convergence_subpixels,
                    "area_source_center": float(source_center.sum()),
                    "area_background_center": float(background_center.sum()),
                    "area_source_sub11": source_area,
                    "area_background_sub11": background_area,
                    "dc_fraction_sub11": abs(area_difference) / max(source_area, 1e-12),
                    "flux_center": center_flux_cal,
                    "flux_sub11": calibrated_flux,
                    "flux_sub41": convergence_flux_cal,
                    "delta_11_41_over_error": abs(
                        calibrated_flux - convergence_flux_cal
                    ) / np.sqrt(max(calibrated_variance, 1e-300)),
                })

                pixel_scale = float(
                    np.mean(proj_plane_pixel_scales(image_wcs)) * 3600.0
                )
                noise = differential_noise_diagnostics(
                    image, source_weights, background_weights, lens_xy, pixel_scale
                )
                empirical_variance = (
                    noise["sigma_full"] ** 2
                    + shared_variance + poisson_variance
                )
                noise_rows.append({
                    "lens_id": config.lens_id,
                    "band": job.band,
                    "band_type": job.band_type,
                    "image_file": job.path.name,
                    "pixel_scale_arcsec": pixel_scale,
                    "analytic_error": np.sqrt(max(residual_variance, 0.0)),
                    "empirical_over_analytic": (
                        np.sqrt(empirical_variance / residual_variance)
                        if residual_variance > 0 else np.nan
                    ),
                    "shared_annulus_variance": shared_variance,
                    "poisson_variance": poisson_variance,
                    **noise,
                })
                exposure_rows.append({
                    "lens_id": config.lens_id,
                    "band": job.band,
                    "band_type": job.band_type,
                    "image_file": job.path.name,
                    "image_hdu": hdu_index,
                    "relative_image_path": config.relative_to_project(job.path),
                    "nb_aperture_shift_applied": apply_nb_shift,
                    "nb_aperture_shift_rejected_by_threshold": bool(
                        centre_row is not None and not apply_nb_shift
                    ),
                    "nb_measured_centre_separation_arcsec": (
                        float(centre_row["separation_from_bb_arcsec"])
                        if centre_row is not None else np.nan
                    ),
                    "nb_max_shift_arcsec": (
                        float(centre_row["max_coordinate_correction_arcsec"])
                        if centre_row is not None
                        and np.isfinite(centre_row.get(
                            "max_coordinate_correction_arcsec", np.nan
                        ))
                        else np.nan
                    ),
                    "nb_aperture_shift_dx_pix": shift_dx,
                    "nb_aperture_shift_dy_pix": shift_dy,
                    "nb_antisymmetry_ra_deg": (
                        float(centre_row["ra_deg"]) if centre_row is not None else np.nan
                    ),
                    "nb_antisymmetry_dec_deg": (
                        float(centre_row["dec_deg"]) if centre_row is not None else np.nan
                    ),
                    "source_area": source_area,
                    "background_area": background_area,
                    "overlap_weight": overlap,
                    "annulus_background_level": annulus["level"],
                    "annulus_pixel_std": annulus["pixel_sigma"],
                    "annulus_samples": annulus["samples"],
                    "aperture_pixel_variance": aperture_variance,
                    "shared_annulus_variance": shared_variance,
                    "poisson_variance": poisson_variance,
                    "residual_flux_before_calibration": residual_flux,
                    "residual_variance_before_calibration": residual_variance,
                    "zero_point": job.zero_point,
                    "zero_point_error": job.zero_point_error,
                    "calibrated_residual_flux": calibrated_flux,
                    "calibrated_residual_error": np.sqrt(calibrated_variance),
                })

            if valid_lens:
                lens_area = float(lens_weights.sum())
                lens_sum = float(np.sum(image * lens_weights))
                lens_flux = lens_sum - lens_area * annulus["level"]
                if variance_image is None:
                    aperture_map_variance = lens_area * pixel_variance
                else:
                    valid_variance = np.isfinite(variance_image) & (lens_weights > 0)
                    if not np.any(valid_variance):
                        continue
                    aperture_map_variance = float(np.sum(
                        variance_image[valid_variance] * lens_weights[valid_variance] ** 2
                    ))
                lens_variance = aperture_map_variance + (
                    (np.pi / 2.0) * lens_area ** 2 / max(annulus["samples"], 1)
                ) * pixel_variance
                if job.exposure_time is not None:
                    lens_variance += max(lens_flux, 0.0) / max(
                        job.exposure_time, 1e-12
                    )
                else:
                    lens_variance *= config.coadd_error_factor ** 2
                lens_cal_flux, lens_cal_variance = _calibrate(
                    lens_flux, lens_variance, job
                )
                lens_rows.append({
                    "lens_id": config.lens_id,
                    "band": job.band,
                    "lens_calibrated_flux": lens_cal_flux,
                    "lens_calibrated_variance": lens_cal_variance,
                })
        except (ValueError, IndexError) as error:
            warnings.warn(f"Skipping {job.band} {job.path.name}: {error}")

    if not source_rows or not lens_rows:
        raise RuntimeError("Photometry produced no source or lens measurements")

    source = _combine_exposures(source_rows, "source")
    lens = _combine_exposures(lens_rows, "lens")
    exposure = pd.DataFrame(exposure_rows)
    convergence = pd.DataFrame(convergence_rows)
    noise = pd.DataFrame(noise_rows)

    _save_selection_figures(config, selection_payloads)
    source.to_csv(config.result_dir / f"photometry_source_long_{config.lens_id}.csv", index=False)
    lens.to_csv(config.result_dir / f"photometry_lens_long_{config.lens_id}.csv", index=False)
    _write_wide(
        source,
        config.result_dir / f"source_photometry_all_bands_{config.lens_id}.csv",
        config,
    )
    _write_wide(
        lens,
        config.result_dir / f"source_photometry_{config.lens_id}_lens.csv",
        config,
    )
    exposure.to_csv(
        config.result_dir / f"source_minus_background_exposure_diagnostics_{config.lens_id}.csv",
        index=False,
    )
    convergence.to_csv(
        config.result_dir / f"mask_convergence_{config.lens_id}.csv", index=False
    )
    noise.to_csv(
        config.result_dir / f"differential_noise_diagnostics_{config.lens_id}.csv",
        index=False,
    )
    return PhotometryProducts(source, lens, exposure, convergence, noise)
