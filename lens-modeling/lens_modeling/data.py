from __future__ import annotations

from dataclasses import dataclass
import ast
from pathlib import Path
import re

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import autolens as al

from .config import RunConfig


@dataclass
class DatasetBundle:
    dataset: al.Imaging
    dataset_masked: al.Imaging
    data_native: np.ndarray
    variance_native: np.ndarray
    detector_mask_native: np.ndarray
    wcs_header: fits.Header
    light_centre_yx: tuple[float, float]
    mass_centre_yx: tuple[float, float]
    mask_centre_yx: tuple[float, float]
    mask_radius: float
    image_positions: list[tuple[float, float]]
    satellite_positions: list[tuple[float, float]]


def _central_odd_crop(array: np.ndarray, size: int) -> np.ndarray:
    size = min(int(size), *array.shape)
    if size % 2 == 0:
        size -= 1
    y0 = (array.shape[0] - size) // 2
    x0 = (array.shape[1] - size) // 2
    return np.asarray(array[y0 : y0 + size, x0 : x0 + size], dtype=float)


def _autolens_yx_from_world(
    wcs: WCS, shape: tuple[int, int], ra_deg: float, dec_deg: float, pixel_scale: float
) -> tuple[float, float]:
    x_pixel, y_pixel = wcs.world_to_pixel_values(float(ra_deg), float(dec_deg))
    centre_y = 0.5 * (shape[0] - 1)
    centre_x = 0.5 * (shape[1] - 1)
    return (
        float((y_pixel - centre_y) * pixel_scale),
        float((x_pixel - centre_x) * pixel_scale),
    )


def _read_positions(path) -> dict:
    text = path.read_text(encoding="utf-8")
    values = {"IMAGE_POSITIONS": [], "SATELLITE_POSITIONS": []}
    for name in ("IMAGE_POSITIONS", "SATELLITE_POSITIONS"):
        match = re.search(rf"{name}\s*=\s*(\[[\s\S]*?\])", text)
        if match:
            values[name] = [tuple(map(float, item)) for item in ast.literal_eval(match.group(1))]
    for name in ("MASK_RADIUS",):
        match = re.search(rf"{name}\s*=\s*([^\n#]+)", text)
        if match:
            values[name] = float(ast.literal_eval(match.group(1).strip()))
    for name in (
        "MASK_CENTRE", "MASK_CENTER", "LENS_LIGHT_CENTRE", "LENS_LIGHT_CENTER",
        "LIGHT_CENTRE", "LIGHT_CENTER", "MASS_CENTRE", "MASS_CENTER",
    ):
        match = re.search(rf"{name}\s*=\s*([^\n#]+)", text)
        if match:
            value = ast.literal_eval(match.group(1).strip())
            if value is not None:
                values[name] = tuple(map(float, value[:2]))
    return values


def resolve_uncertainty_input(config: RunConfig) -> tuple[str, Path, int]:
    """Return the configured detector uncertainty as variance or inverse variance.

    HSC products use variance maps; CFHT products use inverse-variance weight
    maps.  Explicit configuration wins and standardized sibling directories
    provide a migration-safe fallback.
    """
    if config.variance_cutout is not None and config.weight_cutout is not None:
        raise ValueError("Configure only one of variance_cutout and weight_cutout")
    if config.weight_cutout is not None:
        return "weight", config.weight_cutout, config.weight_hdu
    if config.variance_cutout is not None:
        return "variance", config.variance_cutout, config.variance_hdu
    science = config.science_cutout
    if science.parent.name.lower() == "science":
        root = science.parent.parent
        weight = root / "weight" / science.name
        variance = root / "variance" / science.name
        if weight.is_file():
            return "weight", weight, 0
        if variance.is_file():
            return "variance", variance, 0
    return "variance", science, config.variance_hdu


def load_dataset(config: RunConfig) -> DatasetBundle:
    with fits.open(config.science_cutout) as hdul:
        data = np.asarray(hdul[config.science_hdu].data, dtype=float)
        header = hdul[config.science_hdu].header.copy()
    uncertainty_kind, uncertainty_path, uncertainty_hdu = resolve_uncertainty_input(config)
    with fits.open(uncertainty_path) as hdul:
        uncertainty = np.asarray(hdul[uncertainty_hdu].data, dtype=float)
    if uncertainty_kind == "weight":
        good = np.isfinite(uncertainty) & (uncertainty > 0.0)
        variance = np.full(uncertainty.shape, np.nan, dtype=float)
        np.divide(1.0, uncertainty, out=variance, where=good)
    else:
        variance = uncertainty
    if data.shape != variance.shape:
        raise ValueError(
            f"science/{uncertainty_kind} shapes differ: {data.shape} versus {variance.shape}"
        )
    if data.ndim != 2 or not np.all(np.isfinite(data)):
        raise ValueError("science cutout must be a finite 2D array")

    positions = _read_positions(config.positions_file)
    if "MASK_RADIUS" not in positions:
        raise ValueError(
            f"MASK_RADIUS must be defined in the positions file: {config.positions_file}"
        )
    mask_radius = float(positions["MASK_RADIUS"])
    centre_value = positions.get(
        "MASK_CENTRE",
        positions.get(
            "MASK_CENTER",
            positions.get("LENS_LIGHT_CENTRE", positions.get("LENS_LIGHT_CENTER")),
        ),
    )
    mask_centre_yx = tuple(centre_value or (0.0, 0.0))
    light_centre_yx = tuple(
        positions.get("LIGHT_CENTRE", positions.get("LIGHT_CENTER", mask_centre_yx))
    )
    mass_centre_yx = tuple(
        positions.get("MASS_CENTRE", positions.get("MASS_CENTER", light_centre_yx))
    )
    positive = variance[np.isfinite(variance) & (variance > 0.0)]
    if positive.size == 0:
        raise ValueError(
            f"{uncertainty_kind} map has no finite positive pixels: {uncertainty_path}"
        )
    variance_safe = np.where(
        np.isfinite(variance) & (variance > 0.0), variance, np.median(positive)
    )
    noise = np.sqrt(variance_safe)

    psf = _central_odd_crop(np.asarray(fits.getdata(config.psf), dtype=float), config.psf_size)
    if not np.all(np.isfinite(psf)) or psf.sum() <= 0.0:
        raise ValueError("PSF is non-finite or has non-positive total flux")
    psf /= psf.sum()

    # AutoLens ``from_fits`` flips FITS row order. Reproduce that official
    # convention so positions_gui (y positive toward increasing FITS row) is
    # directly valid in AutoLens coordinates; products are flipped back to FITS.
    data_model = np.flipud(data)
    noise_model = np.flipud(noise)
    variance_model = np.flipud(variance)
    dataset = al.Imaging(
        data=al.Array2D.no_mask(values=data_model, pixel_scales=config.pixel_scale),
        noise_map=al.Array2D.no_mask(values=noise_model, pixel_scales=config.pixel_scale),
        psf=al.Kernel2D.no_mask(
            values=psf, pixel_scales=config.pixel_scale, normalize=True
        ),
    )
    mask = al.Mask2D.circular(
        shape_native=data.shape,
        pixel_scales=config.pixel_scale,
        radius=mask_radius,
        centre=mask_centre_yx,
    )
    invalid_variance = ~np.isfinite(variance_model) | (variance_model <= 0.0)
    mask = al.Mask2D(
        mask=np.asarray(mask).astype(bool) | invalid_variance,
        pixel_scales=config.pixel_scale,
    )
    # Preserve the mask on the original detector grid.  ``apply_mask`` may
    # trim or FFT-pad the dataset, so its later native shape is not a reliable
    # description of where pixels came from in the detector cutout.
    detector_mask_native = np.flipud(np.asarray(mask, dtype=bool))
    dataset_masked = dataset.apply_mask(mask=mask)
    if config.over_sample:
        # The GUI positions use the same AutoLens (y, x) arcsec convention as
        # the masked dataset.  Centre the radial sub-grid allocation on the
        # measured lens-light centre instead of the array origin, which can be
        # substantially offset for asymmetric cutouts.
        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=dataset_masked.grid,
            sub_size_list=[4, 2, 1],
            radial_list=[0.8, 1.5],
            centre_list=[light_centre_yx],
        )
        dataset_masked = dataset_masked.apply_over_sampling(
            over_sample_size_lp=over_sample_size
        )

    return DatasetBundle(
        dataset=dataset,
        dataset_masked=dataset_masked,
        data_native=data,
        variance_native=variance,
        detector_mask_native=detector_mask_native,
        wcs_header=header,
        light_centre_yx=light_centre_yx,
        mass_centre_yx=mass_centre_yx,
        mask_centre_yx=mask_centre_yx,
        mask_radius=mask_radius,
        image_positions=positions["IMAGE_POSITIONS"],
        satellite_positions=positions["SATELLITE_POSITIONS"],
    )
