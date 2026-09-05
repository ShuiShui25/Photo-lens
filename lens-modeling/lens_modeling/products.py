from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import autolens as al
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import convolve


MGE_SCHEMA = "lens-mge-v2-seven-column-full-detector"
DETECTOR_MAPPING_SCHEMA = "autolens-y-up_to_fits-row-up-v2"


def _native(array) -> np.ndarray:
    return np.asarray(getattr(array, "native", array), dtype=float)


def _basis_centre_yx(basis) -> tuple[float, float]:
    centre = basis.profile_list[0].centre
    if isinstance(centre, (tuple, list)):
        return float(centre[0]), float(centre[1])
    return float(centre.centre_0), float(centre.centre_1)


def autolens_to_fits_pixel(
    centre_yx: tuple[float, float], shape: tuple[int, int], pixel_scale: float
) -> tuple[float, float]:
    """Map AutoLens (y,x) arcsec to zero-indexed FITS-array (x,y) pixels.

    The modeling loader flips the FITS rows before constructing an AutoLens
    array.  AutoLens y therefore increases with the original FITS row number.
    This function is the exact inverse convention and works for every odd/even
    detector shape without reference to a trimmed or FFT-padded fit array.
    """
    centre_y, centre_x = centre_yx
    ny, nx = shape
    return (
        0.5 * (nx - 1) + centre_x / pixel_scale,
        0.5 * (ny - 1) + centre_y / pixel_scale,
    )


def fits_pixel_to_autolens(
    pixel_xy: tuple[float, float], shape: tuple[int, int], pixel_scale: float
) -> tuple[float, float]:
    x_pixel, y_pixel = pixel_xy
    ny, nx = shape
    return (
        (y_pixel - 0.5 * (ny - 1)) * pixel_scale,
        (x_pixel - 0.5 * (nx - 1)) * pixel_scale,
    )


def _primary_header(header: fits.Header) -> fits.Header:
    result = header.copy()
    for key in ("XTENSION", "PCOUNT", "GCOUNT", "EXTNAME", "EXTVER"):
        result.pop(key, None)
    return result


def _atomic_fits_write(hdul: fits.HDUList, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        hdul.writeto(temporary_path, overwrite=True)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_image(path: Path, data: np.ndarray, header: fits.Header, **metadata) -> None:
    output_header = _primary_header(header)
    for key, value in metadata.items():
        output_header[key] = value
    _atomic_fits_write(
        fits.HDUList(
            [fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=output_header)]
        ),
        path,
    )


def _write_exact_image(
    path: Path, data: np.ndarray, header: fits.Header, **metadata
) -> None:
    """Write an exact-fit product without reducing its floating-point precision."""
    output_header = _primary_header(header)
    for key, value in metadata.items():
        output_header[key] = value
    _atomic_fits_write(
        fits.HDUList(
            [fits.PrimaryHDU(data=np.asarray(data, dtype=np.float64), header=output_header)]
        ),
        path,
    )


def fit_native_detector_mapping(fit, bundle) -> dict:
    """Map an AutoArray trimmed/FFT-padded native array to detector pixels.

    ``Imaging.apply_mask`` may trim and then FFT-pad the original detector
    array.  When the detector and padded shapes have different parity, the
    geometry of a newly generated full-detector grid is offset by half a pixel
    from this integer array crop.  Therefore the placement must be inferred
    from the actual cropped science array, not from the nominal grid origin.

    The central integer crop is accepted only if every unmasked science pixel
    is exactly the corresponding original detector pixel.  This assertion is
    the guard against silently exporting a shifted model.
    """
    fit_data = _native(fit.dataset.data)
    fit_mask = np.asarray(fit.dataset.mask.native, dtype=bool)
    detector_model_order = np.flipud(np.asarray(bundle.data_native, dtype=float))
    if fit_data.shape != fit_mask.shape:
        raise RuntimeError("fit data and mask native shapes differ")
    if any(fit_size > full_size for fit_size, full_size in zip(
        fit_data.shape, detector_model_order.shape
    )):
        raise RuntimeError("fit native array is larger than the detector array")

    y0 = (detector_model_order.shape[0] - fit_data.shape[0]) // 2
    x0 = (detector_model_order.shape[1] - fit_data.shape[1]) // 2
    detector_crop = detector_model_order[
        y0 : y0 + fit_data.shape[0], x0 : x0 + fit_data.shape[1]
    ]
    valid = ~fit_mask
    if not np.array_equal(detector_crop[valid], fit_data[valid]):
        difference = detector_crop[valid] - fit_data[valid]
        raise RuntimeError(
            "AutoLens native-to-detector crop validation failed: "
            f"max_abs={np.max(np.abs(difference)):.6g}, "
            f"rms={np.sqrt(np.mean(difference**2)):.6g}"
        )
    return {
        "y0_model_order": int(y0),
        "x0": int(x0),
        "shape": tuple(map(int, fit_data.shape)),
        "valid": valid,
        "science_array_equal": True,
    }


def fit_array_to_detector_fits(array, fit, bundle, mapping: dict) -> np.ndarray:
    """Place one exact AutoLens native fit array on the original FITS grid."""
    native = _native(array)
    if native.shape != mapping["shape"]:
        raise RuntimeError(
            f"fit product shape {native.shape} does not match fit data "
            f"shape {mapping['shape']}"
        )
    model_order = np.full(bundle.data_native.shape, np.nan, dtype=np.float64)
    y0 = mapping["y0_model_order"]
    x0 = mapping["x0"]
    target = model_order[y0 : y0 + native.shape[0], x0 : x0 + native.shape[1]]
    target[mapping["valid"]] = native[mapping["valid"]]
    detector_fits = np.flipud(model_order)

    # Exact in-memory round trip: every modeled pixel must return to the same
    # AutoLens native index before any FITS file is written.
    recovered = np.flipud(detector_fits)[
        y0 : y0 + native.shape[0], x0 : x0 + native.shape[1]
    ]
    if not np.array_equal(recovered[mapping["valid"]], native[mapping["valid"]]):
        raise RuntimeError("exact fit-array detector round trip failed")
    return detector_fits


def write_exact_fit_plane_products(
    fit, bundle, config, *, stage: int, output_name: str, output_dir: Path | None = None
) -> dict[str, Path]:
    """Export exact ``FitImaging`` foreground-plane and subtracted arrays.

    No profile is re-evaluated and no external convolution is performed.  The
    products are the literal arrays used by AutoLens for the maximum-likelihood
    fit, placed back into their original detector pixels.
    """
    output = Path(output_dir) if output_dir is not None else config.output_dir / output_name
    output.mkdir(parents=True, exist_ok=True)
    mapping = fit_native_detector_mapping(fit=fit, bundle=bundle)

    plane_native = fit.model_images_of_planes_list[0]
    subtracted_native = fit.subtracted_images_of_planes_list[-1]
    fit_data_native = _native(fit.dataset.data)
    plane_values = _native(plane_native)
    subtracted_values = _native(subtracted_native)
    valid = mapping["valid"]
    relation_error = (
        fit_data_native[valid] - plane_values[valid] - subtracted_values[valid]
    )
    if not np.allclose(relation_error, 0.0, atol=1.0e-12, rtol=0.0):
        raise RuntimeError(
            "FitImaging plane-subtraction identity failed: "
            f"max_abs={np.max(np.abs(relation_error)):.6g}"
        )

    plane_fits = fit_array_to_detector_fits(
        plane_native, fit=fit, bundle=bundle, mapping=mapping
    )
    subtracted_fits = fit_array_to_detector_fits(
        subtracted_native, fit=fit, bundle=bundle, mapping=mapping
    )

    # Also export the exact main-lens galaxy image used in the fit.  This keeps
    # lens-region construction and the point-fraction denominator on the same
    # pixel convention as the exact foreground-plane subtraction.
    main_lenses = [
        galaxy
        for galaxy in fit.tracer.galaxies
        if hasattr(galaxy, "mass")
        and hasattr(galaxy, "bulge")
        and np.isclose(float(galaxy.redshift), float(config.z_lens), atol=1.0e-8)
    ]
    if len(main_lenses) != 1:
        raise RuntimeError(
            f"Expected one light-bearing main lens in fit, found {len(main_lenses)}"
        )
    galaxy_images = fit.galaxy_model_image_dict
    main_lens_native = galaxy_images[main_lenses[0]]
    main_lens_fits = fit_array_to_detector_fits(
        main_lens_native, fit=fit, bundle=bundle, mapping=mapping
    )

    common = {
        "STAGE": int(stage),
        "FITEXACT": True,
        "MAPSCHEM": "FITNATIVE1",
        "FITY0": int(mapping["y0_model_order"]),
        "FITX0": int(mapping["x0"]),
        "FITNY": int(mapping["shape"][0]),
        "FITNX": int(mapping["shape"][1]),
    }
    plane_path = output / f"{config.system}_autolens_lens_plane_model_wcs.fits"
    subtracted_path = (
        output / f"{config.system}_autolens_lens_light_subtracted_wcs.fits"
    )
    main_lens_path = output / f"{config.system}_lens_light_model_wcs.fits"
    _write_exact_image(
        plane_path,
        plane_fits,
        bundle.wcs_header,
        PRODUCT="AUTOLENS_LENS_PLANE_MODEL",
        **common,
    )
    _write_exact_image(
        subtracted_path,
        subtracted_fits,
        bundle.wcs_header,
        PRODUCT="AUTOLENS_LENS_LIGHT_SUBTRACTED",
        **common,
    )
    _write_exact_image(
        main_lens_path,
        main_lens_fits,
        bundle.wcs_header,
        PRODUCT="AUTOLENS_MAIN_LENS_LIGHT_MODEL",
        **common,
    )

    # Validate the serialized files, not just the in-memory mapping.  Float64
    # FITS storage makes this a strict pixel-for-pixel equality test.
    serialized_checks = {}
    for name, path, native in (
        ("lens_plane", plane_path, plane_values),
        ("lens_light_subtracted", subtracted_path, subtracted_values),
        ("main_lens", main_lens_path, _native(main_lens_native)),
    ):
        written = np.asarray(fits.getdata(path), dtype=np.float64)
        recovered = np.flipud(written)[
            mapping["y0_model_order"] : mapping["y0_model_order"] + mapping["shape"][0],
            mapping["x0"] : mapping["x0"] + mapping["shape"][1],
        ]
        equal = bool(np.array_equal(recovered[valid], native[valid]))
        if not equal:
            raise RuntimeError(f"serialized exact-fit validation failed for {name}")
        serialized_checks[name] = {
            "array_equal": equal,
            "max_abs_difference": 0.0,
            "validated_pixels": int(valid.sum()),
        }

    qa = {
        "system": config.system,
        "stage": int(stage),
        "mapping_schema": "fit-native-integer-detector-crop-v1",
        "detector_shape": list(map(int, bundle.data_native.shape)),
        "fit_native_shape": list(mapping["shape"]),
        "fit_crop_origin_model_order_yx": [
            int(mapping["y0_model_order"]), int(mapping["x0"])
        ],
        "fit_crop_origin_fits_yx": [
            int(bundle.data_native.shape[0] - mapping["y0_model_order"] - mapping["shape"][0]),
            int(mapping["x0"]),
        ],
        "science_crop_array_equal": bool(mapping["science_array_equal"]),
        "plane_subtraction_identity_max_abs": float(
            np.max(np.abs(relation_error)) if relation_error.size else 0.0
        ),
        "serialized_checks": serialized_checks,
        "convolution": "none; arrays copied directly from FitImaging",
    }
    qa_path = output / f"{config.system}_exact_fit_plane_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n")
    return {
        "autolens_lens_plane_model": plane_path,
        "autolens_lens_light_subtracted": subtracted_path,
        "exact_main_lens_model": main_lens_path,
        "exact_fit_qa": qa_path,
    }


def _full_detector_grid(bundle, config):
    grid = al.Grid2D.uniform(
        shape_native=bundle.data_native.shape,
        pixel_scales=config.pixel_scale,
        over_sample_size=1,
    )
    if config.over_sample:
        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=grid,
            sub_size_list=[4, 2, 1],
            radial_list=[0.8, 1.5],
            centre_list=[bundle.light_centre_yx],
        )
        grid = grid.apply_over_sampling(over_sample_size=over_sample_size)
    return grid


def _psf_centroid_offset(psf: np.ndarray) -> tuple[float, float]:
    values = np.asarray(psf, dtype=float)
    yy, xx = np.indices(values.shape, dtype=float)
    total = float(values.sum())
    return (
        float((values * yy).sum() / total - 0.5 * (values.shape[0] - 1)),
        float((values * xx).sum() / total - 0.5 * (values.shape[1] - 1)),
    )


def _centroid_yx(image: np.ndarray) -> tuple[float, float]:
    values = np.asarray(image, dtype=float)
    yy, xx = np.indices(values.shape, dtype=float)
    total = float(values.sum())
    return float((values * yy).sum() / total), float((values * xx).sum() / total)


def render_full_detector_components(profiles, bundle, config):
    """Render intrinsic and PSF-convolved MGE components on the detector.

    Rendering occurs on the full, unmasked AutoLens grid.  Only after the
    convolution is complete are rows flipped once into FITS-array order.  No
    bounding-box inference or shape-centering operation is involved.
    """
    grid = _full_detector_grid(bundle, config)
    psf = _native(bundle.dataset.psf)
    intrinsic_model_order = []
    convolved_model_order = []
    for profile in profiles:
        intrinsic = _native(profile.image_2d_from(grid=grid))
        blurred = convolve(intrinsic, psf, mode="same", method="fft")
        intrinsic_model_order.append(intrinsic)
        convolved_model_order.append(blurred)
    intrinsic_fits = np.flip(np.asarray(intrinsic_model_order), axis=1)
    convolved_fits = np.flip(np.asarray(convolved_model_order), axis=1)
    return intrinsic_fits, convolved_fits, psf


def _profile_table(profiles) -> fits.BinTableHDU:
    columns = [
        fits.Column(name="COMPONENT", format="J", array=np.arange(1, len(profiles) + 1)),
        fits.Column(name="SIGMA_ARCSEC", format="D", array=[float(p.sigma) for p in profiles]),
        fits.Column(name="INTENSITY", format="D", array=[float(p.intensity) for p in profiles]),
        fits.Column(name="CENTRE_Y", format="D", array=[float(p.centre[0]) for p in profiles]),
        fits.Column(name="CENTRE_X", format="D", array=[float(p.centre[1]) for p in profiles]),
        fits.Column(name="ELL_COMP_0", format="D", array=[float(p.ell_comps[0]) for p in profiles]),
        fits.Column(name="ELL_COMP_1", format="D", array=[float(p.ell_comps[1]) for p in profiles]),
    ]
    return fits.BinTableHDU.from_columns(columns, name="COMPONENTS")


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_wcs_products_from_profiles(
    *,
    profiles,
    einstein_radius: float,
    bundle,
    config,
    stage: int,
    output_name: str,
    tracer_path: Path | None = None,
) -> dict[str, Path]:
    expected = config.lens_gaussians_per_basis * config.lens_bases
    if len(profiles) != expected:
        raise ValueError(f"expected {expected} main-lens Gaussians, found {len(profiles)}")
    centres = np.asarray([p.centre for p in profiles], dtype=float)
    if not np.allclose(centres, centres[0], atol=1.0e-10, rtol=0.0):
        raise ValueError("main-lens MGE profiles do not share one physical centre")

    output = config.output_dir / output_name
    output.mkdir(parents=True, exist_ok=True)
    centre_yx = tuple(map(float, centres[0]))
    x_pixel, y_pixel = autolens_to_fits_pixel(
        centre_yx, bundle.data_native.shape, config.pixel_scale
    )
    roundtrip = fits_pixel_to_autolens(
        (x_pixel, y_pixel), bundle.data_native.shape, config.pixel_scale
    )
    if not np.allclose(roundtrip, centre_yx, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("AutoLens/FITS detector-coordinate round trip failed")
    ra_deg, dec_deg = WCS(bundle.wcs_header).pixel_to_world_values(x_pixel, y_pixel)

    intrinsic_cube, component_cube, psf = render_full_detector_components(
        profiles, bundle, config
    )
    intrinsic_model = intrinsic_cube.sum(axis=0)
    lens_light_model = component_cube.sum(axis=0)
    reconstruction = component_cube.sum(axis=0)
    relative_l2 = float(
        np.linalg.norm(reconstruction - lens_light_model)
        / max(np.linalg.norm(lens_light_model), 1.0e-30)
    )
    if relative_l2 > 1.0e-12:
        raise RuntimeError(f"MGE component reconstruction failed: relative L2={relative_l2}")

    outside_mask = np.asarray(bundle.detector_mask_native, dtype=bool)
    if outside_mask.shape != bundle.data_native.shape:
        raise RuntimeError("full-detector mask shape does not match science data")
    model_product = lens_light_model.copy()
    reduced_product = bundle.data_native - lens_light_model
    intrinsic_product = intrinsic_model.copy()
    component_product = component_cube.copy()
    model_product[outside_mask] = np.nan
    reduced_product[outside_mask] = np.nan
    intrinsic_product[outside_mask] = np.nan
    component_product[:, outside_mask] = np.nan

    psf_model_dy, psf_model_dx = _psf_centroid_offset(psf)
    # A y offset in AutoLens array order changes sign after the final flip.
    psf_fits_dy, psf_fits_dx = -psf_model_dy, psf_model_dx
    intrinsic_centroid = _centroid_yx(intrinsic_model)
    convolved_centroid = _centroid_yx(lens_light_model)

    model_path = output / f"{config.system}_lens_light_model_wcs.fits"
    reduced_path = output / f"{config.system}_lens_light_reduced_wcs.fits"
    intrinsic_path = output / f"{config.system}_lens_light_intrinsic_wcs.fits"
    common_metadata = {
        "STAGE": stage,
        "MAPSCHEM": "AL2FITS2",
        "PSFDY": psf_fits_dy,
        "PSFDX": psf_fits_dx,
    }
    _write_image(
        model_path,
        model_product,
        bundle.wcs_header,
        PRODUCT="LENS_LIGHT_MODEL",
        **common_metadata,
    )
    _write_image(
        intrinsic_path,
        intrinsic_product,
        bundle.wcs_header,
        PRODUCT="LENS_LIGHT_INTRINSIC",
        **common_metadata,
    )
    _write_image(
        reduced_path,
        reduced_product,
        bundle.wcs_header,
        PRODUCT="DATA_MINUS_LENS_LIGHT",
        **common_metadata,
    )

    component_path = output / f"{config.system}_lens_mge_components_wcs.fits"
    component_header = _primary_header(bundle.wcs_header)
    component_header["PRODUCT"] = "LENS_MGE_COMPONENTS"
    component_header["STAGE"] = stage
    component_header["NGAUSS"] = len(profiles)
    component_header["RELL2"] = relative_l2
    component_header["EINRAD"] = float(einstein_radius)
    component_header["MGESCHEM"] = "MGE7FULL"
    component_header["MAPSCHEM"] = "AL2FITS2"
    _atomic_fits_write(
        fits.HDUList(
            [
                fits.PrimaryHDU(data=component_product.astype(np.float32), header=component_header),
                _profile_table(profiles),
                fits.ImageHDU(data=intrinsic_cube.astype(np.float32), name="INTRINSIC"),
            ]
        ),
        component_path,
    )

    centre_path = output / f"{config.system}_lens_light_centre_wcs.json"
    centre_payload = {
        "system": config.system,
        "stage": stage,
        "mapping_schema": DETECTOR_MAPPING_SCHEMA,
        "autolens_y_arcsec": centre_yx[0],
        "autolens_x_arcsec": centre_yx[1],
        "fits_x_pixel_zero_indexed": float(x_pixel),
        "fits_y_pixel_zero_indexed": float(y_pixel),
        "ra_deg": float(ra_deg),
        "dec_deg": float(dec_deg),
        "intrinsic_model_centroid_fits_yx": list(map(float, intrinsic_centroid)),
        "convolved_model_centroid_fits_yx": list(map(float, convolved_centroid)),
        "psf_centroid_offset_model_yx_pix": [psf_model_dy, psf_model_dx],
        "psf_centroid_offset_fits_yx_pix": [psf_fits_dy, psf_fits_dx],
        "reduced_image_definition": (
            f"science data minus full-detector PSF-convolved stage-{stage} lens-plane light model"
        ),
    }
    centre_path.write_text(
        json.dumps(centre_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    qa_path = output / f"{config.system}_wcs_product_qa.json"
    qa = {
        "component_sum_relative_l2": relative_l2,
        "detector_shape": list(bundle.data_native.shape),
        "finite_model_pixels": int(np.count_nonzero(~outside_mask)),
        "mapping_roundtrip_max_abs_arcsec": float(
            np.max(np.abs(np.asarray(roundtrip) - np.asarray(centre_yx)))
        ),
        "profile_centre_fits_xy": [float(x_pixel), float(y_pixel)],
        "intrinsic_centroid_fits_yx": list(map(float, intrinsic_centroid)),
        "convolved_centroid_fits_yx": list(map(float, convolved_centroid)),
        "psf_centroid_offset_fits_yx_pix": [psf_fits_dy, psf_fits_dx],
        "mge_schema": MGE_SCHEMA,
        "mapping_schema": DETECTOR_MAPPING_SCHEMA,
    }
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = output / f"{config.system}_wcs_product_manifest.json"
    manifest = {
        "system": config.system,
        "stage": stage,
        "tracer_path": str(tracer_path.resolve()) if tracer_path else None,
        "tracer_sha256": _sha256(tracer_path),
        "science_path": str(config.science_cutout),
        "science_sha256": _sha256(config.science_cutout),
        "psf_path": str(config.psf),
        "psf_sha256": _sha256(config.psf),
        "mge_schema": MGE_SCHEMA,
        "mapping_schema": DETECTOR_MAPPING_SCHEMA,
        "products": {
            "centre": str(centre_path),
            "model": str(model_path),
            "intrinsic": str(intrinsic_path),
            "reduced": str(reduced_path),
            "mge_components": str(component_path),
            "qa": str(qa_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "centre": centre_path,
        "model": model_path,
        "intrinsic": intrinsic_path,
        "reduced": reduced_path,
        "mge_components": component_path,
        "qa": qa_path,
        "manifest": manifest_path,
    }


def write_wcs_products(
    result, bundle, config, *, stage: int = 3, output_name: str = "wcs_products"
) -> dict[str, Path]:
    fit = result.max_log_likelihood_fit
    instance = result.max_log_likelihood_instance
    fitted_light_model = fit.model_obj_linear_light_profiles_to_light_profiles
    profiles = list(fitted_light_model.galaxies.lens.bulge.profile_list)
    tracer_path = None
    try:
        tracer_path = Path(result.search.paths.files_path) / "tracer.json"
    except Exception:
        pass
    products = write_wcs_products_from_profiles(
        profiles=profiles,
        einstein_radius=float(instance.galaxies.lens.mass.einstein_radius),
        bundle=bundle,
        config=config,
        stage=stage,
        output_name=output_name,
        tracer_path=tracer_path,
    )
    # Overwrite the re-rendered main-lens product and add full-plane products
    # using the literal maximum-likelihood FitImaging arrays.  This eliminates
    # the half-pixel parity ambiguity of independent full-detector rendering.
    products.update(
        write_exact_fit_plane_products(
            fit=fit,
            bundle=bundle,
            config=config,
            stage=stage,
            output_name=output_name,
        )
    )
    return products
