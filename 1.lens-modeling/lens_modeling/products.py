from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import autolens as al
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


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


def _psf_centroid_offset(psf: np.ndarray) -> tuple[float, float]:
    values = np.asarray(psf, dtype=float)
    yy, xx = np.indices(values.shape, dtype=float)
    total = float(values.sum())
    return (
        float((values * yy).sum() / total - 0.5 * (values.shape[0] - 1)),
        float((values * xx).sum() / total - 0.5 * (values.shape[1] - 1)),
    )


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


def write_wcs_products(
    result, bundle, config, *, stage: int = 3, output_name: str = "wcs_products"
) -> dict[str, Path]:
    fit = result.max_log_likelihood_fit
    tracer = fit.model_obj_linear_light_profiles_to_light_profiles
    tracer_path = None
    try:
        tracer_path = Path(result.search.paths.files_path) / "tracer.json"
    except Exception:
        pass
    return write_consistent_fit_products(
        fit, tracer, bundle, config, stage=stage,
        output_dir=config.output_dir / output_name, tracer_path=tracer_path)


def write_consistent_fit_products(fit, tracer, bundle, config, *, stage,
                                  output_dir, tracer_path=None):
    """Export all light components and centres on the actual likelihood grid.

    A separate full-detector grid can have a different subpixel origin after
    masking or FFT padding. Profiles, convolution and coordinate conversion
    must therefore share the grid that was used by this fit.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mapping = fit_native_detector_mapping(fit, bundle)
    exact = write_exact_fit_plane_products(
        fit, bundle, config, stage=stage, output_name=output.name, output_dir=output)
    mapped = lambda image: fit_array_to_detector_fits(image, fit, bundle, mapping)
    ny, nx = mapping['shape']
    x0 = mapping['x0']
    y0 = bundle.data_native.shape[0] - mapping['y0_model_order'] - ny
    origin = np.asarray(fit.grids.lp.origin, dtype=float)

    def centre(profile):
        y, x = map(float, profile.centre)
        return (x0 + (nx-1)/2 + (x-origin[1])/config.pixel_scale,
                y0 + (ny-1)/2 + (y-origin[0])/config.pixel_scale)

    def write(path, data, product):
        header = _primary_header(bundle.wcs_header)
        header['PRODUCT'] = product
        header['MAPSCHEM'] = 'FITNATIVE1'
        header['STAGE'] = stage
        _atomic_fits_write(fits.HDUList([fits.PrimaryHDU(np.asarray(data, dtype=np.float64), header)]), path)

    lenses = [g for g in tracer.galaxies if hasattr(g, 'mass') and hasattr(g, 'bulge')]
    if len(lenses) != 1:
        raise ValueError('Expected one light-bearing main lens')
    lens = lenses[0]
    profiles = list(lens.bulge.profile_list)
    if not np.allclose([p.centre for p in profiles], profiles[0].centre, atol=1e-10, rtol=0):
        raise ValueError('Main-lens MGE profiles must share one centre')
    galaxies = [al.Galaxy(redshift=config.z_lens, bulge=p) for p in profiles]
    light_tracer = al.Tracer(galaxies=galaxies)
    blurred = light_tracer.galaxy_blurred_image_2d_dict_from(
        grid=fit.grids.lp, psf=fit.dataset.psf, blurring_grid=fit.grids.blurring)
    cube = np.array([mapped(blurred[g]) for g in galaxies])
    intrinsic = np.array([mapped(p.image_2d_from(grid=fit.grids.lp)) for p in profiles])
    main_image = fits.getdata(exact['exact_main_lens_model'])
    valid = np.isfinite(main_image)
    error = float(np.max(np.abs(cube.sum(0)[valid] - main_image[valid])))
    if not np.allclose(cube.sum(0)[valid], main_image[valid], atol=1e-7, rtol=1e-7):
        raise RuntimeError(f'MGE components do not reproduce the fit: {error}')
    header = _primary_header(bundle.wcs_header)
    header['MAPSCHEM'] = 'FITNATIVE1'
    header['MGESCHEM'] = 'MGE7FULL'
    header['NGAUSS'] = len(profiles)
    header['RELL2'] = float(np.linalg.norm((cube.sum(0)-main_image)[valid]) / np.linalg.norm(main_image[valid]))
    header['EINRAD'] = float(lens.mass.einstein_radius)
    cube_path = output / f'{config.system}_lens_mge_components_wcs.fits'
    _atomic_fits_write(fits.HDUList([fits.PrimaryHDU(cube, header), _profile_table(profiles),
                                   fits.ImageHDU(intrinsic, name='INTRINSIC')]), cube_path)
    intrinsic_path = output / f'{config.system}_lens_light_intrinsic_wcs.fits'
    reduced_path = output / f'{config.system}_lens_light_reduced_wcs.fits'
    write(intrinsic_path, intrinsic.sum(0), 'MAIN_LENS_INTRINSIC')
    write(reduced_path, fits.getdata(exact['autolens_lens_light_subtracted']), 'FULL_FOREGROUND_SUBTRACTED')
    x, y = centre(profiles[0])
    sky = WCS(bundle.wcs_header).celestial.pixel_to_world(x, y).icrs
    oldx, oldy = autolens_to_fits_pixel(profiles[0].centre, bundle.data_native.shape, config.pixel_scale)
    psfy, psfx = _psf_centroid_offset(np.asarray(fit.dataset.psf.native))
    centre_path = output / f'{config.system}_lens_light_centre_wcs.json'
    centre_record = dict(system=config.system, stage=stage, mapping_schema='fit-native-grid',
        autolens_y_arcsec=float(profiles[0].centre[0]), autolens_x_arcsec=float(profiles[0].centre[1]),
        fits_x_pixel_zero_indexed=x, fits_y_pixel_zero_indexed=y, ra_deg=float(sky.ra.deg), dec_deg=float(sky.dec.deg),
        grid_correction_xy_pixels=[x-oldx, y-oldy], psf_centroid_offset_fits_yx_pix=[-psfy, psfx],
        centre_definition='intrinsic shared MGE centre mapped through actual fit grid; not a convolved-light centre')
    centre_path.write_text(json.dumps(centre_record, indent=2)+'\n')
    records = []
    for galaxy_index, galaxy in enumerate(tracer.galaxies):
        if hasattr(galaxy, 'mass') or not hasattr(galaxy, 'bulge') or not np.isclose(galaxy.redshift, config.z_lens):
            continue
        index = len(records)
        satellite_profiles = list(galaxy.bulge.profile_list)
        model = mapped(fit.galaxy_model_image_dict[fit.tracer.galaxies[galaxy_index]])
        raw = sum(p.image_2d_from(grid=fit.grids.lp) for p in satellite_profiles)
        mp = output / f'{config.system}_satellite_{index}_light_model_wcs.fits'
        ip = output / f'{config.system}_satellite_{index}_light_intrinsic_wcs.fits'
        write(mp, model, 'SATELLITE_LIGHT_MODEL')
        write(ip, mapped(raw), 'SATELLITE_LIGHT_INTRINSIC')
        sx, sy = centre(satellite_profiles[0])
        sk = WCS(bundle.wcs_header).celestial.pixel_to_world(sx, sy).icrs
        records.append(dict(index=index, profile_count=len(satellite_profiles),
            fits_x_pixel_zero_indexed=sx, fits_y_pixel_zero_indexed=sy,
            autolens_y_arcsec=float(satellite_profiles[0].centre[0]), autolens_x_arcsec=float(satellite_profiles[0].centre[1]),
            ra_deg=float(sk.ra.deg), dec_deg=float(sk.dec.deg), model=mp.name, intrinsic=ip.name,
            positive_model_flux=float(np.nansum(np.clip(model, 0, None)))))
    manifest = dict(system=config.system, stage=stage,
        tracer=os.path.relpath(tracer_path, output) if tracer_path else None,
        satellite_count=len(records), satellites=records,
        autolens_lens_plane_model=Path(exact['autolens_lens_plane_model']).name,
        autolens_lens_light_subtracted=Path(exact['autolens_lens_light_subtracted']).name,
        exact_fit_qa=Path(exact['exact_fit_qa']).name)
    (output / f'{config.system}_satellite_light_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    qa = dict(system=config.system, stage=stage, component_sum_max_abs_error=error,
        grid_correction_xy_pixels=[x-oldx, y-oldy], fit_shape=list(mapping['shape']),
        detector_shape=list(bundle.data_native.shape), fit_grid_origin_yx=origin.tolist())
    (output / 'consistent_export_complete.json').write_text(json.dumps(qa, indent=2)+'\n')
    (output / f'{config.system}_wcs_product_qa.json').write_text(json.dumps(qa, indent=2)+'\n')
    product_manifest = dict(system=config.system, stage=stage, mapping='actual fit-native detector grid',
        files={key:Path(value).name for key,value in exact.items()},
        mge_components=cube_path.name, lens_intrinsic=intrinsic_path.name,
        lens_light_reduced=reduced_path.name, lens_centre=centre_path.name)
    (output / f'{config.system}_wcs_product_manifest.json').write_text(json.dumps(product_manifest, indent=2)+'\n')
    return {**exact, 'mge_components':cube_path, 'lens_intrinsic':intrinsic_path,
            'lens_light_reduced':reduced_path, 'lens_centre':centre_path}
