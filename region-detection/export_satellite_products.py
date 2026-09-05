#!/usr/bin/env python3
"""Export exact FitImaging plane products and independent satellite models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import autolens as al
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_ROOT = SCRIPT_DIR.parent / "lens-modeling"
sys.path.insert(0, str(MODELING_ROOT))

from lens_modeling.config import load_config  # noqa: E402
from lens_modeling.data import load_dataset  # noqa: E402
from lens_modeling.products import (  # noqa: E402
    autolens_to_fits_pixel,
    fit_array_to_detector_fits,
    fit_native_detector_mapping,
    render_full_detector_components,
    write_exact_fit_plane_products,
)


def stage_tracer(output_dir: Path, stage: int) -> Path:
    matches = list(output_dir.glob(f"stage{stage}_*/**/files/tracer.json"))
    if stage == 4:
        current = [path for path in matches if "_clipsig1" in str(path)]
        if len(current) == 1:
            return current[0]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one stage-{stage} tracer under {output_dir}, found {len(matches)}"
        )
    return matches[0]


def maximum_likelihood_fit(config, bundle, tracer, stage: int):
    """Rebuild the deterministic fit represented by an on-disk tracer.

    Stage 4 requires the same stage-3 source adapt image and inversion settings
    used by the production likelihood.  No non-linear search is rerun.
    """
    if stage == 3:
        return al.FitImaging(dataset=bundle.dataset_masked, tracer=tracer)
    if stage != 4:
        raise ValueError(f"Exact plane export supports stages 3 and 4, not {stage}")

    stage3_path = stage_tracer(config.output_dir, 3)
    stage3_tracer = al.from_json(file_path=stage3_path)
    stage3_fit = al.FitImaging(
        dataset=bundle.dataset_masked, tracer=stage3_tracer
    )
    source_adapt_image = stage3_fit.model_images_of_planes_list[-1]
    pixel_sources = [
        galaxy for galaxy in tracer.galaxies if hasattr(galaxy, "pixelization")
    ]
    if len(pixel_sources) != 1:
        raise RuntimeError(
            f"Expected one stage-4 pixelized source, found {len(pixel_sources)}"
        )
    adapt_images = al.AdaptImages(
        galaxy_image_dict={pixel_sources[0]: source_adapt_image}
    )
    return al.FitImaging(
        dataset=bundle.dataset_masked,
        tracer=tracer,
        adapt_images=adapt_images,
        settings_inversion=al.SettingsInversion(
            use_positive_only_solver=True,
            positive_only_uses_p_initial=True,
            use_border_relocator=True,
            no_regularization_add_to_curvature_diag_value=1.0e-3,
            use_w_tilde_numpy=False,
            use_source_loop=False,
            tolerance=1.0e-8,
            maxiter=250,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--products-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    bundle = load_dataset(config)
    tracer_path = stage_tracer(config.output_dir, args.stage)
    tracer = al.from_json(file_path=tracer_path)
    fit = maximum_likelihood_fit(config, bundle, tracer, args.stage)
    mapping = fit_native_detector_mapping(fit=fit, bundle=bundle)
    galaxy_images = fit.galaxy_model_image_dict
    satellites = [
        galaxy
        for galaxy in tracer.galaxies
        if not hasattr(galaxy, "mass")
        and hasattr(galaxy, "bulge")
        and np.isclose(float(galaxy.redshift), float(config.z_lens), atol=1.0e-8)
    ]

    output = args.products_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    outside_mask = np.asarray(bundle.detector_mask_native, dtype=bool)
    records = []
    for index, satellite in enumerate(satellites):
        profiles = list(satellite.bulge.profile_list)
        intrinsic_cube, _, _ = render_full_detector_components(
            profiles, bundle, config
        )
        intrinsic = intrinsic_cube.sum(axis=0)
        model = fit_array_to_detector_fits(
            galaxy_images[satellite], fit=fit, bundle=bundle, mapping=mapping
        )
        intrinsic[outside_mask] = np.nan
        centre_yx = tuple(map(float, profiles[0].centre))
        centre_x, centre_y = autolens_to_fits_pixel(
            centre_yx, bundle.data_native.shape, config.pixel_scale
        )
        ra_deg, dec_deg = WCS(bundle.wcs_header).pixel_to_world_values(
            centre_x, centre_y
        )
        model_path = output / f"{config.system}_satellite_{index}_light_model_wcs.fits"
        intrinsic_path = output / f"{config.system}_satellite_{index}_light_intrinsic_wcs.fits"
        for path, data, product in (
            (model_path, model, "SATELLITE_LIGHT_MODEL"),
            (intrinsic_path, intrinsic, "SATELLITE_LIGHT_INTRINSIC"),
        ):
            header = bundle.wcs_header.copy()
            for key in ("XTENSION", "PCOUNT", "GCOUNT", "EXTNAME", "EXTVER"):
                header.pop(key, None)
            header["PRODUCT"] = product
            header["STAGE"] = int(args.stage)
            header["SATINDEX"] = int(index)
            fits.writeto(path, data.astype(np.float32), header, overwrite=True)
        records.append({
            "index": int(index),
            "profile_count": int(len(profiles)),
            "autolens_y_arcsec": centre_yx[0],
            "autolens_x_arcsec": centre_yx[1],
            "fits_x_pixel_zero_indexed": float(centre_x),
            "fits_y_pixel_zero_indexed": float(centre_y),
            "ra_deg": float(ra_deg),
            "dec_deg": float(dec_deg),
            "model": str(model_path),
            "intrinsic": str(intrinsic_path),
            "positive_model_flux": float(np.nansum(np.clip(model, 0.0, None))),
        })

    exact_products = write_exact_fit_plane_products(
        fit=fit,
        bundle=bundle,
        config=config,
        stage=args.stage,
        output_name=output.name,
        output_dir=output,
    )
    plane_model_path = exact_products["autolens_lens_plane_model"]
    subtracted_path = exact_products["autolens_lens_light_subtracted"]

    manifest = {
        "system": config.system,
        "stage": int(args.stage),
        "tracer": str(tracer_path.resolve()),
        "selection": (
            "lens-redshift light-bearing galaxies without a mass profile; "
            "the higher-redshift source is excluded"
        ),
        "satellite_count": len(records),
        "satellites": records,
        "autolens_lens_plane_model": str(plane_model_path),
        "autolens_lens_light_subtracted": str(subtracted_path),
        "autolens_subtracted_definition": (
            "literal FitImaging subtracted_images_of_planes_list[-1], mapped "
            "from the trimmed/FFT-padded native array to detector pixels"
        ),
        "exact_fit_qa": str(exact_products["exact_fit_qa"]),
        "fit_array_validation": "strict serialized array_equal on every unmasked pixel",
    }
    manifest_path = output / f"{config.system}_satellite_light_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
