from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .diagnostics import validate_empirical_kernel_math
from .io_utils import write_input_manifest
from .photometry import PhotometryProducts, run_photometry
from .photoz import plot_final_photoz_sed, run_photoz_variants


@dataclass
class PipelineResult:
    lens_id: str
    result_dir: Path
    summary: pd.DataFrame
    common_bands: list[str]
    band_status: pd.DataFrame
    elapsed_seconds: float


def _remove_obsolete_reduced_source_products(config: PipelineConfig) -> None:
    """Remove Photo-lens products from the retired second lens subtraction."""
    obsolete = [
        config.result_dir / f"photometry_bcnz_reduced_source_{config.lens_id}.csv",
        config.result_dir / "lens_subtraction_summary.csv",
        config.result_dir / f"photoz_variants_{config.lens_id}.png",
        config.result_dir / f"final_photoz_sed_{config.lens_id}_wNB1.png",
    ]
    for weight in config.nb_weights:
        tag = f"wNB{weight:g}".replace(".", "p").replace("-", "m")
        obsolete.extend([
            config.result_dir / tag / "galcat_in_reduced_source.pq",
            config.result_dir / tag / "pzcat_reduced_source.pq",
            config.result_dir / tag / "signed_pulls_reduced_source.csv",
        ])
    for path in obsolete:
        path.unlink(missing_ok=True)
    # Photo-lens is now a w_NB=1 production run; remove the retired A/B cache so
    # stale w_NB=2 catalogues cannot be mistaken for final products.
    retired_weight_dir = config.result_dir / "wNB2"
    if retired_weight_dir.is_dir():
        shutil.rmtree(retired_weight_dir)


def _relative_outputs(config: PipelineConfig) -> list[str]:
    return sorted(
        path.relative_to(config.project_root).as_posix()
        for path in config.result_dir.rglob("*")
        if path.is_file()
    )


def _validate_mask_products(
    config: PipelineConfig,
    products: PhotometryProducts,
) -> dict:
    convergence = products.mask_convergence
    if convergence.empty:
        raise AssertionError("Mask-convergence table is empty")
    maximum = float(convergence.delta_11_41_over_error.max())
    if maximum >= 0.2:
        worst = convergence.loc[
            convergence.delta_11_41_over_error.idxmax(),
            ["band", "image_file", "delta_11_41_over_error"],
        ].to_dict()
        raise AssertionError(
            f"Subpixel convergence failed (max={maximum:.4f}): {worst}"
        )
    if not np.all(convergence.aperture_subpixels == config.aperture_subpixels):
        raise AssertionError("APERTURE_SUBPIXELS was not propagated consistently")
    fractional = (
        ~np.isclose(convergence.area_source_sub11 % 1.0, 0.0)
        | ~np.isclose(convergence.area_background_sub11 % 1.0, 0.0)
    )
    if not fractional.all():
        bad = convergence.loc[~fractional, ["band", "image_file"]]
        raise AssertionError(
            "Some polygon areas remain integral after subpixel masking: "
            + bad.head().to_json(orient="records")
        )
    return {
        "max_delta_11_41_over_error": maximum,
        "n_mask_rows": int(len(convergence)),
        "n_nb_exposures": int((convergence.band_type == "nb").sum()),
        "n_cfht_exposures": int((convergence.band_type == "bb").sum()),
    }


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run Photo-lens photometry, diagnostics, and both NB-weight photo-z fits."""
    started = time.time()
    config.validate()
    config.result_dir.mkdir(parents=True, exist_ok=True)
    _remove_obsolete_reduced_source_products(config)
    print(f"Project root: {config.project_root}")
    print(f"Result directory: {config.result_dir}")
    print(f"Lens ID: {config.lens_id}")
    print(f"APERTURE_SUBPIXELS: {config.aperture_subpixels}")
    print(f"Convergence subpixels: {config.convergence_subpixels}")
    print(f"NB weights: {config.nb_weights}; official candidate={config.official_nb_weight}")

    manifest_path = write_input_manifest(config)
    kernel_validation = validate_empirical_kernel_math()
    pd.DataFrame([kernel_validation]).to_csv(
        config.result_dir / "kernel_math_validation.csv", index=False
    )

    photometry = run_photometry(config)
    mask_validation = _validate_mask_products(config, photometry)
    summary, products, common, band_status = run_photoz_variants(
        config, photometry.source, photometry.lens
    )
    redshift_grid = pd.read_csv(
        config.result_dir / "photoz_redshift_grid.csv"
    ).z.to_numpy(dtype=float)
    final_figure_paths = plot_final_photoz_sed(
        config, products, redshift_grid, common
    )
    official_order_rows = summary.loc[
        np.isclose(summary.nb_weight, config.official_nb_weight),
        "p_source_gt_lens",
    ].drop_duplicates()
    if len(official_order_rows) != 1:
        raise AssertionError(
            "Expected one P(z_source > z_lens) value for the official NB weight"
        )
    p_source_gt_lens = float(official_order_rows.iloc[0])

    elapsed = time.time() - started
    record = {
        "status": "complete",
        "lens_id": config.lens_id,
        "aperture_subpixels": config.aperture_subpixels,
        "convergence_subpixels": config.convergence_subpixels,
        "coadd_error_factor": config.coadd_error_factor,
        "nb_weights": list(config.nb_weights),
        "official_nb_weight": config.official_nb_weight,
        "p_source_gt_lens": p_source_gt_lens,
        "p_source_gt_lens_definition": (
            "integral dz_l p_l(z_l) integral_{z_l}^infinity dz_s p_s(z_s), "
            "using trapezoidal integration of normalized BCNz p(z)"
        ),
        "nb_aperture_centre_table": (
            config.relative_to_project(config.nb_aperture_centre_path)
            if config.nb_aperture_centre_path is not None
            else None
        ),
        "nb_aperture_shift_mode": (
            "per_band_antisymmetry_centre_relative_to_positions_gui_bb"
            if config.nb_aperture_centre_path is not None
            else "disabled"
        ),
        "common_bands": common,
        "excluded_bands": band_status.loc[
            ~band_status.effective_common, "band"
        ].tolist(),
        "elapsed_seconds": elapsed,
        "manifest": config.relative_to_project(manifest_path),
        "figures": {
            "final_lens_photoz_sed": config.relative_to_project(
                final_figure_paths["lens"]
            ),
            "final_source_photoz_sed": config.relative_to_project(
                final_figure_paths["source"]
            ),
            "nb_aperture_overview": config.relative_to_project(
                config.result_dir / f"aperture_selection_NB_{config.lens_id}.png"
            ),
            "bb_aperture_overview": config.relative_to_project(
                config.result_dir / f"aperture_selection_BB_{config.lens_id}.png"
            ),
        },
        "kernel_validation": kernel_validation,
        "mask_validation": mask_validation,
        "outputs": _relative_outputs(config),
    }
    marker = config.result_dir / "run_complete.json"
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"Completed lens {config.lens_id} in {elapsed:.1f} s")
    print(summary[[
        "target", "nb_weight", "zb", "odds", "pz_width",
        "chi2", "n_band", "pull_std_nb", "pull_std_bb",
        "p_source_gt_lens",
    ]].to_string(index=False))
    return PipelineResult(
        lens_id=config.lens_id,
        result_dir=config.result_dir,
        summary=summary,
        common_bands=common,
        band_status=band_status,
        elapsed_seconds=elapsed,
    )
