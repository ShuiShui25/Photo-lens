#!/usr/bin/env python3
"""Finalize PHOTO-LENS032 from PHOTO-LENS031 with NB535 excluded for 3110.

All measured photometry and all non-3110 system products remain byte-for-byte
copies of PHOTO-LENS031.  For 3110 this script installs the already-computed
BCNz refit that omits NB535, then synchronizes every fit-dependent table.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import bcnz
import xarray as xr
from bcnz.fit.photoz import minimize_all_z


HERE = Path(__file__).resolve().parent
RESULT = HERE / "results" / "3110"
UPSTREAM = (
    HERE.parent.parent / "PHOTO-LENS03" / "photometric-estimation"
    / "hsc-g-response" / "source-parameter-transfer-3441"
    / "targeted-nb-removals"
)
PHOTO_LENS03_PHOTOZ = HERE.parent.parent / "PHOTO-LENS03" / "photometric-estimation"
REMOVED = "NB535"


def probability_source_behind_lens(
    lens_probability: np.ndarray,
    source_probability: np.ndarray,
    redshift: np.ndarray,
) -> float:
    lens = np.asarray(lens_probability, dtype=float)
    source = np.asarray(source_probability, dtype=float)
    lens /= np.trapz(lens, redshift)
    source /= np.trapz(source, redshift)
    dz = np.diff(redshift)
    upper_tail = np.r_[np.cumsum((0.5 * (source[:-1] + source[1:]) * dz)[::-1])[::-1], 0.0]
    return float(np.clip(np.trapz(lens * upper_tail, redshift), 0.0, 1.0))


def signed_pulls(pz_row: pd.Series, flat_row: pd.Series, fit_bands: list[str]) -> pd.DataFrame:
    rows = []
    for band in fit_bands:
        flux = float(flat_row[f"flux_{band}"])
        error = float(flat_row[f"flux_error_{band}"])
        model = float(pz_row[f"model_{band}"])
        rows.append({
            "band": band,
            "kind": "NB" if band.startswith("pau_nb") else "BB",
            "flux": flux,
            "flux_error": error,
            "model_flux": model,
            "pull": (flux - model) / error,
        })
    table = pd.DataFrame(rows)
    table.insert(0, "target", "")
    table.insert(0, "nb_weight", 1.0)
    return table


def update_fixed_specz_model(fit_bands: list[str]) -> dict:
    """Refit the diagnostic source model at the known z_s without NB535."""
    runs = bcnz.config.eriksen2019()
    cached = bcnz.model.cache_model(PHOTO_LENS03_PHOTOZ / "model", runs)
    g_only = bcnz.model.cache_model(
        PHOTO_LENS03_PHOTOZ / "hsc-g-response" / "model-g", runs
    )
    for run in cached:
        keep = cached[run].band.values != "subaru_g"
        cached[run] = xr.concat([cached[run].isel(band=keep), g_only[run]], dim="band")
    models = {
        run: model.sel(band=fit_bands).rename(sed="model")
        for run, model in cached.items()
    }
    flat = pd.read_parquet(RESULT / "wNB1" / "galcat_in_source.pq")
    chi2_all, _, fitted_flux = minimize_all_z(
        flat, models, fit_bands=fit_bands, Niter=1000, Nskip=10
    )
    previous = json.loads((RESULT / "source_fixed_specz_model.json").read_text())
    spec_z = float(previous["source_spec_z"])
    native_grid = np.asarray(chi2_all.z.values, dtype=float)
    fixed_z = float(native_grid[np.argmin(np.abs(native_grid - spec_z))])
    fixed_runs = chi2_all.sel(ref_id=3110, z=fixed_z, method="nearest")
    fixed_run = int(fixed_runs.run.values[int(np.argmin(fixed_runs.values))])
    fixed_chi2 = float(fixed_runs.sel(run=fixed_run).values)
    fixed_model = fitted_flux.sel(
        ref_id=3110, z=fixed_z, run=fixed_run, method="nearest"
    ).sel(band=fit_bands).values.astype(float)
    pd.DataFrame({
        "band": fit_bands,
        "fixed_specz_model_flux": fixed_model,
    }).to_csv(RESULT / "source_fixed_specz_model.csv", index=False)
    (RESULT / "source_fixed_specz_model.json").write_text(json.dumps({
        "source_spec_z": spec_z,
        "fixed_grid_z": fixed_z,
        "fixed_best_run": fixed_run,
        "fixed_specz_chi2": fixed_chi2,
        "excluded_bands": [REMOVED],
        "release": "PHOTO-LENS032",
    }, indent=2) + "\n")
    return {
        "spec_z": spec_z,
        "grid_z": fixed_z,
        "best_run": fixed_run,
        "chi2": fixed_chi2,
        "model": dict(zip(fit_bands, fixed_model)),
    }


def main() -> None:
    replacements = {
        "lens": UPSTREAM / "system_3110_transfer_remove_NB535_lens.pq",
        "source": UPSTREAM / "system_3110_transfer_remove_NB535_source.pq",
    }
    for target, source in replacements.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, RESULT / "wNB1" / f"pzcat_{target}.pq")

    common = pd.read_csv(RESULT / "common_photoz_bands.csv")
    common = common.loc[common.band.astype(str).ne(REMOVED)].reset_index(drop=True)
    common.to_csv(RESULT / "common_photoz_bands.csv", index=False)
    fit_bands = [
        f"pau_nb{band[2:]}" if band.startswith("NB") else band
        for band in common.band.astype(str)
    ]

    status_path = RESULT / "photoz_band_selection_status.csv"
    status = pd.read_csv(status_path)
    status["exclusion_reason"] = status["exclusion_reason"].fillna("").astype(str)
    selected = status.band.astype(str).eq(REMOVED)
    if selected.sum() != 1:
        raise RuntimeError("Expected exactly one NB535 row in band-status table")
    for column in ("source_usable", "lens_usable", "effective_common"):
        status.loc[selected, column] = False
    status.loc[selected, "exclusion_reason"] = (
        "PHOTO-LENS032 fixed exclusion: high-impact NB535"
    )
    status.to_csv(status_path, index=False)

    z = pd.read_csv(RESULT / "photoz_redshift_grid.csv").z.to_numpy(float)
    pcols = [f"z{i}" for i in range(len(z))]
    pzcats = {
        target: pd.read_parquet(RESULT / "wNB1" / f"pzcat_{target}.pq")
        for target in ("lens", "source")
    }
    porder = probability_source_behind_lens(
        pzcats["lens"].loc[3110, pcols].to_numpy(float),
        pzcats["source"].loc[3110, pcols].to_numpy(float),
        z,
    )

    pull_stats = {}
    for target in ("lens", "source"):
        flat = pd.read_parquet(RESULT / "wNB1" / f"galcat_in_{target}.pq")
        table = signed_pulls(pzcats[target].loc[3110], flat.loc[3110], fit_bands)
        table.loc[:, "target"] = target
        table.to_csv(RESULT / "wNB1" / f"signed_pulls_{target}.csv", index=False)
        pull_stats[target] = {
            kind: float(np.std(table.loc[table.kind.eq(kind), "pull"], ddof=1))
            for kind in ("NB", "BB")
        }

    summary_path = RESULT / "photoz_variant_summary.csv"
    summary = pd.read_csv(summary_path)
    for target in ("lens", "source"):
        row = pzcats[target].loc[3110]
        selected = summary.target.eq(target)
        for column in (
            "zb", "zb_mean", "odds", "pz_width", "chi2", "n_band",
            "n_narrow", "best_run",
        ):
            summary.loc[selected, column] = row[column]
        summary.loc[selected, "chi2_per_band"] = float(row.chi2 / row.n_band)
        summary.loc[selected, "pull_std_nb"] = pull_stats[target]["NB"]
        summary.loc[selected, "pull_std_bb"] = pull_stats[target]["BB"]
        bb = pd.read_csv(RESULT / "wNB1" / f"signed_pulls_{target}.csv")
        bb = bb.loc[bb.kind.eq("BB")]
        summary.loc[selected, "signed_bb_pulls"] = json.dumps(
            dict(zip(bb.band, bb.pull.round(6))), sort_keys=True
        )
        summary.loc[selected, "p_source_gt_lens"] = porder
    summary.to_csv(summary_path, index=False)

    fixed_specz = update_fixed_specz_model(fit_bands)

    probability = pd.read_csv(RESULT / "redshift_order_probability.csv")
    probability.loc[:, "p_source_gt_lens"] = porder
    probability.to_csv(RESULT / "redshift_order_probability.csv", index=False)

    run_path = RESULT / "run_complete.json"
    run = json.loads(run_path.read_text())
    run["release"] = "PHOTO-LENS032"
    run["parent_release"] = "PHOTO-LENS031"
    run["p_source_gt_lens"] = porder
    run["common_bands"] = common.band.astype(str).tolist()
    run["excluded_bands"] = sorted(set(run.get("excluded_bands", [])) | {REMOVED})
    run["release_change"] = "NB535 excluded from both lens and source BCNz fits for system 3110"
    run["manifest"] = "photometric-estimation/results/3110/input_manifest.csv"
    run["figures"] = {
        "final_lens_photoz_sed": "photometric-estimation/results/3110/final_photoz_sed_lens_3110_wNB1.png",
        "final_source_photoz_sed": "photometric-estimation/results/3110/final_photoz_sed_source_3110_wNB1.png",
        "nb_aperture_overview": "photometric-estimation/results/3110/aperture_selection_NB_3110.png",
        "bb_aperture_overview": "photometric-estimation/results/3110/aperture_selection_BB_3110.png",
    }
    run_path.write_text(json.dumps(run, indent=2) + "\n")

    # Regenerate the two final diagnostic figures so they simultaneously show
    # the corrected p(z), omit the NB535 flux point, mark the removed wavelength,
    # and retain the newly refitted source spec-z model curve.
    sys.path.insert(0, str(PHOTO_LENS03_PHOTOZ))
    from paus_lensing034 import PipelineConfig
    from paus_lensing034.photoz import plot_final_photoz_sed
    products = {"wNB1": {}}
    for target in ("lens", "source"):
        products["wNB1"][target] = {
            "pzcat": pzcats[target],
            "flat": pd.read_parquet(RESULT / "wNB1" / f"galcat_in_{target}.pq"),
        }
    products["wNB1"]["source"]["fixed_specz"] = fixed_specz
    plot_config = PipelineConfig(
        lens_id=3110,
        aperture_subpixels=81,
        convergence_subpixels=161,
        include_hsc_g=True,
        output_rel=RESULT.parent,
    )
    plot_final_photoz_sed(
        plot_config, products, z, common.band.astype(str).tolist(),
        removed_bands=[REMOVED],
    )

    manifest = {
        "release": "PHOTO-LENS032",
        "parent": "PHOTO-LENS031",
        "unchanged_systems": [2, 8, 12, 3336, 3441, 13369, 12971, 13246],
        "changed_system": 3110,
        "excluded_band": REMOVED,
        "lens_modeling_rerun": False,
        "photometry_reextracted": False,
        "photoz_refit": True,
        "p_source_gt_lens": porder,
    }
    (HERE / "photo_lens032_release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
