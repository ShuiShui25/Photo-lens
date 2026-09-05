from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bcnz
import xarray as xr
from bcnz.fit.photoz import minimize_all_z

from .config import PipelineConfig


def _common_catalogues(
    config: PipelineConfig,
    source: pd.DataFrame,
    lens: pd.DataFrame,
) -> tuple[list[str], dict[str, pd.DataFrame], pd.DataFrame]:
    tables = {"source": source, "lens": lens}
    usable = {}
    for target, table in tables.items():
        usable[target] = set(table.loc[
            np.isfinite(table.flux)
            & np.isfinite(table.flux_error)
            & (table.flux_error > 0),
            "band",
        ])
    common = [
        band for band in config.ordered_bands
        if band not in config.excluded_bands and all(band in usable[target] for target in tables)
    ]
    if not any(x.startswith("NB") for x in common):
        raise RuntimeError("No common narrow bands remain")
    if not any(x in config.bb_names for x in common):
        raise RuntimeError(f"No common {config.bb_survey} broad bands remain")
    selected = {
        target: table[table.band.isin(common)].copy()
        for target, table in tables.items()
    }
    status_rows = []
    for band in config.ordered_bands:
        source_usable = band in usable["source"]
        lens_usable = band in usable["lens"]
        reasons = []
        if band in config.excluded_bands:
            reasons.append("Photo-lens fixed exclusion")
        if not source_usable:
            reasons.append("source missing/non-finite/error<=0")
        if not lens_usable:
            reasons.append("lens missing/non-finite/error<=0")
        status_rows.append({
            "band": band,
            "kind": "NB" if band.startswith("NB") else "BB",
            "source_usable": source_usable,
            "lens_usable": lens_usable,
            "effective_common": band in common,
            "exclusion_reason": "; ".join(reasons),
        })
    return common, selected, pd.DataFrame(status_rows)


def _to_paus_nb_names(names: list[str]) -> list[str]:
    return [f"pau_nb{int(name.removeprefix('NB'))}" for name in names]


def _long_catalog_to_bcnz(
    long_path: Path,
    bb_names: list[str],
) -> pd.DataFrame:
    nb_data = bcnz.data.load_coadd_file(long_path)
    parent = pd.read_csv(long_path)
    parent = parent[parent.band.isin(bb_names)]
    error_names = [f"D{x}" for x in bb_names]
    error = (
        parent.pivot(index="ref_id", columns="band", values="flux_error")
        .rename(columns=dict(zip(bb_names, error_names)))
    )
    flux = parent.pivot(index="ref_id", columns="band", values="flux")
    wide = flux.join(error, how="inner")
    bb_flux = wide[bb_names]
    bb_error = wide[error_names].rename(columns=dict(zip(error_names, bb_names)))
    hierarchical = pd.concat({"flux": bb_flux, "flux_error": bb_error}, axis=1)
    return bcnz.data.fix_noise(nb_data.join(hierarchical, how="inner"))


def _apply_nb_weight(
    flat: pd.DataFrame,
    fit_bands: list[str],
    weight: float,
) -> pd.DataFrame:
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError(f"Invalid NB weight: {weight}")
    weighted = flat.copy()
    columns = [
        f"flux_error_{band}" for band in fit_bands if band.startswith("pau_nb")
    ]
    weighted.loc[:, columns] = weighted.loc[:, columns] / np.sqrt(weight)
    return weighted


def _weight_tag(weight: float) -> str:
    return f"wNB{weight:g}".replace(".", "p").replace("-", "m")


def _signed_pulls(
    pz_row: pd.Series,
    flat_row: pd.Series,
    fit_bands: list[str],
) -> list[dict]:
    rows = []
    for band in fit_bands:
        flux_col = f"flux_{band}"
        error_col = f"flux_error_{band}"
        model_col = f"model_{band}"
        if not all(x in flat_row.index for x in (flux_col, error_col)):
            continue
        if model_col not in pz_row.index:
            continue
        flux = float(flat_row[flux_col])
        error = float(flat_row[error_col])
        model = float(pz_row[model_col])
        if not (np.isfinite(flux) and np.isfinite(error) and error > 0 and np.isfinite(model)):
            continue
        rows.append({
            "band": band,
            "kind": "NB" if band.startswith("pau_nb") else "BB",
            "flux": flux,
            "flux_error": error,
            "model_flux": model,
            "pull": (flux - model) / error,
        })
    return rows


def run_photoz_variants(
    config: PipelineConfig,
    source: pd.DataFrame,
    lens: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, list[str], pd.DataFrame]:
    common, catalogues, band_status = _common_catalogues(config, source, lens)
    common_nb = [x for x in common if x.startswith("NB")]
    common_bb = [x for x in common if x in config.bb_names]
    fit_bands = _to_paus_nb_names(common_nb) + common_bb

    pd.DataFrame({
        "band": common,
        "kind": ["NB" if x.startswith("NB") else "BB" for x in common],
    }).to_csv(config.result_dir / "common_photoz_bands.csv", index=False)
    band_status.to_csv(
        config.result_dir / "photoz_band_selection_status.csv", index=False
    )
    for target, table in catalogues.items():
        table.to_csv(
            config.result_dir / f"photometry_bcnz_{target}_{config.lens_id}.csv",
            index=False,
        )

    runs = bcnz.config.eriksen2019()
    raw_models = bcnz.model.cache_model(config.model_dir, runs)
    if config.include_hsc_g:
        if config.bb_survey != "HSC":
            raise ValueError("include_hsc_g is valid only for HSC systems")
        g_models = bcnz.model.cache_model(config.hsc_g_model_dir, runs)
        for run in raw_models:
            if list(g_models[run].band.values) != ["subaru_g"]:
                raise ValueError("HSC-G cache must contain exactly band=subaru_g")
            keep = raw_models[run].band.values != "subaru_g"
            raw_models[run] = xr.concat(
                [raw_models[run].isel(band=keep), g_models[run]], dim="band"
            )
    model_d = {
        run: model.sel(band=fit_bands).rename(sed="model")
        for run, model in raw_models.items()
    }
    redshift_grid = np.asarray(next(iter(model_d.values())).z.values, dtype=float)
    pd.DataFrame({"z": redshift_grid}).to_csv(
        config.result_dir / "photoz_redshift_grid.csv", index=False
    )

    unweighted_flat = {}
    for target, table in catalogues.items():
        path = config.result_dir / f"photometry_bcnz_{target}_{config.lens_id}.csv"
        galcat = _long_catalog_to_bcnz(path, common_bb)
        if config.calibrate_bcnz:
            zp = bcnz.calib.cache_zp(
                config.result_dir, galcat, raw_models, fit_bands
            )
            galcat = bcnz.calib.apply_zp(
                galcat, zp, norm_filter=bcnz.data.catalogs.rband(config.field)
            )
        unweighted_flat[target] = bcnz.fit.flatten_input(galcat)

    summaries = []
    products = {}
    flats_by_weight = {}
    for weight in config.nb_weights:
        tag = _weight_tag(weight)
        variant_dir = config.result_dir / tag
        variant_dir.mkdir(parents=True, exist_ok=True)
        products[tag] = {}
        flats_by_weight[weight] = {}
        for target, flat0 in unweighted_flat.items():
            flat = _apply_nb_weight(flat0, fit_bands, weight)
            flat_path = variant_dir / f"galcat_in_{target}.pq"
            pz_path = variant_dir / f"pzcat_{target}.pq"
            flat.to_parquet(flat_path)
            pzcat = bcnz.fit.photoz_flatten(
                flat, model_d, dict(runs.EBV), fit_bands
            )
            pzcat.to_parquet(pz_path)
            flats_by_weight[weight][target] = flat

            pz_row = pzcat.loc[config.lens_id]
            flat_row = flat.loc[config.lens_id]
            pulls = _signed_pulls(pz_row, flat_row, fit_bands)
            pull_table = pd.DataFrame(pulls)
            pull_table.insert(0, "target", target)
            pull_table.insert(0, "nb_weight", weight)
            pull_table.to_csv(variant_dir / f"signed_pulls_{target}.csv", index=False)
            nb_pull = pull_table.loc[pull_table.kind == "NB", "pull"].to_numpy()
            bb_pull = pull_table.loc[pull_table.kind == "BB", "pull"].to_numpy()
            summaries.append({
                "lens_id": config.lens_id,
                "target": target,
                "nb_weight": weight,
                "tag": tag,
                "zb": float(pz_row.zb),
                "zb_mean": float(pz_row.zb_mean),
                "odds": float(pz_row.odds),
                "pz_width": float(pz_row.pz_width),
                "chi2": float(pz_row.chi2),
                "n_band": int(pz_row.n_band),
                "n_narrow": int(pz_row.n_narrow),
                "best_run": int(pz_row.best_run),
                "chi2_per_band": float(pz_row.chi2 / pz_row.n_band),
                "pull_std_nb": (
                    float(np.std(nb_pull, ddof=1)) if nb_pull.size > 1 else np.nan
                ),
                "pull_std_bb": (
                    float(np.std(bb_pull, ddof=1)) if bb_pull.size > 1 else np.nan
                ),
                "signed_bb_pulls": json.dumps(
                    {
                        row["band"]: round(float(row["pull"]), 6)
                        for row in pulls if row["kind"] == "BB"
                    },
                    sort_keys=True,
                ),
            })
            products[tag][target] = {"pzcat": pzcat, "flat": flat}
            if target == "source" and np.isclose(weight, config.official_nb_weight):
                spec_values = _spec_redshifts(config)
                if "source" in spec_values:
                    chi2_all, _, fitted_flux = minimize_all_z(
                        flat, model_d, fit_bands=fit_bands, Niter=1000, Nskip=10
                    )
                    native_grid = np.asarray(chi2_all.z.values, dtype=float)
                    spec_z = float(spec_values["source"])
                    fixed_z = float(native_grid[np.argmin(np.abs(native_grid - spec_z))])
                    fixed_runs = chi2_all.sel(
                        ref_id=config.lens_id, z=fixed_z, method="nearest"
                    )
                    fixed_run = int(
                        fixed_runs.run.values[int(np.argmin(fixed_runs.values))]
                    )
                    fixed_chi2 = float(fixed_runs.sel(run=fixed_run).values)
                    fixed_model = fitted_flux.sel(
                        ref_id=config.lens_id, z=fixed_z, run=fixed_run,
                        method="nearest",
                    ).sel(band=fit_bands).values.astype(float)
                    products[tag][target]["fixed_specz"] = {
                        "spec_z": spec_z,
                        "grid_z": fixed_z,
                        "best_run": fixed_run,
                        "chi2": fixed_chi2,
                        "model": dict(zip(fit_bands, fixed_model)),
                    }
                    pd.DataFrame({
                        "band": fit_bands,
                        "fixed_specz_model_flux": fixed_model,
                    }).to_csv(
                        config.result_dir / "source_fixed_specz_model.csv",
                        index=False,
                    )
                    (config.result_dir / "source_fixed_specz_model.json").write_text(
                        json.dumps({
                            "source_spec_z": spec_z,
                            "fixed_grid_z": fixed_z,
                            "fixed_best_run": fixed_run,
                            "fixed_specz_chi2": fixed_chi2,
                        }, indent=2) + "\n"
                    )

    if 1.0 in flats_by_weight and 2.0 in flats_by_weight:
        for target in unweighted_flat:
            one = flats_by_weight[1.0][target]
            two = flats_by_weight[2.0][target]
            nb_columns = [
                f"flux_error_{band}" for band in fit_bands
                if band.startswith("pau_nb")
            ]
            bb_columns = [
                f"flux_error_{band}" for band in fit_bands
                if not band.startswith("pau_nb")
            ]
            if not np.allclose(
                one[nb_columns].to_numpy(),
                np.sqrt(2.0) * two[nb_columns].to_numpy(),
                rtol=1e-10, atol=1e-12, equal_nan=True,
            ):
                raise AssertionError(f"NB weight isolation failed for {target}")
            if not np.allclose(
                one[bb_columns].to_numpy(), two[bb_columns].to_numpy(),
                rtol=1e-10, atol=1e-12, equal_nan=True,
            ):
                raise AssertionError(f"BB errors changed with NB weight for {target}")

    order_rows = []
    pz_columns = [f"z{i}" for i in range(len(redshift_grid))]
    for weight in config.nb_weights:
        tag = _weight_tag(weight)
        lens_pz = products[tag]["lens"]["pzcat"].loc[
            config.lens_id, pz_columns
        ].to_numpy(dtype=float)
        source_pz = products[tag]["source"]["pzcat"].loc[
            config.lens_id, pz_columns
        ].to_numpy(dtype=float)
        order_rows.append({
            "lens_id": config.lens_id,
            "nb_weight": weight,
            "tag": tag,
            "p_source_gt_lens": probability_source_behind_lens(
                lens_pz, source_pz, redshift_grid
            ),
            "definition": "integral dz_l p_l(z_l) integral_{z_l}^infinity dz_s p_s(z_s)",
            "strict_inequality": True,
            "redshift_grid_points": len(redshift_grid),
        })
    order_table = pd.DataFrame(order_rows)
    order_table.to_csv(
        config.result_dir / "redshift_order_probability.csv", index=False
    )

    summary = pd.DataFrame(summaries).merge(
        order_table[["nb_weight", "p_source_gt_lens"]],
        on="nb_weight", how="left", validate="many_to_one",
    )
    summary.to_csv(config.result_dir / "photoz_variant_summary.csv", index=False)
    return summary, products, common, band_status


def probability_source_behind_lens(
    lens_probability: np.ndarray,
    source_probability: np.ndarray,
    redshift_grid: np.ndarray,
) -> float:
    """Integrate the normalized lens/source p(z) over z_source > z_lens."""
    lens = np.asarray(lens_probability, dtype=float)
    source = np.asarray(source_probability, dtype=float)
    redshift = np.asarray(redshift_grid, dtype=float)
    if (
        lens.ndim != 1 or source.ndim != 1 or redshift.ndim != 1
        or lens.shape != source.shape or lens.shape != redshift.shape
    ):
        raise ValueError("Lens p(z), source p(z), and redshift grid must be matching 1D arrays")
    if (
        not np.isfinite(lens).all() or not np.isfinite(source).all()
        or not np.isfinite(redshift).all()
    ):
        raise ValueError("Lens p(z), source p(z), and redshift grid must be finite")
    if len(redshift) < 2 or np.any(np.diff(redshift) <= 0.0):
        raise ValueError("Redshift grid must be strictly increasing with at least two points")
    if np.any(lens < 0.0) or np.any(source < 0.0):
        raise ValueError("Lens and source p(z) must be non-negative")
    lens_norm = float(np.trapz(lens, redshift))
    source_norm = float(np.trapz(source, redshift))
    if lens_norm <= 0.0 or source_norm <= 0.0:
        raise ValueError("Lens and source p(z) must have positive normalization")
    lens = lens / lens_norm
    source = source / source_norm
    dz = np.diff(redshift)
    source_segments = 0.5 * (source[:-1] + source[1:]) * dz
    source_upper_tail = np.concatenate((
        np.cumsum(source_segments[::-1])[::-1], np.asarray([0.0])
    ))
    value = float(np.trapz(lens * source_upper_tail, redshift))
    return float(np.clip(value, 0.0, 1.0))


def _spec_redshifts(config: PipelineConfig) -> dict[str, float]:
    catalogue = pd.read_csv(config.catalog_path)
    selected = catalogue[catalogue.lens_id == config.lens_id]
    values = {}
    for target, column in (("lens", "zl_spec"), ("source", "zs_spec")):
        finite = pd.to_numeric(selected[column], errors="coerce").dropna()
        if len(finite):
            values[target] = float(np.median(finite.to_numpy(dtype=float)))
    return values


def _mark_spec_redshift(axis, values: dict[str, float], target: str) -> None:
    styles = {
        "lens": ("crimson", "lens spec-z"),
        "source": ("purple", "source spec-z"),
    }
    if target not in values:
        axis.text(
            0.98, 0.92, f"{target} spec-z unavailable in catalogue",
            transform=axis.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.8},
        )
        return
    color, label = styles[target]
    axis.axvline(
        values[target], color=color, linestyle="--", linewidth=1.3,
        alpha=0.85, label=f"{label}={values[target]:.4g}",
    )


def plot_final_photoz_sed(
    config: PipelineConfig,
    products: dict,
    redshift_grid: np.ndarray,
    common_bands: list[str],
    removed_bands: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Save separate official p(z)+SED figures for source and lens."""
    tag = _weight_tag(config.official_nb_weight)
    targets = ["lens", "source"]
    colors = {"source": "tab:blue", "lens": "tab:red"}
    fit_bands = _to_paus_nb_names(
        [band for band in common_bands if band.startswith("NB")]
    ) + [band for band in common_bands if band in config.bb_names]
    bb_wavelengths = {
        "subaru_g": 477.0,
        "subaru_r": 621.84,
        "subaru_i": 772.70,
        "subaru_z": 890.85,
        "subaru_y": 977.51,
        "cfht_u": 374.0,
        "cfht_g": 487.0,
        "cfht_r": 625.0,
        "cfht_i": 767.0,
        "cfht_z": 915.0,
    }

    pz_columns = [f"z{i}" for i in range(len(redshift_grid))]
    spec_redshifts = _spec_redshifts(config)
    outputs = {}
    for target in targets:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
        row = products[tag][target]["pzcat"].loc[config.lens_id]
        axes[0].plot(
            redshift_grid, row[pz_columns].to_numpy(dtype=float),
            color=colors[target], label=f"{target}: zb={row.zb:.3f}",
        )
        _mark_spec_redshift(axes[0], spec_redshifts, target)
        axes[0].set(
            xlabel="redshift z", ylabel="p(z)",
            title=(f"System {config.lens_id}: {target} BCNz p(z) "
                   f"({tag}, official candidate)"),
        )
        axes[0].grid(alpha=0.25)
        axes[0].legend(fontsize=8)

        pz_row = products[tag][target]["pzcat"].loc[config.lens_id]
        flat_row = products[tag][target]["flat"].loc[config.lens_id]
        wavelengths = np.asarray([
            float(band.removeprefix("pau_nb"))
            if band.startswith("pau_nb") else bb_wavelengths[band]
            for band in fit_bands
        ])
        observed = np.asarray(
            [flat_row[f"flux_{band}"] for band in fit_bands], dtype=float
        )
        errors = np.asarray(
            [flat_row[f"flux_error_{band}"] for band in fit_bands], dtype=float
        )
        model = np.asarray(
            [pz_row[f"model_{band}"] for band in fit_bands], dtype=float
        )
        is_nb = np.asarray([band.startswith("pau_nb") for band in fit_bands])
        is_bb = ~is_nb
        order = np.argsort(wavelengths)
        axes[1].errorbar(
            wavelengths[is_nb], observed[is_nb], yerr=errors[is_nb],
            fmt="o", ms=4, color="tab:blue", ecolor="tab:blue", capsize=2,
            linestyle="none", label=f"effective PAUS NB values ({is_nb.sum()})",
        )
        axes[1].errorbar(
            wavelengths[is_bb], observed[is_bb], yerr=errors[is_bb],
            fmt="s", ms=7, markerfacecolor="none", markeredgewidth=1.5,
            color="tab:orange", ecolor="tab:orange", capsize=3,
            linestyle="none", label=f"effective {config.bb_survey} BB values ({is_bb.sum()})",
        )
        axes[1].plot(
            wavelengths[order], model[order], "--", color="black", lw=1.4,
            label="BCNz best model (chi-square flux)",
        )
        for removed_index, removed_band in enumerate(removed_bands or []):
            removed_wavelength = (
                float(removed_band.removeprefix("NB"))
                if removed_band.startswith("NB")
                else bb_wavelengths[removed_band]
            )
            axes[1].axvline(
                removed_wavelength,
                color="crimson",
                linestyle=":",
                linewidth=1.4,
                alpha=0.85,
                label=(
                    f"removed: {', '.join(removed_bands)}"
                    if removed_index == 0 else None
                ),
            )
            axes[1].annotate(
                f"{removed_band} removed",
                xy=(removed_wavelength, 0.98),
                xycoords=("data", "axes fraction"),
                xytext=(3, 0),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color="crimson",
            )
        fixed = products[tag][target].get("fixed_specz")
        if target == "source" and fixed is not None:
            fixed_model = np.asarray(
                [fixed["model"][band] for band in fit_bands], dtype=float
            )
            axes[1].plot(
                wavelengths[order], fixed_model[order], "-", color="purple",
                lw=1.7,
                label=(f"BCNz model at source spec-z={fixed['spec_z']:.4g} "
                       f"(grid {fixed['grid_z']:.3f})"),
            )
        for index in np.flatnonzero(is_bb):
            axes[1].annotate(
                fit_bands[index].removeprefix("cfht_").removeprefix("subaru_"),
                (wavelengths[index], observed[index]), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=8,
                color="tab:orange",
            )
        axes[1].set(
            xlabel="central wavelength [nm]", ylabel="flux",
            title=(f"{target.capitalize()} effective-band measurements and "
                   f"best-fit model (zb={pz_row.zb:.3f})"),
        )
        axes[1].grid(alpha=0.2)
        axes[1].legend(fontsize=8)

        output = (
            config.result_dir
            / f"final_photoz_sed_{target}_{config.lens_id}_{tag}.png"
        )
        fig.savefig(output, dpi=200)
        plt.close(fig)
        outputs[target] = output
    return outputs
