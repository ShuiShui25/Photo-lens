#!/usr/bin/env python3
"""Compare 50%, 80%, and 100% compact satellite masks for system 12971.

This is an isolated sensitivity test.  It does not modify PHOTO-LENS031's
production regions or production photo-z result.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import bcnz
import numpy as np
import pandas as pd
import xarray as xr
from astropy.io import fits


HERE = Path(__file__).resolve().parent
PL031 = HERE.parent
ROOT = PL031.parent
PL03 = ROOT / "PHOTO-LENS03"
REGION_CODE = PL03 / "region-detection"
PHOTOZ_CODE = PL03 / "photometric-estimation"
TEST = HERE / "satellite-retention-test-12971"
SYSTEM = "12971"
VARIANTS = (("sat050", 0.50), ("sat080", 0.80), ("sat100", 1.00))
REMOVAL_LABEL = "sat100_remove_NB645_NB715"
REMOVED_BANDS = ("NB645", "NB715")
REMOVAL_VARIANTS = (
    (REMOVAL_LABEL, REMOVED_BANDS),
    ("sat100_remove_NB645_NB715_NB845", ("NB645", "NB715", "NB845")),
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare_stable_variant(region, app, label: str, retention: float) -> Path:
    base = REGION_CODE / "outputs" / f"system_{SYSTEM}"
    output = TEST / "stable-inputs" / label / f"system_{SYSTEM}"
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(base, output)

    products = app.products_for(SYSTEM)
    centre = json.loads(
        (products / f"{SYSTEM}_lens_light_centre_wcs.json").read_text()
    )
    source = region.load_product(
        products / f"{SYSTEM}_autolens_lens_light_subtracted_wcs.fits",
        centre,
        region.ANALYSIS_HALF_SIZE_PIX,
    )
    region.SATELLITE_COMPACT_FLUX_RETENTION = retention
    masks, metadata = region.modeled_satellite_masks(products, SYSTEM, source)
    if not masks:
        raise RuntimeError("System 12971 unexpectedly has no modeled satellites")
    union = masks[0].copy()
    for mask in masks[1:]:
        union |= mask
    region.write_ds9_regions(
        output / f"system{SYSTEM}_satellites_v2.reg",
        masks,
        source,
        [f"satellite_{i}_v2" for i in range(len(masks))],
        ["red"] * len(masks),
    )
    region.write_ds9_regions(
        output / f"system{SYSTEM}_satellite_union_v2.reg",
        [union], source, ["satellite_union_v2"], ["red"],
    )
    summary_path = output / f"system{SYSTEM}_v2_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["satellite_exclusion"] = metadata
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return output


def run_regions():
    sys.path.insert(0, str(REGION_CODE))
    region = load(REGION_CODE / "region_selection_v2.py", "pl03_region_sat_test")
    app = load(
        REGION_CODE / "apply_point_fraction_selection.py", "pl03_point_sat_test"
    )
    app.POINT_CONFIG["source_sextractor_overrides"] = {
        SYSTEM: {
            "fixed_detect_sigma": 4.0,
            "grow_sigma_ratio": 0.75,
            "selection_basis": "PHOTO-LENS031 stable source configuration",
        }
    }
    records = []
    for label, retention in VARIANTS:
        stable = prepare_stable_variant(region, app, label, retention)
        app.OUTPUT = TEST / "regions" / label
        record = app.analyse(region, SYSTEM, stable_dir=stable)
        record["variant"] = label
        record["satellite_flux_retention"] = retention
        records.append(record)
    (TEST / "region_test_manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )


def run_photoz():
    sys.path.insert(0, str(PHOTOZ_CODE))
    from paus_lensing034 import PipelineConfig, run_pipeline

    for label, _ in VARIANTS:
        region_dir = TEST / "regions" / label / f"system_{SYSTEM}"
        output_rel = Path("../PHOTO-LENS031/photometric-estimation") / TEST.name / "results" / label
        config = PipelineConfig(
            lens_id=int(SYSTEM),
            aperture_subpixels=81,
            convergence_subpixels=161,
            include_hsc_g=False,
            source_region_override=region_dir / f"system{SYSTEM}_source_point_fraction040.reg",
            background_region_override=region_dir / f"system{SYSTEM}_background_point_fraction040.reg",
            lens_region_override=region_dir / f"system{SYSTEM}_lens_point_fraction040.reg",
            nb_aperture_centre_table=(
                PHOTOZ_CODE / "nb-centres" / f"system_{SYSTEM}"
                / f"system{SYSTEM}_nb_antisymmetry_centres.csv"
            ),
            output_rel=output_rel,
        )
        result = run_pipeline(config)
        print(f"[{label}]\n{result.summary.to_string(index=False)}", flush=True)


def run_sat100_removed_bands(
    label: str = REMOVAL_LABEL,
    removed_bands: tuple[str, ...] = REMOVED_BANDS,
):
    """Refit the sat100 photometry after excluding nominated NB bands."""
    sys.path.insert(0, str(PHOTOZ_CODE))
    from paus_lensing034 import PipelineConfig
    from paus_lensing034.photoz import (
        plot_final_photoz_sed,
        probability_source_behind_lens,
    )
    from bcnz.fit.photoz import minimize_all_z

    base = TEST / "results" / "sat100" / SYSTEM
    output_rel = (
        Path("../PHOTO-LENS031/photometric-estimation") / TEST.name
        / "results" / label
    )
    region_dir = TEST / "regions" / "sat100" / f"system_{SYSTEM}"
    config = PipelineConfig(
        lens_id=int(SYSTEM),
        aperture_subpixels=81,
        convergence_subpixels=161,
        include_hsc_g=False,
        source_region_override=region_dir / f"system{SYSTEM}_source_point_fraction040.reg",
        background_region_override=region_dir / f"system{SYSTEM}_background_point_fraction040.reg",
        lens_region_override=region_dir / f"system{SYSTEM}_lens_point_fraction040.reg",
        nb_aperture_centre_table=(
            PHOTOZ_CODE / "nb-centres" / f"system_{SYSTEM}"
            / f"system{SYSTEM}_nb_antisymmetry_centres.csv"
        ),
        output_rel=output_rel,
    )
    config.result_dir.mkdir(parents=True, exist_ok=True)
    runs = bcnz.config.eriksen2019()
    cached = bcnz.model.cache_model(config.model_dir, runs)
    common = pd.read_csv(base / "common_photoz_bands.csv").band.astype(str).tolist()
    common_keep = [band for band in common if band not in removed_bands]
    fit_keep = [
        f"pau_nb{band[2:]}" if band.startswith("NB") else band
        for band in common_keep
    ]
    models = {
        run: model.sel(band=fit_keep).rename(sed="model")
        for run, model in cached.items()
    }
    products = {"wNB1": {}}
    summaries = []
    for target in ("source", "lens"):
        flat = pd.read_parquet(base / "wNB1" / f"galcat_in_{target}.pq")
        pzcat = bcnz.fit.photoz_flatten(flat, models, dict(runs.EBV), fit_keep)
        pzcat.to_parquet(config.result_dir / f"pzcat_{target}.pq")
        products["wNB1"][target] = {"pzcat": pzcat, "flat": flat}
        row = pzcat.loc[int(SYSTEM)]
        summaries.append({
            "lens_id": int(SYSTEM), "target": target, "nb_weight": 1.0,
            "tag": "wNB1", "zb": float(row.zb), "zb_mean": float(row.zb_mean),
            "odds": float(row.odds), "pz_width": float(row.pz_width),
            "chi2": float(row.chi2), "n_band": int(row.n_band),
            "n_narrow": int(row.n_narrow), "best_run": int(row.best_run),
        })

    z = pd.read_csv(base / "photoz_redshift_grid.csv").z.to_numpy(float)
    pcols = [f"z{i}" for i in range(len(z))]
    lens_row = products["wNB1"]["lens"]["pzcat"].loc[int(SYSTEM)]
    source_row = products["wNB1"]["source"]["pzcat"].loc[int(SYSTEM)]
    order = probability_source_behind_lens(
        lens_row[pcols].to_numpy(float), source_row[pcols].to_numpy(float), z
    )
    for row in summaries:
        row["p_source_gt_lens"] = order

    source_flat = products["wNB1"]["source"]["flat"]
    chi2_all, _, fitted_flux = minimize_all_z(
        source_flat, models, fit_bands=fit_keep, Niter=1000, Nskip=10
    )
    fixed_meta = json.loads((base / "source_fixed_specz_model.json").read_text())
    spec_z = float(fixed_meta["source_spec_z"])
    native_grid = np.asarray(chi2_all.z.values, float)
    fixed_z = float(native_grid[np.argmin(np.abs(native_grid - spec_z))])
    fixed_runs = chi2_all.sel(ref_id=int(SYSTEM), z=fixed_z, method="nearest")
    fixed_run = int(fixed_runs.run.values[int(np.argmin(fixed_runs.values))])
    fixed_chi2 = float(fixed_runs.sel(run=fixed_run).values)
    fixed_model = fitted_flux.sel(
        ref_id=int(SYSTEM), z=fixed_z, run=fixed_run, method="nearest"
    ).sel(band=fit_keep).values.astype(float)
    products["wNB1"]["source"]["fixed_specz"] = {
        "spec_z": spec_z, "grid_z": fixed_z, "best_run": fixed_run,
        "chi2": fixed_chi2, "model": dict(zip(fit_keep, fixed_model)),
    }
    pd.DataFrame({
        "band": fit_keep, "fixed_specz_model_flux": fixed_model,
    }).to_csv(config.result_dir / "source_fixed_specz_model.csv", index=False)
    (config.result_dir / "source_fixed_specz_model.json").write_text(
        json.dumps({
            "source_spec_z": spec_z, "fixed_grid_z": fixed_z,
            "fixed_best_run": fixed_run, "fixed_specz_chi2": fixed_chi2,
            "removed_bands": list(removed_bands),
        }, indent=2) + "\n"
    )
    pd.DataFrame(summaries).to_csv(
        config.result_dir / "photoz_variant_summary.csv", index=False
    )
    pd.DataFrame({"band": common_keep}).to_csv(
        config.result_dir / "common_photoz_bands.csv", index=False
    )
    (config.result_dir / "photoz_redshift_grid.csv").write_text(
        (base / "photoz_redshift_grid.csv").read_text()
    )
    plot_final_photoz_sed(
        config, products, z, common_keep, removed_bands=list(removed_bands)
    )
    (config.result_dir / "run_complete.json").write_text(json.dumps({
        "system": int(SYSTEM), "base_variant": "sat100",
        "removed_bands": list(removed_bands), "p_source_gt_lens": order,
        "all_fluxes_and_errors_unchanged": True,
    }, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def build_report():
    records = json.loads((TEST / "region_test_manifest.json").read_text())
    cards = []
    rows = []
    for record in records:
        label = record["variant"]
        retention = 100 * record["satellite_flux_retention"]
        result = TEST / "results" / label / SYSTEM
        summary = __import__("pandas").read_csv(result / "photoz_variant_summary.csv")
        source = summary[summary.target == "source"].iloc[0]
        lens = summary[summary.target == "lens"].iloc[0]
        order = float(source.p_source_gt_lens)
        fixed = json.loads((result / "source_fixed_specz_model.json").read_text())
        fixed_chi2 = float(fixed["fixed_specz_chi2"])
        delta_chi2 = fixed_chi2 - float(source.chi2)
        rows.append(
            f"<tr><td>{retention:.0f}%</td>"
            f"<td>{record['satellite_original_1sigma_area_pix']}</td>"
            f"<td>{record['satellite_area_pix']}</td>"
            f"<td>{record['final_source_area_pix']}</td>"
            f"<td>{lens.zb:.3f}</td><td>{source.zb:.3f}</td>"
            f"<td>{source.zb_mean:.3f}</td><td>{source.odds:.3f}</td>"
            f"<td>{source.chi2:.2f}</td><td>{fixed_chi2:.2f}</td>"
            f"<td>{delta_chi2:.2f}</td><td>{order:.5f}</td></tr>"
        )
        region_image = (
            TEST / "regions" / label / f"system_{SYSTEM}"
            / f"system_{SYSTEM}_point_fraction_regions.png"
        )
        sed_image = result / f"final_photoz_sed_source_{SYSTEM}_wNB1.png"
        region_rel = region_image.relative_to(TEST).as_posix()
        sed_rel = sed_image.relative_to(TEST).as_posix()
        cards.append(f"""
<section><h2>Satellite retention {retention:.0f}%</h2>
<p>Satellite aperture: {record['satellite_area_pix']} pixels；final source:
{record['final_source_area_pix']} pixels；source photo-z={source.zb:.3f}；
P(z<sub>l</sub>&lt;z<sub>s</sub>)={order:.5f}。</p>
<h3>选区</h3><img src="{region_rel}" alt="{label} regions">
<h3>Source p(z)、flux 与 source spec-z 模型</h3>
<img src="{sed_rel}" alt="{label} source pz and SED">
</section>""")
    for removal_label, removed_bands in REMOVAL_VARIANTS:
        removed_result = TEST / "results" / removal_label / SYSTEM
        if not (removed_result / "run_complete.json").is_file():
            continue
        record = next(item for item in records if item["variant"] == "sat100")
        summary = pd.read_csv(removed_result / "photoz_variant_summary.csv")
        source = summary[summary.target == "source"].iloc[0]
        lens = summary[summary.target == "lens"].iloc[0]
        order = float(source.p_source_gt_lens)
        fixed = json.loads(
            (removed_result / "source_fixed_specz_model.json").read_text()
        )
        fixed_chi2 = float(fixed["fixed_specz_chi2"])
        delta_chi2 = fixed_chi2 - float(source.chi2)
        removed_text = " − ".join(removed_bands)
        rows.append(
            f"<tr><td>100% − {removed_text}</td>"
            f"<td>{record['satellite_original_1sigma_area_pix']}</td>"
            f"<td>{record['satellite_area_pix']}</td>"
            f"<td>{record['final_source_area_pix']}</td>"
            f"<td>{lens.zb:.3f}</td><td>{source.zb:.3f}</td>"
            f"<td>{source.zb_mean:.3f}</td><td>{source.odds:.3f}</td>"
            f"<td>{source.chi2:.2f}</td><td>{fixed_chi2:.2f}</td>"
            f"<td>{delta_chi2:.2f}</td><td>{order:.5f}</td></tr>"
        )
        region_image = (
            TEST / "regions" / "sat100" / f"system_{SYSTEM}"
            / f"system_{SYSTEM}_point_fraction_regions.png"
        ).relative_to(TEST).as_posix()
        sed_image = (
            removed_result / f"final_photoz_sed_source_{SYSTEM}_wNB1.png"
        ).relative_to(TEST).as_posix()
        removed_cn = "、".join(removed_bands)
        cards.append(f"""
<section><h2>Satellite retention 100% − {removed_text}</h2>
<p>选区与 satellite 100% 完全相同；仅从 BCNz likelihood 和图中数据点移除
{removed_cn}。source photo-z={source.zb:.3f}；
P(z<sub>l</sub>&lt;z<sub>s</sub>)={order:.5f}。</p>
<h3>选区（与 100% 相同）</h3><img src="{region_image}" alt="sat100 regions">
<h3>Source p(z)、移除后 flux 与 source spec-z 模型</h3>
<img src="{sed_image}" alt="sat100 removed bands source pz and SED">
</section>""")
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>System 12971 satellite-retention photo-z test</title>
<style>body{{font-family:Arial,sans-serif;max-width:1500px;margin:auto;padding:24px;line-height:1.55}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:8px;text-align:center}}
th{{background:#eee}}section{{border-top:2px solid #777;margin-top:35px}}
img{{display:block;max-width:100%;height:auto;margin:10px auto 28px}}
.note{{background:#eef6ff;border-left:5px solid #3978b8;padding:12px}}</style></head><body>
<h1>System 12971：satellite 选区保留率测试</h1>
<div class='note'>仅改变独立 satellite-light 模型选区所保留的 1σ-mask 正通量比例：
50%、80%、100%，并增加 100% 选区下排除 NB645+NB715，以及进一步排除
NB845 的测试。
Source SExtractor 固定为 4.0σ + growth ratio 0.75，
point-reflection source fraction 阈值固定为 0.4；lens region、NB 中心修正、
测光和 BCNz 设置保持一致。50% 是 PHOTO-LENS031 原稳定配置。</div>
<h2>数值汇总</h2><table><thead><tr><th>satellite retention</th>
<th>原1σ面积</th><th>satellite面积</th><th>最终source面积</th>
<th>lens z</th><th>source z</th><th>source mean-z</th><th>source odds</th>
<th>best χ²</th><th>spec-z χ²</th><th>Δχ²(spec-best)</th>
<th>P(zl&lt;zs)</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{''.join(cards)}</body></html>"""
    report = TEST / "system12971_satellite_retention_photoz_report.html"
    report.write_text(html, encoding="utf-8")
    print(report)


def main():
    TEST.mkdir(parents=True, exist_ok=True)
    run_regions()
    run_photoz()
    for label, removed_bands in REMOVAL_VARIANTS:
        run_sat100_removed_bands(label, removed_bands)
    build_report()


if __name__ == "__main__":
    main()
