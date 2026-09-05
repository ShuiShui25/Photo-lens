#!/usr/bin/env python3
"""Build the PHOTO-LENS032 all-HSC/CFHT stable photo-z report."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, SqrtStretch
from astropy.wcs import WCS
from matplotlib.lines import Line2D
from regions import Regions


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT
RESULTS = HERE / "results"
REGIONS = ROOT / "region-detection" / "stable-regions"
ASSETS = HERE / "report-assets"
OUTPUT = HERE / "all_systems_photoz_report.html"
SYSTEMS = (2, 8, 12, 3110, 3336, 3441, 13369, 12971, 13246)


def rel(path: Path) -> str:
    return Path(os.path.relpath(path, OUTPUT.parent)).as_posix()


def probability_mass(row: pd.Series, n_z: int) -> np.ndarray:
    values = np.asarray([row[f"z{i}"] for i in range(n_z)], dtype=float)
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("p(z) has no positive finite mass")
    return values / total


def unequal_grid_probability(lens: np.ndarray, source: np.ndarray) -> float:
    return float(np.clip(1.0 - np.dot(lens, source), 0.0, 1.0))


def load_spec_redshifts(system: int) -> tuple[float | None, float | None]:
    catalogue = pd.read_csv(
        SOURCE / "data/catalogue/PAUS_lenses_crossmatch_v3_SG.csv"
    )
    selected = catalogue.loc[catalogue.lens_id.eq(system)]
    values = []
    for column in ("zl_spec", "zs_spec"):
        finite = pd.to_numeric(selected[column], errors="coerce").dropna()
        values.append(float(np.median(finite)) if len(finite) else None)
    return values[0], values[1]


def overlay_region(axis, path: Path, wcs: WCS, color: str, linestyle: str) -> None:
    if not path.is_file():
        return
    for region in Regions.read(path, format="ds9"):
        region.to_pixel(wcs).plot(
            ax=axis, facecolor="none", edgecolor=color,
            linewidth=2.0, linestyle=linestyle,
        )


def region_pixel_bounds(paths: list[Path], wcs: WCS, shape: tuple[int, int]):
    xs, ys = [], []
    for path in paths:
        if not path.is_file():
            continue
        for region in Regions.read(path, format="ds9"):
            pixel = region.to_pixel(wcs)
            box = pixel.bounding_box
            xs.extend([box.ixmin, box.ixmax])
            ys.extend([box.iymin, box.iymax])
    if not xs:
        return 0, shape[1] - 1, 0, shape[0] - 1
    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    half = max(0.55 * (max(xs) - min(xs)), 0.55 * (max(ys) - min(ys)), 12.0)
    return cx - half, cx + half, cy - half, cy + half


def science_image(system: int):
    config_path = SOURCE / "lens-modeling" / f"system_{system}" / "config.json"
    config = json.loads(config_path.read_text())
    science = Path(config["science_cutout"])
    if not science.is_absolute():
        science = (config_path.parent / science).resolve()
    with fits.open(science) as hdul:
        hdu = hdul[int(config.get("science_hdu", 0))]
        return (
            np.asarray(hdu.data, dtype=float), hdu.header.copy(),
            str(config["bb_survey"]), str(config["bb_band"]), science,
        )


def system_metrics(system: int) -> dict:
    result = RESULTS / str(system)
    summary = pd.read_csv(result / "photoz_variant_summary.csv")
    lens_summary = summary.loc[summary.target.eq("lens")].iloc[0]
    source_summary = summary.loc[summary.target.eq("source")].iloc[0]
    z = pd.read_csv(result / "photoz_redshift_grid.csv").z.to_numpy(float)
    lens_pz = pd.read_parquet(result / "wNB1/pzcat_lens.pq").loc[system]
    source_pz = pd.read_parquet(result / "wNB1/pzcat_source.pq").loc[system]
    lens_mass = probability_mass(lens_pz, len(z))
    source_mass = probability_mass(source_pz, len(z))
    common = pd.read_csv(result / "common_photoz_bands.csv").band.astype(str).tolist()
    effective_nb = [band for band in common if band.startswith("NB")]
    zl_spec, zs_spec = load_spec_redshifts(system)
    data, header, survey, band, science = science_image(system)
    return {
        "system": system,
        "result": result,
        "z": z,
        "lens_pz": lens_pz,
        "source_pz": source_pz,
        "lens_mass": lens_mass,
        "source_mass": source_mass,
        "zl_photo": float(lens_summary.zb),
        "zs_photo": float(source_summary.zb),
        "zl_spec": zl_spec,
        "zs_spec": zs_spec,
        "p_neq_grid": unequal_grid_probability(lens_mass, source_mass),
        "p_lens_lt_source": float(source_summary.p_source_gt_lens),
        "effective_nb": effective_nb,
        "common": common,
        "data": data,
        "header": header,
        "survey": survey,
        "band": band,
        "science": science,
    }


def plot_system(record: dict) -> Path:
    system = record["system"]
    region_dir = REGIONS / f"system_{system}"
    paths = {
        "lens": region_dir / f"system{system}_lens_point_fraction040.reg",
        "source": region_dir / f"system{system}_source_point_fraction040.reg",
        "background": region_dir / f"system{system}_background_point_fraction040.reg",
    }
    wcs = WCS(record["header"]).celestial
    finite = record["data"][np.isfinite(record["data"])]
    norm = ImageNormalize(
        finite, interval=PercentileInterval(99.5), stretch=SqrtStretch(), clip=True
    )
    fig = plt.figure(figsize=(15.2, 6.4), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.55))
    image_axis = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, hspace=0.12)
    lens_axis = fig.add_subplot(right[0, 0])
    source_axis = fig.add_subplot(right[1, 0], sharex=lens_axis)

    image_axis.imshow(record["data"], origin="lower", cmap="gray", norm=norm)
    overlay_region(image_axis, paths["lens"], wcs, "magenta", "-")
    overlay_region(image_axis, paths["source"], wcs, "lime", "-")
    overlay_region(image_axis, paths["background"], wcs, "cyan", "--")
    xmin, xmax, ymin, ymax = region_pixel_bounds(list(paths.values()), wcs, record["data"].shape)
    image_axis.set_xlim(xmin, xmax)
    image_axis.set_ylim(ymin, ymax)
    image_axis.set_aspect("equal")
    image_axis.set_title(f"Original {record['survey']} {record['band']} data")
    image_axis.set_xlabel("FITS x pixel")
    image_axis.set_ylabel("FITS y pixel")
    image_axis.legend(handles=[
        Line2D([0], [0], color="magenta", lw=2, label="lens region"),
        Line2D([0], [0], color="lime", lw=2, label="source region"),
        Line2D([0], [0], color="cyan", lw=2, ls="--", label="reflected background"),
    ], loc="upper right", fontsize=8)

    z = record["z"]
    lens_density = record["lens_mass"] / np.trapz(record["lens_mass"], z)
    source_density = record["source_mass"] / np.trapz(record["source_mass"], z)
    lens_axis.plot(z, lens_density, color="crimson", lw=1.7,
                   label=f"lens photo-z={record['zl_photo']:.3f}")
    source_axis.plot(z, source_density, color="tab:blue", lw=1.7,
                     label=f"source photo-z={record['zs_photo']:.3f}")
    if record["zl_spec"] is not None:
        lens_axis.axvline(record["zl_spec"], color="crimson", ls="--", lw=1.4,
                          label=f"lens spec-z={record['zl_spec']:.4g}")
    if record["zs_spec"] is not None:
        source_axis.axvline(record["zs_spec"], color="purple", ls="--", lw=1.4,
                            label=f"source spec-z={record['zs_spec']:.4g}")
    for axis, title in ((lens_axis, "Lens p(z)"), (source_axis, "Source p(z)")):
        axis.set_ylabel("normalized p(z)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    source_axis.set_xlabel("redshift z")
    lens_axis.tick_params(labelbottom=False)
    fig.suptitle(
        f"System {system}: PHOTO-LENS032 | Pgrid(zl != zs)={record['p_neq_grid']:.5f} | "
        f"P(zl < zs)={record['p_lens_lt_source']:.5f}", fontsize=14,
    )
    ASSETS.mkdir(parents=True, exist_ok=True)
    output = ASSETS / f"system_{system}_regions_and_pz.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def fmt(value, digits=4):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def main() -> None:
    records = [system_metrics(system) for system in SYSTEMS]
    for record in records:
        record["figure"] = plot_system(record)
    summary = pd.DataFrame([{
        "system": row["system"], "survey": row["survey"], "modeling_band": row["band"],
        "zl_photoz": row["zl_photo"], "zs_photoz": row["zs_photo"],
        "zl_spec": row["zl_spec"], "zs_spec": row["zs_spec"],
        "p_grid_zl_not_equal_zs": row["p_neq_grid"],
        "p_zl_less_than_zs": row["p_lens_lt_source"],
        "n_effective_nb": len(row["effective_nb"]),
        "effective_nb": ",".join(row["effective_nb"]),
    } for row in records])
    summary.to_csv(HERE / "all_systems_photoz_summary.csv", index=False)

    overview = "".join(
        f"<tr><td><a href='#s{row['system']}'>{row['system']}</a></td>"
        f"<td>{row['survey']}</td><td>{row['band']}</td>"
        f"<td>{fmt(row['zl_photo'],3)}</td><td>{fmt(row['zs_photo'],3)}</td>"
        f"<td>{fmt(row['p_neq_grid'],5)}</td><td>{fmt(row['p_lens_lt_source'],5)}</td>"
        f"<td>{len(row['effective_nb'])}</td></tr>" for row in records
    )
    sections = "".join(
        f"<section id='s{row['system']}'><h2>System {row['system']} — "
        f"{row['survey']} ({row['band']})</h2>"
        f"<div class='metrics'><b>z<sub>l,phot</sub>={fmt(row['zl_photo'],3)}; "
        f"z<sub>s,phot</sub>={fmt(row['zs_photo'],3)}; "
        f"P<sub>grid</sub>(z<sub>l</sub>≠z<sub>s</sub>)={fmt(row['p_neq_grid'],5)}; "
        f"P(z<sub>l</sub>&lt;z<sub>s</sub>)={fmt(row['p_lens_lt_source'],5)}; "
        f"spec-z lens/source={fmt(row['zl_spec'],4)}/{fmt(row['zs_spec'],4)}</div>"
        f"<div class='bands'><b>Effective NB ({len(row['effective_nb'])}):</b> "
        f"{html.escape(', '.join(row['effective_nb']) or 'none')}</div>"
        f"<img src='{rel(row['figure'])}' alt='system {row['system']} regions and pz'>"
        f"</section>" for row in records
    )
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>PHOTO-LENS032 stable HSC+CFHT photo-z report</title><style>
body{{font-family:system-ui,sans-serif;max-width:1550px;margin:24px auto;background:#f4f6f9;color:#17202a}}
section{{background:white;padding:18px;margin:24px 0;border-radius:10px;box-shadow:0 1px 5px #ccd}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #bbc;padding:7px;text-align:center}}
th{{background:#e5edf7}}img{{display:block;width:100%;height:auto;margin-top:12px}}
.note{{background:#eef6ff;border-left:5px solid #3978b8;padding:12px;margin:12px 0}}
.metrics{{background:#edf8ed;border-left:5px solid #428b46;padding:10px}}
.bands{{background:#faf5e8;border-left:5px solid #b98725;padding:10px;margin-top:10px;overflow-wrap:anywhere}}
code{{background:#eee;padding:2px 5px}}</style></head><body>
<h1>PHOTO-LENS032：稳定版 HSC + CFHT photo-z 结果</h1>
<div class='note'>所有系统使用正确的 AutoLens lens-light-subtracted source detection image，固定
<code>DETECT_THRESH=4.0σ</code>、hysteresis growth ratio=<code>0.75</code>。HSC 使用
PAUS NB + 官方 HSC-G + HSC r/i/z/y；CFHT 使用 PAUS NB + CFHT u/g/r/i/z。所有结果均为 wNB1。
<br><br><b>032 相对 031 的唯一变化：</b>系统 3110 在 lens/source BCNz 拟合中同时排除 NB535；
其余系统、选区和测光保持不变，且没有重新进行透镜建模。
<br><br><b>P(zl≠zs) 定义：</b>连续变量严格不相等的概率恒为1，因此这里报告离散BCNz网格定义
<code>1-sum_i q_l(i)q_s(i)</code>，它依赖网格分辨率。透镜验证应主要参考积分排序概率
<code>P(zl&lt;zs)</code>。</div>
<p><a href='all_systems_photoz_summary.csv'>下载机器可读汇总CSV</a> · <a href='../README.md'>版本说明</a></p>
<table><thead><tr><th>system</th><th>survey</th><th>modeling BB</th><th>zl photo-z</th>
<th>zs photo-z</th><th>Pgrid(zl≠zs)</th><th>P(zl&lt;zs)</th><th>effective NB</th></tr></thead>
<tbody>{overview}</tbody></table>{sections}</body></html>"""
    OUTPUT.write_text(document)
    print(OUTPUT)


if __name__ == "__main__":
    main()
