#!/usr/bin/env python3
"""Build the production lens/source/background aperture HTML report."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from matplotlib.lines import Line2D
from regions import Regions


ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ("2", "8", "12", "3110", "3336", "3441", "13369", "12971", "13246")


def read_image(path: Path) -> tuple[np.ndarray, WCS]:
    data, header = fits.getdata(path, header=True)
    return np.asarray(data, dtype=float), WCS(header).celestial


def image_norm(data: np.ndarray) -> ImageNormalize:
    finite = data[np.isfinite(data)]
    return ImageNormalize(
        finite, interval=PercentileInterval(99.5), stretch=AsinhStretch(0.06)
    )


def overlay(ax, path: Path | None, wcs: WCS, **style) -> None:
    if path is None or not path.is_file():
        return
    for sky_region in Regions.read(path, format="ds9"):
        sky_region.to_pixel(wcs).plot(ax=ax, facecolor="none", **style)


def crop(shape: tuple[int, int], centre: dict, half_size: int = 32):
    x = float(centre["fits_x_pixel_zero_indexed"])
    y = float(centre["fits_y_pixel_zero_indexed"])
    return (
        max(x - half_size, -0.5), min(x + half_size, shape[1] - 0.5),
        max(y - half_size, -0.5), min(y + half_size, shape[0] - 0.5), x, y,
    )


def manual_paths(root: Path | None, system: str) -> tuple[Path | None, Path | None]:
    if root is None:
        return None, None
    source = root / "source_regions" / f"{system}.reg"
    lens = root / "lens_region" / f"{system}.reg"
    return (source if source.is_file() else None, lens if lens.is_file() else None)


def system_figure(system: str, asset: Path, manual_root: Path | None) -> dict:
    region_dir = ROOT / "region-detection/outputs" / f"system_{system}"
    summary = json.loads((region_dir / f"system{system}_v2_summary.json").read_text())
    products = Path(summary["wcs_products_dir"])
    model_path = products / f"{system}_lens_light_model_wcs.fits"
    reduced_path = products / f"{system}_autolens_lens_light_subtracted_wcs.fits"
    for exact_path in (model_path, reduced_path):
        header = fits.getheader(exact_path)
        if not bool(header.get("FITEXACT", False)) or header.get("MAPSCHEM") != "FITNATIVE1":
            raise RuntimeError(f"Report input is not an exact FitImaging export: {exact_path}")
    model, model_wcs = read_image(model_path)
    reduced, reduced_wcs = read_image(
        reduced_path
    )
    centre = json.loads((products / f"{system}_lens_light_centre_wcs.json").read_text())
    auto = {
        "lens": region_dir / f"system{system}_lens_v2.reg",
        "source": region_dir / f"system{system}_source_v2.reg",
        "background": region_dir / f"system{system}_background_v2.reg",
        "satellites": region_dir / f"system{system}_satellites_v2.reg",
    }
    manual_source, manual_lens = manual_paths(manual_root, system)
    xmin, xmax, ymin, ymax, cx, cy = crop(model.shape, centre)
    panels = (
        ("Automatic lens region", model, model_wcs, "lens", "magenta"),
        ("Source after modeled-satellite exclusion", reduced, reduced_wcs, "source", "lime"),
        ("Point-reflected background region", reduced, reduced_wcs, "background", "cyan"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), constrained_layout=True)
    for ax, (title, data, wcs, key, color) in zip(axes, panels, strict=True):
        ax.imshow(data, origin="lower", cmap="gray", norm=image_norm(data))
        overlay(ax, auto[key], wcs, edgecolor=color, linewidth=2.5)
        overlay(ax, auto["satellites"], wcs, edgecolor="red", linestyle="-.", linewidth=1.8)
        overlay(ax, manual_source, wcs, edgecolor="yellow", linestyle="--", linewidth=1.7)
        overlay(ax, manual_lens, wcs, edgecolor="orange", linestyle=":", linewidth=1.9)
        ax.plot(cx, cy, "+", color="deepskyblue", ms=12, mew=2)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), title=title,
               xlabel="FITS x pixel", ylabel="FITS y pixel")
        ax.set_aspect("equal")
    handles = [
        Line2D([0], [0], color="magenta", lw=2.5, label="automatic lens"),
        Line2D([0], [0], color="lime", lw=2.5, label="automatic source"),
        Line2D([0], [0], color="cyan", lw=2.5, label="automatic background"),
        Line2D([0], [0], color="red", ls="-.", lw=1.8, label="modeled satellites"),
        Line2D([0], [0], color="deepskyblue", marker="+", ls="none", label="modeled centre"),
    ]
    if manual_source is not None:
        handles.append(Line2D([0], [0], color="yellow", ls="--", label="manual source"))
    if manual_lens is not None:
        handles.append(Line2D([0], [0], color="orange", ls=":", label="manual lens"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    stage = 4 if "stage4" in products.name else 3
    fig.suptitle(f"System {system}: production automatic apertures (stage {stage})")
    asset.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(asset, dpi=170, bbox_inches="tight")
    plt.close(fig)
    selected = summary["selected"]["sextractor"]
    background = summary["background_region"]
    satellite = summary["satellite_exclusion"]
    return {
        "system": system,
        "stage": stage,
        "asset": asset,
        "source_area": int(selected["area_pix"]),
        "integrated_snr": float(selected["integrated_snr"]),
        "passes": bool(selected["passes_all_hard_gates"]),
        "background_area": int(background["background_area_pix_full_detector"]),
        "background_outside_mask": int(
            background["background_pixels_outside_model_mask"]
        ),
        "background_clipped": bool(
            background["final_ds9_region_clipped_by_model_mask"]
        ),
        "source_satellite_direct_removed": int(
            satellite["source_satellite_overlap_removed_pix"]
        ),
        "source_satellite_paired_removed": int(
            satellite["source_paired_for_background_satellite_removed_pix"]
        ),
        "background_satellite_overlap": int(
            background["background_satellite_overlap_pix"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--manual-region-root", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "region-detection/region_report.html",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    manual_root = args.manual_region_root.resolve() if args.manual_region_root else None
    assets = output.parent / "report-assets"
    records = [
        system_figure(system, assets / f"system_{system}.png", manual_root)
        for system in args.system
    ]
    cards = []
    for record in records:
        rel = record["asset"].relative_to(output.parent).as_posix()
        cards.append(
            f"<section><h2>System {record['system']} — stage {record['stage']}</h2>"
            f"<p>source area={record['source_area']} pix；integrated S/N="
            f"{record['integrated_snr']:.2f}；all hard gates="
            f"{'PASS' if record['passes'] else 'FAIL'}；background area="
            f"{record['background_area']} pix；background pixels outside modeling "
            f"mask={record['background_outside_mask']}；final DS9 clipped="
            f"{'yes' if record['background_clipped'] else 'no'}；source direct "
            f"satellite removal={record['source_satellite_direct_removed']} pix；"
            f"paired source removal for background satellite="
            f"{record['source_satellite_paired_removed']} pix；final background–"
            f"satellite overlap={record['background_satellite_overlap']} pix。</p>"
            f"<img src='{html.escape(rel)}' alt='System {record['system']} regions'></section>"
        )
    manual_note = (
        "已叠加可用的手动 source/lens 参考；它们不参与自动选区。"
        if manual_root else
        "本报告未叠加手动选区；可用 --manual-region-root 选择性添加参考。"
    )
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>PHOTO-LENS03 automatic aperture report</title><style>
body {{font-family:sans-serif;max-width:1700px;margin:2rem auto;line-height:1.5}}
img {{width:100%;border:1px solid #bbb}} section {{margin:2.5rem 0}}
</style></head><body><h1>PHOTO-LENS03 HSC/CFHT 自动选区报告</h1>
<p>lens 背景为 BB 建模 lens-light model WCS product；source 和 background 背景为 lens-light reduced WCS product。background 是在完整 detector 坐标中生成的连续点对称轮廓，允许越过建模 mask；WCS product 在 mask 外的空白只影响本报告底图，不会裁剪最终 region。红色点划线为独立 satellite-light 模型的自动显著区域，已从最终 source aperture 中扣除。{manual_note}</p>
{''.join(cards)}</body></html>"""
    output.write_text(document, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
