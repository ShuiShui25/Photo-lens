"""Selection plots and structured diagnostics for the local dashboard."""
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


def system_figure(system: str, asset: Path, region_dir: Path) -> None:
    manual_root = None
    summary = json.loads((region_dir / f"{system}_v2_summary.json").read_text())
    adaptation = json.loads((region_dir / f"{system}_v2_adaptation.json").read_text())
    failed = summary.get("status") == "failed"
    products = (ROOT / summary["wcs_products_dir"]).resolve()
    model_path = products / f"{system}_lens_light_model_wcs.fits"
    reduced_path = products / f"{system}_autolens_lens_light_subtracted_wcs.fits"
    for exact_path in (model_path, reduced_path):
        header = fits.getheader(exact_path)
        if not bool(header.get("FITEXACT", False)) or header.get("MAPSCHEM") != "FITNATIVE1":
            raise RuntimeError(f"Report input is not an exact FitImaging export: {exact_path}")
    model, model_wcs = read_image(model_path)
    reduced, reduced_wcs = read_image(reduced_path)
    centre = json.loads((products / f"{system}_lens_light_centre_wcs.json").read_text())
    anchor_meta = adaptation["lens_selection"]["source_anchor"]
    components = anchor_meta["components"]
    labels_path = region_dir / anchor_meta["components_label_image"]
    labels = fits.getdata(labels_path) if labels_path.is_file() else None
    auto = {
        "lens": region_dir / f"{system}_lens_v2.reg",
        "lens_initial": region_dir / f"{system}_lens_initial_v2.reg",
        "source": region_dir / f"{system}_source_v2.reg",
        "background": region_dir / f"{system}_background_v2.reg",
        "satellites": region_dir / f"{system}_satellites_v2.reg",
    }
    masks_path = region_dir / f"{system}_selection_masks.npz"
    masks = np.load(masks_path) if masks_path.is_file() else None
    manual_source, manual_lens = manual_paths(manual_root, system)
    xmin, xmax, ymin, ymax, cx, cy = crop(model.shape, centre)
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 11.5), constrained_layout=True)
    axes = axes_grid.ravel()
    # Panel 1: 4-sigma components with anchor decisions.
    ax = axes[0]
    ax.imshow(reduced, origin="lower", cmap="gray", norm=image_norm(reduced))
    if labels is not None:
        # labels live on the analysis crop; place them via the bbox records.
        x0 = components[0]["peak_full_x"] - components[0]["peak_crop_x"] if components else 0
        y0 = components[0]["peak_full_y"] - components[0]["peak_crop_y"] if components else 0
        extent = (x0 - 0.5, x0 + labels.shape[1] - 0.5, y0 - 0.5, y0 + labels.shape[0] - 0.5)
        colours = plt.cm.tab10(np.linspace(0, 1, max(len(components), 1)))
        for record in components:
            index = record["component_index_1based"]
            ax.contour(labels == index, [0.5], colors=[colours[index - 1]], linewidths=1.6,
                       origin="lower", extent=extent)
            marker = {"anchor": "*", "skipped": "x", "rejected": "x", "merged": "o"}[record["fate"]]
            colour = {"anchor": "red", "skipped": "orange", "rejected": "yellow", "merged": "white"}[record["fate"]]
            ax.plot(record["peak_full_x"], record["peak_full_y"], marker, color=colour, ms=12, mew=2)
            ax.text(record["peak_full_x"] + 1.2, record["peak_full_y"] + 1.2, str(index),
                    color="white", fontsize=10, weight="bold")
    core = anchor_meta.get("core_radius_pix")
    if core:
        ax.add_patch(plt.Circle((cx, cy), core, fill=False, color="orange", ls=":", lw=1.4))
    ax.set_title(f"{anchor_meta['detect_sigma']}σ components: star = anchor, cross = removed")
    panels = (
        ("Lens: initial (dashed) / final after r_s", model, model_wcs, "lens", "magenta"),
        ("Source (holes in red)", reduced, reduced_wcs, "source", "lime"),
        ("Point-reflected background", reduced, reduced_wcs, "background", "cyan"),
    )
    for ax, (title, data, wcs, key, color) in zip(axes[1:], panels, strict=True):
        ax.imshow(data, origin="lower", cmap="gray", norm=image_norm(data))
        if key == "lens":
            overlay(ax, auto["lens_initial"], wcs, edgecolor="magenta", linestyle="--", linewidth=1.5)
        if not failed:
            overlay(ax, auto[key], wcs, edgecolor=color, linewidth=2.5)
        overlay(ax, auto["satellites"], wcs, edgecolor="red", linestyle="-.", linewidth=1.8)
        overlay(ax, manual_source, wcs, edgecolor="yellow", linestyle="--", linewidth=1.7)
        overlay(ax, manual_lens, wcs, edgecolor="orange", linestyle=":", linewidth=1.9)
        if key == "source" and masks is not None and masks["holes"].any():
            hy, hx = np.nonzero(masks["holes"])
            ax.plot(hx + int(masks["x0"]), hy + int(masks["y0"]), "s", mfc="none",
                    color="red", ms=6, mew=1.5)
        ax.set_title(title + (" — FAILED" if failed and key != "lens" else ""))
    for ax in axes:
        ax.plot(cx, cy, "+", color="deepskyblue", ms=12, mew=2)
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="Detector x pixel (0-based)", ylabel="Detector y pixel (0-based)")
        ax.set_aspect("equal")
    handles = [
        Line2D([0], [0], color="magenta", lw=2.5, label="lens (final)"),
        Line2D([0], [0], color="magenta", ls="--", lw=1.5, label="lens (initial)"),
        Line2D([0], [0], color="lime", lw=2.5, label="source"),
        Line2D([0], [0], color="cyan", lw=2.5, label="background"),
        Line2D([0], [0], color="red", ls="-.", lw=1.8, label="modeled satellites"),
        Line2D([0], [0], color="red", marker="*", ls="none", label="anchor"),
        Line2D([0], [0], color="orange", ls=":", label="core radius (0.5 FWHM)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    stage = 4 if "stage4" in products.name else 3
    fig.suptitle(f"{system}: selection diagnostics (model stage {stage})")
    asset.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(asset, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if masks is not None:
        masks.close()


def _finite_json(value):
    """Strict JSON for fetch().json(): diagnostics may contain infinity or NaN."""
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_diagnostics(name: str, analysis_dir: Path, result_dir: Path) -> None:
    summary = json.loads((analysis_dir / f"{name}_v2_summary.json").read_text())
    adaptation = json.loads((analysis_dir / f"{name}_v2_adaptation.json").read_text())
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_name = f"{name}_selection_diagnostics.png"
    system_figure(name, result_dir / figure_name, analysis_dir)
    details = {
        "system": name,
        "version": summary["version"],
        "status": summary.get("status", "complete"),
        "figure": f"2.region-detection/regions/{name}/{figure_name}",
        "centre_definition": "All apertures share the fitted lens-light centre (detector pixels, zero based).",
        "source_image": "FitImaging lens-plane-subtracted image (lens and modeled satellites removed)",
        "model_stage": 4 if Path(summary["wcs_products_dir"]).name == "wcs_products_stage4" else 3,
        "lens_method": summary.get("lens_method"),
        "lens_scale": summary.get("lens_scale"),
        "lens_initial_area_pix": summary.get("lens_initial_area_pix"),
        "lens_area_pix": summary.get("lens_area_pix"),
        "anchors": adaptation["lens_selection"]["source_anchor"],
        "lens_contact": adaptation["lens_selection"]["contact"],
        "source_selection": adaptation.get("source_selection", {}),
        "selected": summary.get("selected", {}).get("sextractor", {}),
        "satellites": summary.get("satellite_exclusion", {}),
        "background": summary.get("background_region", {}),
        "readback": summary.get("readback", {}),
        "warnings": summary.get("warnings", []),
        "error": summary.get("error"),
    }
    (result_dir / f"{name}_region_diagnostics.json").write_text(
        json.dumps(_finite_json(details), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
