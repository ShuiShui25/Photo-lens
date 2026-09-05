#!/usr/bin/env python3
"""Apply the production point-reflection source-fraction overlap rule.

The stable-v2 lens mask and an independently detected source mask are built
before any lens/source cross-subtraction.  Only their overlap is classified by

    r_s = (F_data - F_reflected) /
          (F_lens_model + F_data - F_reflected).

Reliable pixels with r_s >= 0.4 belong to the main source.  Accepted pixels and
their model-centre reflections are removed from the lens aperture.  Modeled
satellites and independently detected secondary sources remain blockers for
both the source aperture and its reflected background aperture.
"""
from __future__ import annotations

import argparse
import os
import importlib.util
import json
import math
import re
import shutil
import tempfile
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    PercentileInterval,
    SqrtStretch,
)
from astropy.wcs import WCS
from matplotlib.lines import Line2D
from regions import Regions
from scipy.ndimage import distance_transform_edt, map_coordinates
from skimage.draw import line, polygon


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "regions"
SEXTRACTOR = Path(os.environ.get("PHOTO_LENS_SEXTRACTOR") or shutil.which("source-extractor") or "source-extractor")
MANUAL_ROOT = ROOT / "data/manual-regions"
sys.path.insert(0, str(ROOT))
from photo_lens.workflow import available_systems
SYSTEMS = tuple(item['id'] for item in available_systems())
THRESHOLD = 0.4
POINT_CONFIG = json.loads((HERE / "point_fraction_config.json").read_text())


def load_module():
    path = HERE / "region_selection_v2.py"
    spec = importlib.util.spec_from_file_location("photo_lens_region_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def products_for(system: str) -> Path:
    suffix = "wcs_products_stage4"
    return ROOT / "1.lens-modeling" / f"{system}" / "output" / suffix


def mask_from_ds9(path: Path, product) -> np.ndarray:
    result = np.zeros(product.data.shape, dtype=bool)
    if not path.is_file():
        return result
    for body in re.findall(r"polygon\((.*?)\)", path.read_text(), flags=re.I | re.S):
        values = np.asarray([float(value) for value in body.split(",")]).reshape(-1, 2)
        px, py = product.wcs.world_to_pixel_values(values[:, 0], values[:, 1])
        rr, cc = polygon(py - product.y0, px - product.x0, shape=result.shape)
        result[rr, cc] = True
    return result


def initial_sources(module, system: str, source, lens_product, config: dict):
    base = json.loads(
        (ROOT / "1.lens-modeling/region_selection/config_adaptive_v1_base.json").read_text()
    )
    pixel_scale = float(
        np.sqrt(np.abs(np.linalg.det(source.wcs.pixel_scale_matrix))) * 3600.0
    )
    radius_pix = float(config["einstein_radius_mean"]) / pixel_scale
    empty = np.zeros(source.data.shape, dtype=bool)
    override = POINT_CONFIG.get("source_sextractor_overrides", {}).get(system)
    with tempfile.TemporaryDirectory(prefix=f"pl03_fraction_{system}_") as tmp:
        work = Path(tmp)
        if override:
            old_grow_ratio = module.GROW_SIGMA_RATIO
            try:
                module.GROW_SIGMA_RATIO = float(override["grow_sigma_ratio"])
                selected = module.build_trial(
                    source,
                    lens_product,
                    base["sextractor"]["source"],
                    float(override["fixed_detect_sigma"]),
                    SEXTRACTOR,
                    work / "source_uncut_fixed",
                    empty,
                    radius_pix,
                    None,
                )
            finally:
                module.GROW_SIGMA_RATIO = old_grow_ratio
            if selected["mask"] is None or not selected["mask"].any():
                raise RuntimeError(
                    f"System {system}: configured source SExtractor override failed: "
                    f"{selected.get('error', 'empty mask')}"
                )
            selected["source_sextractor_override"] = dict(override)
        else:
            selected, _ = module.adapt_region(
                source, lens_product, base["sextractor"]["source"], SEXTRACTOR,
                work / "source_uncut", empty, radius_pix, None,
            )
            selected["source_sextractor_override"] = None
        components, _, _ = module.detect_components(
            source, selected["settings"], SEXTRACTOR,
            work / "all_components", empty,
        )
    primary = np.asarray(selected["mask"], dtype=bool)
    if not components:
        components = [primary.copy()]
    primary_index = int(np.argmax([np.sum(mask & primary) for mask in components]))
    requested = [
        int(index) - 1
        for index in POINT_CONFIG.get(
            "source_component_merge_groups_1based", {}
        ).get(system, [])
    ]
    if requested:
        if any(index < 0 or index >= len(components) for index in requested):
            raise ValueError(
                f"System {system}: invalid source-component merge request "
                f"{[index + 1 for index in requested]} for {len(components)} components"
            )
        if primary_index not in requested:
            raise ValueError(
                f"System {system}: merge group must include selected primary "
                f"component {primary_index + 1}"
            )
        # Add every requested SExtractor component and the shortest possible
        # pixel line to the current primary mask.  The bridge matters when two
        # components touch only diagonally: the later 4-connected cleanup must
        # not discard one half of the explicitly merged source.
        for index in requested:
            component = np.asarray(components[index], dtype=bool)
            if not primary.any():
                primary |= component
                continue
            distance, nearest = distance_transform_edt(~primary, return_indices=True)
            yy, xx = np.nonzero(component)
            nearest_index = int(np.argmin(distance[yy, xx]))
            target_y, target_x = int(yy[nearest_index]), int(xx[nearest_index])
            start_y = int(nearest[0, target_y, target_x])
            start_x = int(nearest[1, target_y, target_x])
            bridge_y, bridge_x = line(start_y, start_x, target_y, target_x)
            primary[bridge_y, bridge_x] = True
            primary |= component
        selected["merged_source_components_1based"] = [
            index + 1 for index in requested
        ]
    else:
        selected["merged_source_components_1based"] = []
    secondary = np.zeros_like(primary)
    for index, component in enumerate(components):
        if index != primary_index and index not in requested:
            secondary |= component
    return primary, components, primary_index, secondary, selected


def science_data(config: dict, system_root: Path):
    path = Path(config["science_cutout"])
    if not path.is_absolute():
        path = (system_root / path).resolve()
    with fits.open(path) as hdul:
        hdu = hdul[int(config.get("science_hdu", 0))]
        return np.asarray(hdu.data, dtype=float), hdu.header.copy()


def full_mask(mask: np.ndarray, product, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    result[product.y0:product.y0 + mask.shape[0],
           product.x0:product.x0 + mask.shape[1]] = mask
    return result


def overlay_manual(ax, path: Path, wcs: WCS, **style) -> None:
    if not path.is_file():
        return
    try:
        for region in Regions.read(path, format="ds9"):
            region.to_pixel(wcs).plot(ax=ax, facecolor="none", **style)
    except Exception as exc:
        print(f"WARNING: cannot overlay {path}: {exc}")


def contour(ax, mask: np.ndarray, color: str, **kwargs) -> None:
    if mask.any():
        ax.contour(mask, [0.5], colors=[color], origin="lower", **kwargs)


def plot_system(system: str, raw: np.ndarray, header: fits.Header, source,
                reduced: np.ndarray, components: list[np.ndarray], primary_index: int,
                primary_initial: np.ndarray, secondary: np.ndarray,
                lens_initial: np.ndarray, source_final: np.ndarray,
                lens_final: np.ndarray, background_final: np.ndarray,
                satellite: np.ndarray, accepted: np.ndarray, output: Path) -> None:
    wcs = WCS(header).celestial
    shape = raw.shape
    full = lambda mask: full_mask(mask, source, shape)
    reduced_full = np.full(shape, np.nan)
    reduced_full[source.y0:source.y0 + reduced.shape[0],
                 source.x0:source.x0 + reduced.shape[1]] = reduced
    finite_raw = raw[np.isfinite(raw)]
    finite_reduced = reduced[np.isfinite(reduced)]
    raw_norm = ImageNormalize(
        finite_raw, interval=PercentileInterval(99.5), stretch=SqrtStretch(), clip=True
    )
    reduced_norm = ImageNormalize(
        vmin=0.0,
        vmax=float(np.nanpercentile(finite_reduced, 99.5)),
        stretch=LinearStretch(),
        clip=True,
    )
    cx = source.centre_x
    cy = source.centre_y
    half = min(42, int(min(shape) // 2 - 1))
    limits = (cx - half, cx + half, cy - half, cy + half)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 12.8), sharex=True, sharey=True)
    axes = axes.ravel()

    axes[0].imshow(reduced_full, origin="lower", cmap="gray", norm=reduced_norm)
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(components), 1)))
    for index, component in enumerate(components):
        component_full = full(component)
        contour(axes[0], component_full, colours[index], linewidths=1.8)
        yy, xx = np.nonzero(component_full)
        if len(xx):
            axes[0].text(float(np.mean(xx)), float(np.mean(yy)), str(index + 1),
                         color="white", fontsize=10, weight="bold",
                         ha="center", va="center",
                         bbox={"facecolor": colours[index], "alpha": 0.75, "pad": 1.5})
    axes[0].set_title("AutoLens lens-plane subtracted: raw SExtractor components")

    axes[1].imshow(reduced_full, origin="lower", cmap="gray", norm=reduced_norm)
    contour(axes[1], full(primary_initial), "lime", linewidths=2.5)
    contour(axes[1], full(secondary), "deepskyblue", linewidths=1.5, linestyles="--")
    overlay_manual(axes[1], MANUAL_ROOT / "source_regions" / f"{system}.reg", wcs,
                   edgecolor="yellow", linewidth=1.7, linestyle=":")
    axes[1].set_title(f"Primary source = component {primary_index + 1}; secondary sources dashed")

    axes[2].imshow(raw, origin="lower", cmap="gray", norm=raw_norm)
    contour(axes[2], full(lens_final), "magenta", linewidths=2.3)
    contour(axes[2], full(source_final), "lime", linewidths=2.3)
    ay, axx = np.nonzero(full(accepted))
    if len(axx):
        axes[2].scatter(axx, ay, s=26, marker="s", facecolors="none",
                        edgecolors="yellow", linewidths=1.0)
    overlay_manual(axes[2], MANUAL_ROOT / "source_regions" / f"{system}.reg", wcs,
                   edgecolor="yellow", linewidth=1.4, linestyle=":")
    overlay_manual(axes[2], MANUAL_ROOT / "lens_region" / f"{system}.reg", wcs,
                   edgecolor="orange", linewidth=1.4, linestyle="--")
    axes[2].set_title("Raw data: final lens/source; yellow squares = overlap pixels kept as source")

    axes[3].imshow(raw, origin="lower", cmap="gray", norm=raw_norm)
    contour(axes[3], full(source_final), "lime", linewidths=2.3)
    contour(axes[3], full(background_final), "cyan", linewidths=2.2, linestyles="--")
    contour(axes[3], full(satellite), "red", linewidths=1.8, linestyles="-.")
    contour(axes[3], full(secondary), "deepskyblue", linewidths=1.6, linestyles=":")
    overlay_manual(axes[3], MANUAL_ROOT / "source_regions" / f"{system}.reg", wcs,
                   edgecolor="yellow", linewidth=1.4, linestyle=":")
    axes[3].set_title("Raw data: primary + reflected background + satellite + secondary sources")

    for ax in axes:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal")
        ax.set_xlabel("FITS x pixel")
        ax.set_ylabel("FITS y pixel")
    legend = [
        Line2D([0], [0], color="lime", lw=2, label="primary/final source"),
        Line2D([0], [0], color="magenta", lw=2, label="final lens"),
        Line2D([0], [0], color="cyan", lw=2, ls="--", label="reflected background"),
        Line2D([0], [0], color="red", lw=2, ls="-.", label="satellite"),
        Line2D([0], [0], color="deepskyblue", lw=2, ls=":", label="secondary source"),
        Line2D([0], [0], color="yellow", lw=2, ls=":", label="manual source reference"),
        Line2D([0], [0], color="orange", lw=2, ls="--", label="manual lens reference"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(
        f"System {system}: stable-v2 + point-reflection source fraction (r_s >= {THRESHOLD})",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    fig.savefig(output, dpi=175, bbox_inches="tight")
    plt.close(fig)


def analyse(module, system: str) -> dict:
    system_root = ROOT / "1.lens-modeling" / f"{system}"
    config = json.loads((system_root / "config.json").read_text())
    run_summary = json.loads((system_root / "output" / "run_summary.json").read_text())
    constraints = run_summary["constraints"]
    mask_centre = constraints["mask_centre_yx_arcsec"]
    mask_radius = float(constraints["mask_radius_arcsec"])
    satellite_positions = constraints["satellite_positions_yx_arcsec"]
    satellite_centre_distances = [
        math.hypot(position[0] - mask_centre[0], position[1] - mask_centre[1])
        for position in satellite_positions
    ]
    satellite_centres_inside = [distance <= mask_radius for distance in satellite_centre_distances]
    products = products_for(system)
    exact_qa = json.loads(
        (products / f"{system}_exact_fit_plane_qa.json").read_text()
    )
    exact_checks = exact_qa["serialized_checks"]
    exact_array_equal = bool(
        exact_qa["science_crop_array_equal"]
        and all(item["array_equal"] for item in exact_checks.values())
    )
    if not exact_array_equal:
        raise RuntimeError(f"System {system}: exact FitImaging export QA failed")
    stable = HERE / "outputs" / f"{system}"
    summary = json.loads((stable / f"{system}_v2_summary.json").read_text())
    satellite_summary = summary["satellite_exclusion"]
    centre = json.loads((products / f"{system}_lens_light_centre_wcs.json").read_text())
    autolens_source_path = (
        products / f"{system}_autolens_lens_light_subtracted_wcs.fits"
    )
    exact_lens_path = products / f"{system}_lens_light_model_wcs.fits"
    for exact_path in (autolens_source_path, exact_lens_path):
        header = fits.getheader(exact_path)
        if not bool(header.get("FITEXACT", False)) or header.get("MAPSCHEM") != "FITNATIVE1":
            raise RuntimeError(
                "Point-fraction selection requires a strict FitImaging native "
                f"export, but found a stale/re-rendered product: {exact_path}"
            )
    source = module.load_product(
        autolens_source_path, centre, module.ANALYSIS_HALF_SIZE_PIX,
    )
    lens_product = module.load_product(
        exact_lens_path, centre,
        module.ANALYSIS_HALF_SIZE_PIX,
    )
    primary, components, primary_index, secondary, selected = initial_sources(
        module, system, source, lens_product, config
    )
    lens = mask_from_ds9(stable / f"{system}_lens_v2.reg", lens_product)
    satellite = mask_from_ds9(
        stable / f"{system}_satellite_union_v2.reg", source
    )
    overlap = primary & lens & source.valid & lens_product.valid

    raw, raw_header = science_data(config, system_root)
    raw_crop = raw[source.y0:source.y0 + source.data.shape[0],
                   source.x0:source.x0 + source.data.shape[1]]
    if raw_crop.shape != source.data.shape:
        raise ValueError(f"System {system}: science/product grid mismatch")
    bg = summary["background_region"]
    bg_centre = (float(bg["centre_crop_x"]), float(bg["centre_crop_y"]))
    yy, xx = np.indices(raw_crop.shape, dtype=float)
    f_ref = map_coordinates(
        raw_crop, [2 * bg_centre[1] - yy, 2 * bg_centre[0] - xx],
        order=1, mode="constant", cval=np.nan, prefilter=False,
    )
    source_difference = raw_crop - f_ref
    denominator = lens_product.data + source_difference
    ratio = np.full(raw_crop.shape, np.nan)
    valid_ratio = np.isfinite(source_difference) & np.isfinite(denominator) & (denominator != 0)
    ratio[valid_ratio] = source_difference[valid_ratio] / denominator[valid_ratio]
    denominator_floor = 3.0 * np.sqrt(2.0) * float(source.sigma)
    reliable = (
        overlap & valid_ratio & (source_difference >= 0) &
        (denominator > denominator_floor) & (ratio >= 0) & (ratio <= 1)
    )
    accepted = overlap & reliable & (ratio >= THRESHOLD)
    source_final = primary & (~overlap | accepted)

    # Preserve the stable source-isolation rule and apply it to the reflected
    # control aperture as well.  Satellite and secondary-source masks are never
    # allowed to contaminate either member of the differential pair.
    blocker = satellite | secondary
    before_blocker = int(source_final.sum())
    source_final &= ~blocker
    direct_blocker_removed = before_blocker - int(source_final.sum())
    source_final, paired_removed = module.remove_pixels_reflecting_into_blocker(
        source_final, blocker, bg_centre
    )
    before_connectivity = int(source_final.sum())
    source_final = module.largest_connected_piece(source_final)
    connectivity_removed = before_connectivity - int(source_final.sum())
    if not source_final.any():
        raise RuntimeError(f"System {system}: exclusions removed the main source")

    symmetric_cut = (
        module.symmetric_closure(accepted, source.centre_crop_xy, source.valid)
        if accepted.any() else accepted.copy()
    )
    lens_final = lens & ~symmetric_cut
    for _ in range(8):
        reflected = module._reflect_mask(lens_final, source.centre_crop_xy) & source.valid
        updated = lens_final & reflected
        if np.array_equal(updated, lens_final):
            break
        lens_final = updated
    source_final &= ~lens_final
    background_final = module._reflect_mask(source_final, bg_centre) & source.valid
    if np.any(background_final & blocker):
        raise RuntimeError(f"System {system}: final background intersects blocker")
    if np.any(source_final & lens_final):
        raise RuntimeError(f"System {system}: final lens/source overlap")

    out = OUTPUT / f"{system}"
    out.mkdir(parents=True, exist_ok=True)
    module.write_ds9_regions(
        out / f"{system}_all_sources_point_fraction.reg", components, source,
        [f"source_component_{index + 1}" for index in range(len(components))],
        ["green" if index == primary_index else "blue" for index in range(len(components))],
    )
    module.write_ds9_regions(
        out / f"{system}_secondary_sources_point_fraction.reg", [secondary], source,
        ["secondary_sources"], ["blue"],
    ) if secondary.any() else None
    module.write_ds9_regions(
        out / f"{system}_source_point_fraction040.reg", [source_final], source,
        ["source_point_fraction040"], ["green"],
    )
    module.write_ds9_regions(
        out / f"{system}_lens_point_fraction040.reg", [lens_final], lens_product,
        ["lens_point_fraction040"], ["magenta"],
    )
    module.write_point_reflected_ds9_region(
        out / f"{system}_background_point_fraction040.reg", source_final, source,
        "background_point_fraction040", "cyan",
        centre_full_xy=(bg_centre[0] + source.x0, bg_centre[1] + source.y0),
    )
    if satellite.any():
        module.write_ds9_regions(
            out / f"{system}_satellites_point_fraction.reg", [satellite], source,
            ["satellite_exclusion"], ["red"],
        )

    plot_system(
        system, raw, raw_header, source, source.data, components, primary_index,
        primary, secondary, lens, source_final, lens_final, background_final,
        satellite, accepted, out / f"{system}_point_fraction_regions.png",
    )
    values = ratio[overlap & np.isfinite(ratio)]
    record = {
        "system": system,
        "survey": config["bb_survey"],
        "band": config["bb_band"],
        "formal_model_stage": 4,
        "products": str(products),
        "source_detection_product": str(autolens_source_path),
        "source_detection_definition": (
            "literal FitImaging subtracted_images_of_planes_list[-1], with "
            "strict native-array-to-detector equality validation"
        ),
        "exact_fit_array_equal": exact_array_equal,
        "exact_fit_validated_pixels": int(
            exact_checks["lens_light_subtracted"]["validated_pixels"]
        ),
        "exact_fit_max_abs_difference": float(
            max(item["max_abs_difference"] for item in exact_checks.values())
        ),
        "component_count": len(components),
        "primary_component_index_1based": primary_index + 1,
        "merged_source_components_1based": selected[
            "merged_source_components_1based"
        ],
        "primary_initial_area_pix": int(primary.sum()),
        "source_sextractor_override": selected.get("source_sextractor_override"),
        "secondary_union_area_pix": int(secondary.sum()),
        "lens_initial_area_pix": int(lens.sum()),
        "overlap_area_pix": int(overlap.sum()),
        "reliable_overlap_pix": int(reliable.sum()),
        "accepted_overlap_pix": int(accepted.sum()),
        "rejected_unreliable_pix": int(np.sum(overlap & ~reliable)),
        "rejected_below_threshold_pix": int(np.sum(overlap & reliable & (ratio < THRESHOLD))),
        "fraction_threshold": THRESHOLD,
        "fraction_min_median_max": (
            [float(np.min(values)), float(np.median(values)), float(np.max(values))]
            if values.size else None
        ),
        "satellite_area_pix": int(satellite.sum()),
        "satellite_original_1sigma_area_pix": int(
            satellite_summary.get("current_1sigma_union_area_pix", satellite.sum())
        ),
        "satellite_area_reduction_fraction": float(
            satellite_summary.get("compact_union_area_reduction_fraction", 0.0)
        ),
        "satellite_position_count": len(satellite_positions),
        "satellite_centres_inside_mask": int(sum(satellite_centres_inside)),
        "satellite_centre_distances_arcsec": satellite_centre_distances,
        "model_mask_radius_arcsec": mask_radius,
        "satellite_displayed": bool(satellite.any()),
        "satellite_selected_count": int(
            satellite_summary.get("selected_satellite_count", 0)
        ),
        "direct_blocker_removed_pix": int(direct_blocker_removed),
        "paired_blocker_removed_pix": int(paired_removed),
        "disconnected_remainder_removed_pix": int(connectivity_removed),
        "final_source_area_pix": int(source_final.sum()),
        "final_background_area_pix_in_crop": int(background_final.sum()),
        "final_lens_area_pix": int(lens_final.sum()),
        "final_source_lens_overlap_pix": int(np.sum(source_final & lens_final)),
        "final_background_blocker_overlap_pix": int(np.sum(background_final & blocker)),
        "manual_source_available": (MANUAL_ROOT / "source_regions" / f"{system}.reg").is_file(),
        "manual_lens_available": (MANUAL_ROOT / "lens_region" / f"{system}.reg").is_file(),
        # The HTML report lives one directory above OUTPUT, so the image URL
        # must include OUTPUT's directory name rather than being relative to
        # the manifest stored inside OUTPUT.
        "figure": (
            f"point-fraction-outputs/{system}/"
            f"{system}_point_fraction_regions.png"
        ),
    }
    (out / f"{system}_point_fraction040_summary.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    module = load_module()
    records = [analyse(module, system) for system in args.system]
    (OUTPUT / "point_fraction_manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
