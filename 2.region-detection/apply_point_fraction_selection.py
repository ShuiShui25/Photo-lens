#!/usr/bin/env python3
"""Export final apertures without detecting or selecting the source again.

The selection worker owns the r_s rule, contamination carving, hard gates and
ranking. This adapter writes its exact masks to the paths used by photometry.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import source_selection as selection_core

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "regions"
THRESHOLD = selection_core.RS_THRESHOLD
sys.path.insert(0, str(ROOT))
from photo_lens.workflow import available_systems, model_products as products_for, REGION_METHOD


def load_module():
    spec = importlib.util.spec_from_file_location("photo_lens_region_v2", HERE / "region_selection_v2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def analyse(module, system: str, stable_dir: Path | None = None) -> dict:
    """Export the single selection made by region_selection_v2.

    No source is re-selected here any more.  The r_s rule, carving and gates
    run once inside ``region_selection_v2.py``; this step copies the exact
    masks into the file names photometry reads, draws the diagnostic figure and
    writes the per-system summary.
    """
    system_root = ROOT / "1.lens-modeling" / system
    config = json.loads((system_root / "config.json").read_text())
    products = products_for(system)
    stable = (
        stable_dir.resolve()
        if stable_dir is not None
        else HERE / "outputs" / system
    )
    summary = json.loads((stable / f"{system}_v2_summary.json").read_text())
    adaptation = json.loads((stable / f"{system}_v2_adaptation.json").read_text())
    base_record = {
        "system": system,
        "survey": config["bb_survey"],
        "band": config["bb_band"],
        "formal_model_stage": 4 if products.name == "wcs_products_stage4" else 3,
        "version": REGION_METHOD,
        "products": str(products),
    }
    if summary.get("status") == "failed":
        raise RuntimeError(f"Source selection failed for {system}: {summary.get('error')}")
    centre = json.loads((products / f"{system}_lens_light_centre_wcs.json").read_text())
    source = module.load_product(
        products / f"{system}_autolens_lens_light_subtracted_wcs.fits", centre,
        module.ANALYSIS_HALF_SIZE_PIX,
    )
    lens_product = module.load_product(
        products / f"{system}_lens_light_model_wcs.fits", centre, module.ANALYSIS_HALF_SIZE_PIX,
    )
    masks = np.load(stable / f"{system}_selection_masks.npz")
    if int(masks["x0"]) != source.x0 or int(masks["y0"]) != source.y0:
        raise RuntimeError(f"System {system}: mask crop does not match the product crop")
    source_final = masks["source"].astype(bool)
    lens_initial = masks["lens_initial"].astype(bool)
    lens_final = masks["lens_final"].astype(bool)
    background_final = masks["background"].astype(bool)
    satellite = masks["satellite"].astype(bool)
    secondary = masks["secondary"].astype(bool)
    accepted = masks["accepted"].astype(bool)
    labels = masks["components"]
    components = [labels == index for index in range(1, int(labels.max()) + 1)]
    selected = summary["selected"]["sextractor"]
    members = selected["selected_members_1based"]
    bg = summary["background_region"]
    bg_centre = (float(bg["centre_crop_x"]), float(bg["centre_crop_y"]))
    if np.any(source_final & lens_final):
        raise RuntimeError(f"System {system}: final lens/source overlap")

    out = OUTPUT / system
    out.mkdir(parents=True, exist_ok=True)
    module.write_ds9_regions(
        out / f"{system}_all_sources_point_fraction.reg", components, source,
        [f"source_component_{index + 1}" for index in range(len(components))],
        ["green" if index + 1 in members else "blue" for index in range(len(components))],
    )
    if secondary.any():
        module.write_ds9_regions(
            out / f"{system}_secondary_sources_point_fraction.reg", [secondary], source,
            ["secondary_sources"], ["blue"],
        )
    selection_core.write_exact_cell_region(
        out / f"{system}_source_point_fraction040.reg", source_final, source,
        "source_point_fraction040", "green",
    )
    module.write_ds9_regions(
        out / f"{system}_lens_point_fraction040.reg", [lens_final], lens_product,
        ["lens_point_fraction040"], ["magenta"],
    )
    selection_core.write_exact_reflected_region(
        out / f"{system}_background_point_fraction040.reg", source_final, source,
        (bg_centre[0] + source.x0, bg_centre[1] + source.y0),
        "background_point_fraction040", "cyan",
    )
    if satellite.any():
        module.write_ds9_regions(
            out / f"{system}_satellites_point_fraction.reg", [satellite], source,
            ["satellite_exclusion"], ["red"],
        )
    winner = next(c for c in adaptation["source_selection"]["candidates"] if c["fate"] == "selected")
    ratio = masks["rs_ratio"]
    overlap_values = ratio[np.isfinite(ratio) & lens_initial & source.valid]
    record = {
        **base_record,
        "status": "complete",
        "source_detection_product": summary["source_detection_product"],
        "component_count": len(components),
        "primary_component_index_1based": members[0],
        "merged_source_components_1based": members if len(members) > 1 else [],
        "selected_candidate": winner["candidate"],
        "primary_initial_area_pix": winner["grown_area_pix"],
        "secondary_union_area_pix": int(secondary.sum()),
        "lens_initial_area_pix": int(lens_initial.sum()),
        "overlap_area_pix": winner["lens_overlap_pix"],
        "reliable_overlap_pix": winner["rs_reliable_overlap_pix"],
        "accepted_overlap_pix": winner["rs_accepted_overlap_pix"],
        "fraction_threshold": THRESHOLD,
        "fraction_min_median_max": (
            [float(np.min(overlap_values)), float(np.median(overlap_values)),
             float(np.max(overlap_values))] if overlap_values.size else None
        ),
        "satellite_area_pix": int(satellite.sum()),
        "satellite_selected_count": int(
            summary["satellite_exclusion"].get("selected_satellite_count", 0)
        ),
        "area_before_carving_pix": winner["area_before_carving_pix"],
        "carved_direct_pix": winner["carved_direct_pix"],
        "carved_mirror_pix": winner["carved_mirror_on_others_or_satellite_pix"],
        "carved_self_overlap_pix": winner["carved_self_overlap_pix"],
        "snr_retention": winner["snr_retention"],
        "hole_pixel_count": int(selected["hole_pixel_count"]),
        "final_source_area_pix": int(source_final.sum()),
        "final_background_area_pix_in_crop": int(background_final.sum()),
        "final_lens_area_pix": int(lens_final.sum()),
        "final_source_lens_overlap_pix": int(np.sum(source_final & lens_final)),
        "integrated_snr": float(selected["integrated_snr"]),
        "effective_snr": float(selected["effective_snr"]),
        "lens_model_flux_fraction": float(selected["lens_model_flux_fraction"]),
        "mirror_asymmetry": float(selected["mirror_asymmetry"]),
        "lens_contact": summary["lens_contact"],
        "warnings": summary.get("warnings", []),
        "figure": f"regions/{system}/{system}_regions.png",
    }
    (out / f"{system}_point_fraction040_summary.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=float) + "\n"
    )
    return record


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", nargs="+", required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Root for named per-system directories")
    args = parser.parse_args()
    names = {item["id"] for item in available_systems()}
    if any(name not in names for name in args.system):
        parser.error("Unknown system")
    if args.input_dir and len(args.system) != 1:
        parser.error("--input-dir requires exactly one system")
    if args.output_dir:
        OUTPUT = args.output_dir.resolve()
    module = load_module()
    for name in args.system:
        record = analyse(module, name, args.input_dir)
        print(f"{name}: exported source={record['final_source_area_pix']}, "
              f"lens={record['final_lens_area_pix']}, background={record['final_background_area_pix_in_crop']} pixels", flush=True)


if __name__ == "__main__":
    main()
