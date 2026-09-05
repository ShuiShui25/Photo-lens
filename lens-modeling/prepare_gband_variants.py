#!/usr/bin/env python3
"""Create non-destructive G-band variants of existing system configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_system_configs import image_hdu, positive_uncertainty_hdu


ROOT = Path(__file__).resolve().parent.parent
SUPPORTED = {"8": "HSC", "12": "CFHT", "3336": "HSC"}


def build(system: str) -> Path:
    survey = SUPPORTED[system]
    system_model = ROOT / "lens-modeling" / f"system_{system}"
    baseline_path = system_model / "config.json"
    config = json.loads(baseline_path.read_text(encoding="utf-8"))
    filename = f"{survey}-G.fits"
    data_root = ROOT / "data" / "bb" / system
    science = data_root / "science" / filename
    psf = data_root / "psf" / filename
    uncertainty_kind = "variance" if survey == "HSC" else "weight"
    uncertainty = data_root / uncertainty_kind / filename
    for path in (science, psf, uncertainty):
        if not path.is_file():
            raise FileNotFoundError(path)
    science_hdu, shape = image_hdu(science)
    uncertainty_hdu = positive_uncertainty_hdu(uncertainty, shape=shape)

    # Paths remain relative to system_<id>, exactly like the baseline config.
    relative_root = Path("../../data/bb") / system
    config.update(
        {
            "science_cutout": str(relative_root / "science" / filename),
            "psf": str(relative_root / "psf" / filename),
            "output_dir": "output_gband",
            "science_hdu": science_hdu,
            "bb_survey": survey,
            "bb_band": f"{survey}-G",
            "stage4_regularization": "constant_brightness_zeroth",
            "stage4_signal_scale": 0.5,
        }
    )
    config.pop("variance_cutout", None)
    config.pop("weight_cutout", None)
    config.pop("variance_hdu", None)
    config.pop("weight_hdu", None)
    config[f"{uncertainty_kind}_cutout"] = str(
        relative_root / uncertainty_kind / filename
    )
    config[f"{uncertainty_kind}_hdu"] = uncertainty_hdu
    target = system_model / "config_gband.json"
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"[{system}] {survey}-G -> {target}; output_gband")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("systems", nargs="*", default=list(SUPPORTED))
    args = parser.parse_args()
    for system in args.systems:
        if system not in SUPPORTED:
            raise ValueError(f"No G-band variant requested for system {system}")
        build(system)


if __name__ == "__main__":
    main()
