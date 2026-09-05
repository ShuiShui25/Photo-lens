#!/usr/bin/env python
"""Discover a complete BB backend and write reproducible four-stage configs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("2", "8", "12", "3110", "3336", "3441", "13369", "12971", "13246")

# These bands were validated by full four-stage fits and are intentionally
# decoupled from the band used to create positions_gui.txt.  Their G/R grids
# have identical shape and WCS centres, so the angular positions remain valid.
PREFERRED_MODEL_BANDS = {
    "8": ("HSC", "HSC-G"),
    "12": ("CFHT", "CFHT-G"),
    "3336": ("HSC", "HSC-G"),
}

# Redshifts and mass initialization come from the validated PHOTO-LENS02 branches.
META = {
    "2": dict(z_lens=0.1642, z_source=0.3239, ra=138.02212, dec=0.48364, er=1.63),
    "8": dict(z_lens=0.238, z_source=1.2, ra=36.296, dec=-4.90932, er=1.20),
    "12": dict(z_lens=0.456, z_source=1.59, ra=211.22692, dec=52.00686, er=2.55),
    "3110": dict(z_lens=0.646156, z_source=1.85, ra=34.40513, dec=-5.22475, er=1.27),
    "3336": dict(z_lens=0.499, z_source=1.22, ra=35.9423, dec=-5.5719, er=1.22),
    "3441": dict(z_lens=0.494, z_source=1.23, ra=36.5445, dec=-4.3366, er=1.19),
    "13369": dict(z_lens=0.354, z_source=1.41, ra=215.0142, dec=52.5272, er=1.79),
    "12971": dict(z_lens=0.716, z_source=1.47, ra=211.7097, dec=52.4387, er=0.94),
    "13246": dict(z_lens=0.2987, z_source=0.8115, ra=214.09304, dec=51.60839, er=1.37),
}


def image_hdu(path: Path, shape: tuple[int, int] | None = None) -> tuple[int, tuple[int, int]]:
    with fits.open(path, memmap=True) as hdul:
        for index, hdu in enumerate(hdul):
            value = getattr(hdu, "data", None)
            if value is None or getattr(value, "ndim", 0) != 2:
                continue
            candidate = tuple(int(v) for v in value.shape)
            if shape is None or candidate == shape:
                return index, candidate
    raise ValueError(f"No matching 2D image HDU in {path}")


def positive_uncertainty_hdu(path: Path, shape: tuple[int, int]) -> int:
    """Select a variance/weight plane, never the co-stored science plane.

    HSC variance downloads retain the science image in HDU 1 and variance in
    HDU 3.  Shape-only discovery therefore silently chooses the wrong plane.
    A physical uncertainty map must be finite and strictly positive almost
    everywhere; choose the last matching plane satisfying that invariant.
    """
    candidates = []
    with fits.open(path, memmap=True) as hdul:
        for index, hdu in enumerate(hdul):
            value = getattr(hdu, "data", None)
            if value is None or getattr(value, "ndim", 0) != 2 or tuple(value.shape) != shape:
                continue
            import numpy as np
            sample = np.asarray(value[:: max(1, value.shape[0] // 128), :: max(1, value.shape[1] // 128)], dtype=float)
            finite = np.isfinite(sample)
            positive_fraction = float(np.mean(finite & (sample > 0.0)))
            if positive_fraction > 0.99:
                candidates.append(index)
    if not candidates:
        raise ValueError(f"No predominantly positive uncertainty HDU in {path}")
    return candidates[-1]


def position_survey_band(path: Path) -> tuple[str | None, str | None]:
    first = "\n".join(path.read_text(encoding="utf-8").splitlines()[:5]).upper()
    match = re.search(r"(HSC|CFHT)-([UGRIZY]+)", first)
    if match:
        return match.group(1), f"{match.group(1)}-{match.group(2)}"
    # Older standardized CFHT position files retained the raw MegaCam name.
    raw = re.search(r"CFHTLS[^\n/]*?_([UGRIZ])_", first)
    if raw:
        return "CFHT", f"CFHT-{raw.group(1)}"
    return None, None


def complete_bands(system_dir: Path, survey: str) -> list[str]:
    science = system_dir / "science"
    if not science.is_dir():
        return []
    result = []
    for data in sorted(science.glob(f"{survey}-*.fits")):
        band = data.stem.upper()
        psf = system_dir / "psf" / data.name
        variance = system_dir / "variance" / data.name
        weight = system_dir / "weight" / data.name
        uncertainty = variance if survey == "HSC" else (weight if weight.is_file() else variance)
        if psf.is_file() and uncertainty.is_file():
            result.append(band)
    return result


def choose_backend(system: str) -> tuple[str, str, str]:
    system_dir = ROOT / "data" / "bb" / system
    positions = system_dir / "positions_gui.txt"
    if not positions.is_file():
        raise FileNotFoundError(f"Missing positions file: {positions}")
    position_survey, position_band = position_survey_band(positions)
    hsc = complete_bands(system_dir, "HSC")
    cfht = complete_bands(system_dir, "CFHT")

    if system in PREFERRED_MODEL_BANDS:
        survey, band = PREFERRED_MODEL_BANDS[system]
        available = hsc if survey == "HSC" else cfht
        if band not in available:
            raise FileNotFoundError(
                f"Validated preferred band {band} is incomplete for system {system}: "
                f"available={available}"
            )
        return survey, band, "validated G-band modeling preference"

    # Pixel positions are tied to the displayed FITS grid.  HSC is preferred
    # only when the current positions file was measured on a complete HSC band.
    if position_survey == "HSC" and hsc:
        band = position_band if position_band in hsc else hsc[0]
        return "HSC", band, "complete HSC preferred"
    if position_survey == "CFHT" and cfht:
        band = position_band if position_band in cfht else cfht[0]
        return "CFHT", band, "HSC incomplete/incompatible positions; CFHT fallback"
    if hsc:
        return "HSC", hsc[0], "complete HSC preferred (band fallback)"
    if cfht:
        return "CFHT", cfht[0], "no complete HSC; CFHT fallback"
    raise FileNotFoundError(
        f"System {system} has no complete science+uncertainty+PSF BB set; "
        f"HSC={hsc}, CFHT={cfht}, positions survey={position_survey}"
    )


def build(system: str) -> dict:
    survey, band, reason = choose_backend(system)
    system_dir = ROOT / "data" / "bb" / system
    filename = f"{band}.fits"
    science = system_dir / "science" / filename
    psf = system_dir / "psf" / filename
    science_hdu, shape = image_hdu(science)
    uncertainty_kind = "variance"
    uncertainty = system_dir / "variance" / filename
    if survey == "CFHT" and (system_dir / "weight" / filename).is_file():
        uncertainty_kind = "weight"
        uncertainty = system_dir / "weight" / filename
    uncertainty_hdu = positive_uncertainty_hdu(uncertainty, shape=shape)
    meta = META[system]
    # Configs live in ``lens-modeling/system_<id>``; keep every input relative
    # to that directory so the whole PHOTO-LENS03 tree is relocatable.
    relative_root = Path("../../data/bb") / system
    config = {
        "system": system,
        "science_cutout": str(relative_root / "science" / filename),
        f"{uncertainty_kind}_cutout": str(relative_root / uncertainty.parent.name / filename),
        "psf": str(relative_root / "psf" / filename),
        "positions_file": str(relative_root / "positions_gui.txt"),
        "output_dir": "output",
        "z_lens": meta["z_lens"], "z_source": meta["z_source"],
        "lens_ra_deg": meta["ra"], "lens_dec_deg": meta["dec"],
        "einstein_radius_mean": meta["er"],
        "pixel_scale": 0.168 if survey == "HSC" else 0.187,
        "science_hdu": science_hdu,
        f"{uncertainty_kind}_hdu": uncertainty_hdu,
        "light_centre_sigma": 0.2, "mass_centre_sigma": 0.2,
        "lens_gaussians_per_basis": 15, "lens_bases": 2,
        "source_gaussians": 10, "n_live": 75, "n_eff": 100,
        "n_batch": 5, "number_of_cores": 1, "psf_size": 31,
        "over_sample": True, "use_jax": True, "positions_threshold": 0.5,
        "satellite_light_centre_sigma": 0.08,
        "stage4_mesh_shape": [20, 20], "stage4_use_jax": True,
        "stage4_regularization": "constant_brightness_zeroth",
        "stage4_signal_scale": 0.5,
        "bb_survey": survey, "bb_band": band,
    }
    return config, reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("systems", nargs="*", default=list(SYSTEMS))
    args = parser.parse_args()
    manifest = {}
    for system in args.systems:
        config, reason = build(system)
        target = ROOT / "lens-modeling" / f"system_{system}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        manifest[system] = {
            "survey": config["bb_survey"], "band": config["bb_band"],
            "reason": reason, "config": str(target / "config.json"),
        }
        print(f"[{system}] {config['bb_survey']} {config['bb_band']}: {reason}")
    (ROOT / "lens-modeling" / "bb_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
