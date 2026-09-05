#!/usr/bin/env python3
"""Measure per-band PAUS NB antisymmetry centres relative to positions_gui BB."""
from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp


HERE = Path(__file__).resolve().parent
PHOTO_LENS_ROOT = HERE.parent
SPECIAL = {"NB595", "NB605", "NB815", "NB845"}


def load_estimator():
    path = PHOTO_LENS_ROOT / "region-detection/measure_symmetry_centre.py"
    spec = importlib.util.spec_from_file_location("nb_symmetry_estimator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load estimator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_2d(path: Path):
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
                return np.asarray(hdu.data, dtype=float), hdu.header.copy()
    raise ValueError(f"No 2-D image in {path}")


def bb_reference(system: int) -> dict:
    positions = PHOTO_LENS_ROOT / f"data/bb/{system}/positions_gui.txt"
    text = positions.read_text()
    image = re.search(r"^# FITS image:\s*(.+?)\s*$", text, re.M)
    pixel = re.search(
        r"^PIXEL_LENS_LIGHT_CENTRE_FLOAT\s*=\s*\(([^,]+),\s*([^\)]+)\)",
        text,
        re.M,
    )
    pixel_integer = re.search(
        r"^PIXEL_LENS_LIGHT_CENTRE\s*=\s*\(([^,]+),\s*([^\)]+)\)",
        text,
        re.M,
    )
    radii = re.search(
        r"^LENS_LIGHT_CENTRE_ANTISYMMETRY_RADII_PIX\s*=\s*"
        r"\(([^,]+),\s*([^\)]+)\)",
        text,
        re.M,
    )
    method = re.search(
        r"^LENS_LIGHT_CENTRE_METHOD\s*=\s*['\"]([^'\"]+)", text, re.M
    )
    mask_radius = re.search(r"^MASK_RADIUS\s*=\s*([^\s]+)", text, re.M)
    if not image or (pixel is None and pixel_integer is None):
        raise ValueError(f"Incomplete positions_gui metadata: {positions}")
    image_path = Path(image.group(1)).expanduser()
    if not image_path.is_absolute():
        image_path = positions.parent / image_path
    image_path = image_path.resolve()
    _, header = first_2d(image_path)
    wcs = WCS(header).celestial
    chosen_pixel = pixel if pixel is not None else pixel_integer
    assert chosen_pixel is not None
    row, col = float(chosen_pixel.group(1)), float(chosen_pixel.group(2))
    ra, dec = wcs.pixel_to_world_values(col, row)
    scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    if radii is not None:
        rin_arcsec = float(radii.group(1)) * scale
        rout_arcsec = float(radii.group(2)) * scale
    else:
        # Some updated CFHT GUI files deliberately preserve a manual/existing
        # lens centre and therefore store no antisymmetry-fit diagnostics.  The
        # centre remains the authoritative BB reference; use the same physical
        # annulus scale as the automatically fitted HSC/older-CFHT files solely
        # for measuring relative PAUS-NB offsets.
        radius_limit = 1.6
        if mask_radius is not None:
            radius_limit = min(radius_limit, 0.75 * float(mask_radius.group(1)))
        rin_arcsec = 1.5 * scale
        rout_arcsec = radius_limit
    if rout_arcsec <= rin_arcsec:
        raise ValueError(f"Invalid antisymmetry annulus derived from {positions}")
    return {
        "positions_file": positions,
        "image": image_path,
        "ra_deg": float(ra),
        "dec_deg": float(dec),
        "rin_arcsec": rin_arcsec,
        "rout_arcsec": rout_arcsec,
        "reference_method": method.group(1) if method else "positions_gui_pixel_centre",
    }


def _single_epoch_pairs(system: int, wavelength: int) -> list[tuple[Path, Path]]:
    root = PHOTO_LENS_ROOT / f"data/nb/single_epoch/{wavelength}/{system}"
    pairs = []
    for science in sorted(
        path for path in root.glob("*.fits") if ".weight." not in path.name
    ):
        weight = science.with_name(science.name.replace(".fits", ".weight.fits"))
        if weight.is_file():
            pairs.append((science, weight))
    return pairs


def _write_inverse_variance_coadd(
    system: int, wavelength: int, science_path: Path, weight_path: Path
) -> None:
    """Build the centre-diagnostic coadd when only single epochs are available."""
    pairs = _single_epoch_pairs(system, wavelength)
    if not pairs:
        raise FileNotFoundError(
            f"No coadd or single-epoch science/weight pair for {system} NB{wavelength}"
        )
    reference, header = first_2d(pairs[0][0])
    reference_wcs = WCS(header).celestial
    shape = reference.shape
    weighted_sum = np.zeros(shape, dtype=float)
    weight_sum = np.zeros(shape, dtype=float)
    for input_science, input_weight in pairs:
        image, image_header = first_2d(input_science)
        weight, weight_header = first_2d(input_weight)
        image_reprojected, image_footprint = reproject_interp(
            (image, WCS(image_header).celestial), reference_wcs, shape_out=shape
        )
        weight_reprojected, weight_footprint = reproject_interp(
            (weight, WCS(weight_header).celestial), reference_wcs, shape_out=shape
        )
        good = (
            np.isfinite(image_reprojected)
            & np.isfinite(weight_reprojected)
            & (weight_reprojected > 0.0)
            & (image_footprint > 0.0)
            & (weight_footprint > 0.0)
        )
        weighted_sum[good] += image_reprojected[good] * weight_reprojected[good]
        weight_sum[good] += weight_reprojected[good]
    coadd = np.full(shape, np.nan, dtype=float)
    good = weight_sum > 0.0
    coadd[good] = weighted_sum[good] / weight_sum[good]
    header["NCOMBINE"] = (len(pairs), "Number of inverse-variance combined epochs")
    science_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(coadd, header=header).writeto(science_path, overwrite=True)
    fits.PrimaryHDU(weight_sum, header=header).writeto(weight_path, overwrite=True)


def coadd_paths(system: int, wavelength: int) -> tuple[Path, Path]:
    root = PHOTO_LENS_ROOT / f"data/nb/coadd/{wavelength}/{system}"
    science = sorted(
        p for p in root.glob("coadd_cutout*.fits")
        if not p.name.endswith(".weight.fits")
    )
    weight = sorted(root.glob("*.weight.fits"))
    if not science and not weight:
        science_path = root / f"coadd_cutout_{system}.fits"
        weight_path = root / f"coadd_cutout_{system}.weight.fits"
        _write_inverse_variance_coadd(
            system, wavelength, science_path, weight_path
        )
        science, weight = [science_path], [weight_path]
    if len(science) != 1 or len(weight) != 1:
        raise FileNotFoundError(
            f"Expected one science and one inverse-variance coadd in {root}"
        )
    return science[0], weight[0]


def measure(system: int, max_shift_arcsec: float | None) -> tuple[pd.DataFrame, dict]:
    estimator = load_estimator()
    bb = bb_reference(system)
    rows = []
    for wavelength in range(455, 846, 10):
        band = f"NB{wavelength}"
        try:
            science_path, weight_path = coadd_paths(system, wavelength)
        except FileNotFoundError:
            rows.append({
                "system": system,
                "band": band,
                "wavelength_nm": wavelength,
                "ra_deg": bb["ra_deg"],
                "dec_deg": bb["dec_deg"],
                "bb_ra_deg": bb["ra_deg"],
                "bb_dec_deg": bb["dec_deg"],
                "delta_ra_cosdec_arcsec": np.nan,
                "delta_dec_arcsec": np.nan,
                "separation_from_bb_arcsec": np.nan,
                "max_coordinate_correction_arcsec": max_shift_arcsec,
                "apply_coordinate_correction": False,
                "x_pixel_zero_indexed": np.nan,
                "y_pixel_zero_indexed": np.nan,
                "pixel_scale_arcsec": np.nan,
                "r_in_pix": np.nan,
                "r_out_pix": np.nan,
                "chi2_per_pixel": np.nan,
                "n_fit_pixels": 0,
                "minimum_at_search_edge": False,
                "method": "no_nb_data_fallback_to_bb_centre",
                "science_file": "",
                "inverse_variance_file": "",
                "data_available": False,
            })
            continue
        image, header = first_2d(science_path)
        weight, _ = first_2d(weight_path)
        if image.shape != weight.shape:
            raise ValueError(f"Science/weight shape mismatch for system {system} {band}")
        wcs = WCS(header).celestial
        scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
        variance = np.full(image.shape, 1.0e30, dtype=float)
        good = np.isfinite(weight) & (weight > 0.0)
        variance[good] = 1.0 / weight[good]
        x0, y0 = (
            float(v) for v in wcs.world_to_pixel_values(bb["ra_deg"], bb["dec_deg"])
        )
        half = 1.5
        for _ in range(3):
            fit = estimator.fit_centre(
                image,
                variance,
                x0,
                y0,
                bb["rin_arcsec"] / scale,
                bb["rout_arcsec"] / scale,
                half=half,
                step=0.05,
                order=3,
            )
            if not fit["at_edge"]:
                break
            half *= 2.0
        ra, dec = (
            float(v) for v in wcs.pixel_to_world_values(fit["x"], fit["y"])
        )
        dra = (
            (ra - bb["ra_deg"])
            * np.cos(np.deg2rad(bb["dec_deg"]))
            * 3600.0
        )
        ddec = (dec - bb["dec_deg"]) * 3600.0
        separation = float(np.hypot(dra, ddec))
        apply_correction = bool(
            max_shift_arcsec is None or separation <= max_shift_arcsec
        )
        rows.append({
            "system": system,
            "band": band,
            "wavelength_nm": wavelength,
            "ra_deg": ra,
            "dec_deg": dec,
            "bb_ra_deg": bb["ra_deg"],
            "bb_dec_deg": bb["dec_deg"],
            "delta_ra_cosdec_arcsec": dra,
            "delta_dec_arcsec": ddec,
            "separation_from_bb_arcsec": separation,
            "max_coordinate_correction_arcsec": max_shift_arcsec,
            "apply_coordinate_correction": apply_correction,
            "x_pixel_zero_indexed": float(fit["x"]),
            "y_pixel_zero_indexed": float(fit["y"]),
            "pixel_scale_arcsec": scale,
            "r_in_pix": bb["rin_arcsec"] / scale,
            "r_out_pix": bb["rout_arcsec"] / scale,
            "chi2_per_pixel": float(fit["chi2_min"]),
            "n_fit_pixels": int(fit["n_pix"]),
            "minimum_at_search_edge": bool(fit["at_edge"]),
            "method": "variance_weighted_point_antisymmetry_minimum",
            "science_file": science_path.relative_to(PHOTO_LENS_ROOT).as_posix(),
            "inverse_variance_file": weight_path.relative_to(PHOTO_LENS_ROOT).as_posix(),
            "data_available": True,
        })
    return pd.DataFrame(rows), bb


def save_plot(table: pd.DataFrame, bb: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    regular = ~table.band.isin(SPECIAL)
    points = axes[0].scatter(
        table.loc[regular, "delta_ra_cosdec_arcsec"],
        table.loc[regular, "delta_dec_arcsec"],
        c=table.loc[regular, "wavelength_nm"],
        cmap="viridis",
        s=45,
    )
    axes[0].scatter(0, 0, marker="+", s=220, color="black", linewidth=2, label="BB centre")
    for row in table.loc[~regular].itertuples(index=False):
        axes[0].scatter(
            row.delta_ra_cosdec_arcsec, row.delta_dec_arcsec,
            marker="*", s=160, edgecolor="black", linewidth=0.5,
        )
        axes[0].annotate(row.band, (row.delta_ra_cosdec_arcsec, row.delta_dec_arcsec),
                         xytext=(5, 5), textcoords="offset points", fontsize=8)
    rejected = ~table.apply_coordinate_correction.astype(bool)
    if rejected.any():
        axes[0].scatter(
            table.loc[rejected, "delta_ra_cosdec_arcsec"],
            table.loc[rejected, "delta_dec_arcsec"],
            marker="x", s=95, color="red", linewidth=2.0,
            label="rejected (> threshold)", zorder=6,
        )
    axes[0].axhline(0, color="0.75", lw=0.8)
    axes[0].axvline(0, color="0.75", lw=0.8)
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set(
        xlabel=r"$\Delta$RA cos(Dec) from BB [arcsec]",
        ylabel=r"$\Delta$Dec from BB [arcsec]",
        title="NB antisymmetry centres in the sky plane",
    )
    axes[0].legend(loc="best")
    fig.colorbar(points, ax=axes[0], label="NB wavelength [nm]")
    axes[1].plot(table.wavelength_nm, table.delta_ra_cosdec_arcsec, "o-", ms=4,
                 label=r"$\Delta$RA cos(Dec)")
    axes[1].plot(table.wavelength_nm, table.delta_dec_arcsec, "o-", ms=4,
                 label=r"$\Delta$Dec")
    axes[1].axhline(0, color="black", lw=0.8)
    for wavelength in table.loc[rejected, "wavelength_nm"]:
        axes[1].axvline(wavelength, color="red", alpha=0.13, lw=1.0)
    axes[1].set(
        xlabel="NB wavelength [nm]", ylabel="offset from BB [arcsec]",
        title="Band-level shifts applied to source and background",
    )
    axes[1].legend(loc="best")
    fig.suptitle(
        f"System {int(table.system.iloc[0])}: per-NB aperture coordinate calibration\n"
        f"BB RA={bb['ra_deg']:.9f}°, Dec={bb['dec_deg']:.9f}°",
        fontsize=14,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-shift-arcsec",
        type=float,
        default=None,
        help="Reject coordinate corrections with radial NB-BB offset above this value.",
    )
    args = parser.parse_args()
    if args.max_shift_arcsec is not None and args.max_shift_arcsec <= 0:
        parser.error("--max-shift-arcsec must be positive")
    for system in args.system:
        output_dir = args.output_root.resolve() / f"system_{system}"
        output_dir.mkdir(parents=True, exist_ok=True)
        table, bb = measure(system, args.max_shift_arcsec)
        csv_path = output_dir / f"system{system}_nb_antisymmetry_centres.csv"
        png_path = csv_path.with_suffix(".png")
        table.to_csv(csv_path, index=False)
        save_plot(table, bb, png_path)
        print(csv_path)


if __name__ == "__main__":
    main()
