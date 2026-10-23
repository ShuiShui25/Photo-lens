#!/usr/bin/env python3
"""Measure the point-symmetry centre of the lens light directly from the data.

The photometry self-subtraction ``F(R) - F(P.R)`` needs the centre about which
the *observed* (PSF-convolved) image is point-symmetric.  That centre is
``c_galaxy + d_PSF``; it is not the AutoLens galaxy centre, and it does not have
to be read from the WCS products at all.

Estimator
---------
A relaxed early-type is point-symmetric about its centre, PSF convolution
preserves point symmetry, and a flat sky is symmetric about every point.  So for
the correct centre ``c`` the antisymmetric residual

    A_c(x) = I(x) - I(2c - x)

is pure noise wherever the lens dominates.  For a trial centre offset by delta,
``A_c ~ 2 delta . grad L``: the residual is the offset times the local gradient,
which is why the statistic is sharp near a bright core.  We minimise

    Q(c) = sum_annulus A_c(x)^2 / (var(x) + var(2c - x))

over a grid in ``c``, then refine parabolically.

The annulus is defined about the *trial* centre, so it is point-symmetric about
that centre by construction and the statistic is never biased by a half-truncated
pixel set.

Usage
-----
    python measure_symmetry_centre.py --system J022511-045433
    python measure_symmetry_centre.py --system J022511-045433 --out-csv centres.csv
    python measure_symmetry_centre.py --system J022511-045433 --rin 1.5 --rout 4.5 --plot

Run with the ``bcnz`` environment (numpy / scipy / astropy; matplotlib only for
``--plot``).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
BAND_ORDER = ["G", "R", "I", "Z", "Y"]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _first_image_hdu(path: Path):
    """Return the first HDU holding 2-D data. Cutouts vary: some put the image in
    the primary HDU, some in the first extension."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data.astype(float), hdu.header
    raise ValueError(f"no 2-D image HDU in {path}")


def _variance_image_hdu(path: Path):
    """Return the variance plane from an HSC cutout.

    Variance downloads may contain both a copy of the science image and a
    later ``VARIANCE`` extension.  Prefer the named extension; retain support
    for legacy single-plane variance files by falling back to the last 2-D
    image HDU.
    """
    fallback = None
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is None or hdu.data.ndim != 2:
                continue
            item = (hdu.data.astype(float), hdu.header)
            fallback = item
            if hdu.name.strip().upper() == "VARIANCE":
                return item
    if fallback is not None:
        return fallback
    raise ValueError(f"no 2-D variance HDU in {path}")


def available_bands(root: Path, system: str) -> list[str]:
    sci = root / "data/bb" / str(system) / "science"
    found = {p.stem.split("-")[-1].upper() for p in sci.glob("HSC-*.fits")}
    return [b for b in BAND_ORDER if b in found] + sorted(found - set(BAND_ORDER))


def load_band(root: Path, system: str, band: str):
    base = root / "data/bb" / str(system)
    img, hdr = _first_image_hdu(base / "science" / f"HSC-{band}.fits")
    try:
        var, _ = _variance_image_hdu(base / "variance" / f"HSC-{band}.fits")
        if var.shape != img.shape:
            var = None
    except (FileNotFoundError, ValueError):
        var = None
    if var is None:
        # Constant weight from a robust scatter estimate. The argmin is invariant
        # to an overall scale; only the *spatial* weighting is lost.
        finite = img[np.isfinite(img)]
        sigma = 1.4826 * np.median(np.abs(finite - np.median(finite)))
        var = np.full_like(img, max(sigma, 1e-30) ** 2)
    return img, var, WCS(hdr)


# --------------------------------------------------------------------------- #
# DS9 regions
# --------------------------------------------------------------------------- #
def _sexagesimal(token: str, is_ra: bool) -> float:
    token = token.strip()
    if ":" not in token:
        return float(token)
    parts = [float(p) for p in token.split(":")]
    sign = -1.0 if token.lstrip().startswith("-") else 1.0
    value = abs(parts[0]) + parts[1] / 60.0 + (parts[2] if len(parts) > 2 else 0.0) / 3600.0
    return sign * value * (15.0 if is_ra else 1.0)


def read_region_vertices(path: Path) -> np.ndarray | None:
    """Sky polygon vertices in ICRS, respecting the DS9 frame declaration."""
    from regions import Regions
    for region in Regions.read(path, format='ds9'):
        if hasattr(region, 'vertices') and hasattr(region.vertices, 'icrs'):
            vertices = region.vertices.icrs
            return np.column_stack((vertices.ra.deg, vertices.dec.deg))
    return None


def read_region_centre(path: Path) -> tuple[float, float] | None:
    """Centre of an ``ellipse``/``circle`` region, in degrees."""
    from regions import Regions
    for region in Regions.read(path, format='ds9'):
        if hasattr(region, 'center') and hasattr(region.center, 'icrs'):
            centre = region.center.icrs
            return float(centre.ra.deg), float(centre.dec.deg)
    return None


def reflection_centre(source: Path, background: Path) -> tuple[float, float] | None:
    """Centre implied by a source/background region pair, if they are an exact
    point reflection of one another."""
    a, b = read_region_vertices(source), read_region_vertices(background)
    if a is None or b is None or a.shape != b.shape:
        return None
    mid = 0.5 * (a + b)
    cos_dec = math.cos(math.radians(float(mid[:, 1].mean())))
    scatter = max(float(mid[:, 0].std()) * cos_dec, float(mid[:, 1].std())) * 3600.0
    if scatter > 0.01:  # arcsec; not a clean reflection pair
        return None
    return float(mid[:, 0].mean()), float(mid[:, 1].mean())


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #
def _stamp(img, var, xc, yc, pad):
    x0, y0 = int(round(xc)), int(round(yc))
    ylo, yhi = max(y0 - pad, 0), min(y0 + pad + 1, img.shape[0])
    xlo, xhi = max(x0 - pad, 0), min(x0 + pad + 1, img.shape[1])
    sl = (slice(ylo, yhi), slice(xlo, xhi))
    return np.nan_to_num(img[sl]), np.nan_to_num(var[sl]), xc - xlo, yc - ylo


def _project_out_plane(values, wx, wy, weights):
    """Remove the component of the residual that a linear sky gradient would
    produce: a sky ``a + g.x`` gives ``A = 2 g.(x - c)``, which is antisymmetric
    and therefore degenerate with a genuine centre offset."""
    design = np.column_stack([wx, wy])
    w = 1.0 / np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(design * w[:, None], values * w, rcond=None)
    return values - design @ coef


def chi2_at(img, var, x0, y0, rin, rout, order, mask=None, fit_sky_gradient=False):
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx, dy = xx - x0, yy - y0
    radius = np.hypot(dx, dy)
    sel = (radius >= rin) & (radius <= rout)
    if mask is not None:
        sel &= ~mask
    if sel.sum() < 8:
        return np.inf, 0

    coords = [2 * y0 - yy, 2 * x0 - xx]
    mirror_img = map_coordinates(img, coords, order=order, mode="nearest")
    mirror_var = map_coordinates(var, coords, order=1, mode="nearest")
    if mask is not None:
        # A masked pixel must not contribute through its mirror either.
        mirror_masked = map_coordinates(
            mask.astype(float), coords, order=1, mode="nearest"
        ) > 1e-6
        sel &= ~mirror_masked
        if sel.sum() < 8:
            return np.inf, 0

    resid = (img - mirror_img)[sel]
    weight = (var + mirror_var)[sel]
    weight = np.where(weight > 0, weight, np.inf)
    if fit_sky_gradient:
        resid = _project_out_plane(resid, dx[sel], dy[sel], weight)
    return float(np.sum(resid**2 / weight) / sel.sum()), int(sel.sum())


def _parabolic(values, index, step):
    if not 0 < index < len(values) - 1:
        return 0.0
    denom = values[index - 1] - 2 * values[index] + values[index + 1]
    if denom == 0:
        return 0.0
    shift = 0.5 * (values[index - 1] - values[index + 1]) / denom * step
    return float(np.clip(shift, -step, step))


def fit_centre(img, var, xc, yc, rin, rout, *, half=1.5, step=0.05, order=3,
               pad=None, mask=None, fit_sky_gradient=False):
    """Grid scan plus parabolic refinement.

    A grid scan is used rather than a simplex: Nelder-Mead with its default
    initial simplex has been observed to return a zero displacement (it never
    leaves the starting point) on this problem.
    """
    pad = pad or int(math.ceil(rout)) + int(math.ceil(half)) + 6
    stamp, svar, sx, sy = _stamp(img, var, xc, yc, pad)
    smask = None
    if mask is not None:
        smask, _, _, _ = _stamp(mask.astype(float), var, xc, yc, pad)
        smask = smask > 0.5

    grid = np.arange(-half, half + 1e-9, step)
    surface = np.empty((grid.size, grid.size))
    npix = 0
    for i, ddy in enumerate(grid):
        for j, ddx in enumerate(grid):
            surface[i, j], npix = chi2_at(
                stamp, svar, sx + ddx, sy + ddy, rin, rout, order,
                mask=smask, fit_sky_gradient=fit_sky_gradient,
            )

    i, j = np.unravel_index(np.argmin(surface), surface.shape)
    dxr = grid[j] + _parabolic(surface[i, :], j, step)
    dyr = grid[i] + _parabolic(surface[:, j], i, step)
    edge = min(i, j, grid.size - 1 - i, grid.size - 1 - j) == 0
    return {
        "x": xc + dxr, "y": yc + dyr, "dx": dxr, "dy": dyr,
        "chi2_min": float(surface[i, j]), "n_pix": npix,
        "at_edge": bool(edge), "surface": surface, "grid": grid,
    }


# --------------------------------------------------------------------------- #
# Reference centres
# --------------------------------------------------------------------------- #
def wcs_product_centre(root: Path, system: str) -> tuple[float, float] | None:
    path = (root / "1.lens-modeling" / f"{system}"
            / "output/wcs_products_stage4" / f"{system}_lens_light_centre_wcs.json")
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if "ra_deg" in data and "dec_deg" in data:
        return float(data["ra_deg"]), float(data["dec_deg"])
    return None


def manual_centre(root: Path, system: str) -> tuple[float, float] | None:
    pair = reflection_centre(
        root / "2.region-detection/regions" / system / f"{system}_source_point_fraction040.reg",
        root / "2.region-detection/regions" / system / f"{system}_background_point_fraction040.reg",
    ) if (root / "2.region-detection/regions" / system / f"{system}_background_point_fraction040.reg").exists() else None
    if pair is not None:
        return pair
    lens = root / "2.region-detection/regions" / system / f"{system}_lens_point_fraction040.reg"
    return read_region_centre(lens) if lens.exists() else None


def arc_inner_radius_pix(region: Path, wcs: WCS, x0: float, y0: float) -> float | None:
    """Smallest distance from the centre to the source region, in pixels."""
    vertices = read_region_vertices(region)
    if vertices is None:
        return None
    from astropy.coordinates import SkyCoord
    px, py = wcs.world_to_pixel(SkyCoord(vertices[:, 0], vertices[:, 1], unit='deg', frame='icrs'))
    return float(np.min(np.hypot(np.asarray(px) - x0, np.asarray(py) - y0)))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_system(args, root: Path, system: str) -> list[dict]:
    bands = args.bands or available_bands(root, system)
    if not bands:
        print(f"system {system}: no HSC science cutouts found", file=sys.stderr)
        return []

    refs = {}
    if (c := manual_centre(root, system)) is not None:
        refs["manual"] = c
    if (c := wcs_product_centre(root, system)) is not None:
        refs["wcs_product"] = c

    if args.centre_ra is not None and args.centre_dec is not None:
        start = (args.centre_ra, args.centre_dec)
    elif args.start in refs:
        start = refs[args.start]
    elif refs:
        start = next(iter(refs.values()))
    else:
        print(f"system {system}: no reference centre; pass --centre-ra/--centre-dec",
              file=sys.stderr)
        return []

    source_region = Path(args.source_region) if args.source_region else None
    if source_region is None:
        auto = root / "2.region-detection/regions" / system / f"{system}_source_point_fraction040.reg"
        source_region = auto if auto.exists() else None

    print(f"\n=== system {system}   start = ({start[0]:.7f}, {start[1]:.7f})")
    for name, (ra, dec) in refs.items():
        print(f"    reference {name:12s} = ({ra:.7f}, {dec:.7f})")

    rows = []
    for band in bands:
        try:
            img, var, wcs = load_band(root, system, band)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  {band}: skipped ({exc})")
            continue
        from astropy.coordinates import SkyCoord
        x0, y0 = (float(v) for v in wcs.world_to_pixel(SkyCoord(*start, unit='deg', frame='icrs')))

        rout = args.rout
        if rout is None:
            r_arc = (arc_inner_radius_pix(source_region, wcs, x0, y0)
                     if source_region else None)
            if r_arc is None:
                print(f"  {band}: skipped (no --rout and no source region to derive it)")
                continue
            rout = args.arc_margin * r_arc

        fit = fit_centre(img, var, x0, y0, args.rin, rout,
                         half=args.half, step=args.step, order=args.order,
                         fit_sky_gradient=args.fit_sky_gradient)
        sky_fit = wcs.pixel_to_world(fit['x'], fit['y']).icrs
        ra_fit, dec_fit = float(sky_fit.ra.deg), float(sky_fit.dec.deg)

        row = {"system": system, "band": band, "ra_deg": ra_fit, "dec_deg": dec_fit,
               "x_pix": fit["x"], "y_pix": fit["y"], "r_in_pix": args.rin,
               "r_out_pix": rout, "n_pix": fit["n_pix"],
               "chi2_min": fit["chi2_min"], "at_edge": fit["at_edge"]}
        parts = [f"  {band:2s} r=[{args.rin:.1f},{rout:.1f}]px n={fit['n_pix']:4d}"
                 f" chi2/N={fit['chi2_min']:8.4f}"]
        for name, (ra, dec) in refs.items():
            rx, ry = (float(v) for v in wcs.world_to_pixel(SkyCoord(ra, dec, unit='deg', frame='icrs')))
            row[f"d{name}_x_pix"] = fit["x"] - rx
            row[f"d{name}_y_pix"] = fit["y"] - ry
            parts.append(f" {name}=({fit['x']-rx:+6.3f},{fit['y']-ry:+6.3f})")
        if fit["at_edge"]:
            parts.append("  [WARNING: minimum on scan edge, raise --half]")
        print("".join(parts))
        rows.append(row)

        if args.plot:
            _save_plot(args.plot, system, band, fit)

    _report_scatter(rows, refs, args.scatter_bands)
    return rows


def _report_scatter(rows, refs, scatter_bands):
    subset = [r for r in rows if r["band"] in scatter_bands] or rows
    if len(subset) < 2:
        return
    print(f"    band-to-band scatter over {'+'.join(r['band'] for r in subset)}:")
    for name in refs:
        dx = np.array([r[f"d{name}_x_pix"] for r in subset])
        dy = np.array([r[f"d{name}_y_pix"] for r in subset])
        print(f"      vs {name:12s} mean=({dx.mean():+6.3f},{dy.mean():+6.3f})"
              f"  std=({dx.std():.3f},{dy.std():.3f}) px")


def _save_plot(outdir, system, band, fit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid = fit["grid"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.imshow(fit["surface"], origin="lower",
               extent=[grid[0], grid[-1], grid[0], grid[-1]])
    ax1.plot(fit["dx"], fit["dy"], "r+", ms=12)
    ax1.set(xlabel="dx [pix]", ylabel="dy [pix]", title=f"system {system} HSC-{band}")
    row = fit["surface"][np.argmin(np.min(fit["surface"], axis=1))]
    ax2.plot(grid, row, ".-")
    ax2.axvline(fit["dx"], color="r")
    ax2.set(xlabel="dx [pix]", ylabel=r"$\chi^2/N$", title="cut through minimum")
    fig.tight_layout()
    fig.savefig(outdir / f"symmetry_centre_{system}_{band}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", type=str, nargs="+", required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="repository root (default: inferred from this file)")
    p.add_argument("--bands", nargs="+", default=None,
                   help="default: every HSC band present for the system")
    p.add_argument("--start", default="manual", choices=["manual", "wcs_product"],
                   help="which reference centre to start the scan from")
    p.add_argument("--centre-ra", type=float, default=None)
    p.add_argument("--centre-dec", type=float, default=None)
    p.add_argument("--rin", type=float, default=1.5,
                   help="inner annulus radius in pixels; excludes the core where "
                        "mirror interpolation error is largest (default: 1.5)")
    p.add_argument("--rout", type=float, default=None,
                   help="outer annulus radius in pixels; default is "
                        "--arc-margin times the source region's inner radius")
    p.add_argument("--arc-margin", type=float, default=0.9,
                   help="safety factor keeping the annulus inside the arc (0.9)")
    p.add_argument("--source-region", default=None,
                   help="DS9 region used only to derive --rout")
    p.add_argument("--half", type=float, default=1.5, help="scan half-range, pixels")
    p.add_argument("--step", type=float, default=0.05, help="scan step, pixels")
    p.add_argument("--order", type=int, default=3, choices=[1, 2, 3, 4, 5],
                   help="spline order for the mirror; never use 0 (nearest "
                        "neighbour shifts the result by tens of percent)")
    p.add_argument("--fit-sky-gradient", action="store_true",
                   help="project out the linear term, which a sky gradient would "
                        "otherwise inject as a spurious centre offset")
    p.add_argument("--scatter-bands", nargs="+", default=["R", "I", "Z", "Y"],
                   help="bands used for the scatter summary; G sits on a "
                        "different grid for some systems (default: R I Z Y)")
    p.add_argument("--plot", default=None, metavar="DIR",
                   help="write a chi2 map and cut per band into DIR")
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    rows = []
    for system in args.system:
        rows.extend(run_system(args, args.root, system))

    if args.out_csv and rows:
        import csv
        keys = sorted({k for r in rows for k in r})
        head = ["system", "band"] + [k for k in keys if k not in ("system", "band")]
        with args.out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=head)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
