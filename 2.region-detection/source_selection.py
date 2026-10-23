#!/usr/bin/env python3
"""Image-only anchors and single-pass source selection.

The same selection feeds the region diagnostics and downstream photometry:

1. One SExtractor segmentation of the lens-plane-subtracted image at
   ``ANCHOR_DETECT_SIGMA`` (cached; shared by anchors and source selection).
2. Anchors: the brightest pixel of every component, except components whose
   peak lies in the lens core (r_peak < 0.5 PSF FWHM) or whose area is below
   ``ANCHOR_MIN_COMPONENT_AREA_PIX``.  Duplicated peaks are merged.  No
   self-reflection test.  Every decision is recorded.
3. Source: every component (or a configured merge group) is a candidate.  Each
   is grown by hysteresis with an empty lens mask (growth that would create a
   self-reflection overlap is rejected), classified against the lens
   aperture with r_s, then carved:
     * pixels on, or reflecting onto, any other component (its uncarved
       4-sigma segmentation) or a modeled satellite;
     * pixels whose reflection lands on the candidate itself (both members of
       every self-overlapping pair).
   Reflection overlap uses continuous unit cells, as the DS9 background region
   does.  The largest 4-connected piece is kept only if it retains at least
   ``MIN_CARVE_SNR_RETENTION`` of the pre-carving integrated S/N.  Hard gates
   are applied per candidate; the passing candidate with the largest
   effective S/N wins, where the lens-light model inside the region enters as
   a systematic term, F_diff / sqrt(sigma_stat^2 + (eps * |M_lens|)^2) with
   eps = ``EFFECTIVE_SNR_EPSILON``.  This prefers images that are not buried
   under lens light, whose point-reflection residual would dominate the SED.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
    label,
)
from skimage.draw import line

ANCHOR_DETECT_SIGMA = 4.0
SOURCE_GROW_SIGMA_RATIO = 0.5
ANCHOR_MIN_COMPONENT_AREA_PIX = 20
ANCHOR_CORE_RADIUS_FWHM = 0.5
LENS_CONTACT_DISTANCE_FWHM = 0.5
MIN_SOURCE_AREA_PIX = 20
MIN_SOURCE_INTEGRATED_SNR = 15.0
MIN_CARVE_SNR_RETENTION = 0.70
RS_THRESHOLD = 0.4
EFFECTIVE_SNR_EPSILON = 0.015
DIAGNOSTIC_SIGMAS = (3.75, 4.25)
MAX_LENS_SYMMETRISATION_ITERATIONS = 100
FOUR_CONNECTED = generate_binary_structure(2, 1)


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------

def integrated_snr(mask: np.ndarray, signal: np.ndarray) -> float:
    count = int(np.sum(mask))
    return float(np.sum(signal[mask]) / np.sqrt(count)) if count else float("-inf")


def reflected_cell_hits(
    mask: np.ndarray, blocker: np.ndarray, centre_xy: tuple[float, float]
) -> np.ndarray:
    """Pixels of ``mask`` whose reflected unit cell overlaps a ``blocker`` cell.

    Two unit pixels overlap with positive area when both centre separations
    are below one pixel, so the reflected point can touch at most the two
    floor/ceil neighbours along each axis.
    """
    hits = np.zeros_like(mask, dtype=bool)
    if not mask.any() or not blocker.any():
        return hits
    yy, xx = np.nonzero(mask)
    xr = 2.0 * centre_xy[0] - xx
    yr = 2.0 * centre_xy[1] - yy
    eps = 1.0e-10
    found = np.zeros(len(yy), dtype=bool)
    for ix in (np.floor(xr), np.ceil(xr)):
        for iy in (np.floor(yr), np.ceil(yr)):
            close = (np.abs(xr - ix) < 1.0 - eps) & (np.abs(yr - iy) < 1.0 - eps)
            jx, jy = ix.astype(int), iy.astype(int)
            inside = (
                close & (jx >= 0) & (jx < blocker.shape[1])
                & (jy >= 0) & (jy < blocker.shape[0])
            )
            found[inside] |= blocker[jy[inside], jx[inside]]
    hits[yy[found], xx[found]] = True
    return hits


def largest_piece_4(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, count = label(mask, structure=FOUR_CONNECTED)
    if count == 0:
        return np.zeros_like(mask, dtype=bool), 0
    sizes = np.bincount(labels.ravel())[1:]
    return labels == (1 + int(np.argmax(sizes))), int(count)


def hole_mask(mask: np.ndarray) -> np.ndarray:
    return binary_fill_holes(mask) & ~mask


def bbox(mask: np.ndarray, product) -> dict:
    yy, xx = np.nonzero(mask)
    return {
        "crop_xmin": int(xx.min()), "crop_xmax": int(xx.max()),
        "crop_ymin": int(yy.min()), "crop_ymax": int(yy.max()),
        "full_xmin": int(xx.min() + product.x0), "full_xmax": int(xx.max() + product.x0),
        "full_ymin": int(yy.min() + product.y0), "full_ymax": int(yy.max() + product.y0),
    }


def psf_fwhm_pix(psf_path: Path, psf_pixel_scale: float, product_pixel_scale: float) -> float:
    """Second-moment Gaussian-equivalent FWHM, in product pixels."""
    psf = np.asarray(fits.getdata(psf_path), dtype=float)
    psf = np.clip(np.nan_to_num(psf), 0.0, None)
    psf /= psf.sum()
    yy, xx = np.indices(psf.shape)
    cy, cx = float((psf * yy).sum()), float((psf * xx).sum())
    variance = float((psf * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum()) / 2.0
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.sqrt(variance) * psf_pixel_scale / product_pixel_scale


def science_crop(config: dict, system_root: Path, product) -> np.ndarray:
    """Original science image on the product crop (used only by r_s)."""
    path = Path(config["science_cutout"])
    if not path.is_absolute():
        path = (system_root / path).resolve()
    with fits.open(path) as hdul:
        data = np.asarray(hdul[int(config.get("science_hdu", 0))].data, dtype=float)
    crop = data[product.y0:product.y0 + product.data.shape[0],
                product.x0:product.x0 + product.data.shape[1]]
    if crop.shape != product.data.shape:
        raise ValueError("science/product grid mismatch")
    return crop


# ---------------------------------------------------------------------------
# Segmentation (computed once per threshold, shared by every consumer)
# ---------------------------------------------------------------------------

class SegmentationCache:
    def __init__(self, module, source, base_settings: dict, sextractor: Path, work_root: Path):
        self.module = module
        self.source = source
        self.base_settings = base_settings
        self.sextractor = sextractor
        self.work_root = work_root
        self._cache: dict[float, list[np.ndarray]] = {}

    def settings(self, sigma: float) -> dict:
        settings = deepcopy(self.base_settings)
        settings["detect_snr_floor"] = float(sigma)
        settings["detect_peak_fraction"] = 0.0
        return settings

    def components(self, sigma: float) -> list[np.ndarray]:
        key = round(float(sigma), 4)
        if key not in self._cache:
            segmentation = self.module.run_sextractor(
                self.source, self.work_root / f"sigma_{key:.2f}",
                self.sextractor, self.settings(key),
            )
            self._cache[key] = self.module._split_segmentation(np.asarray(segmentation))
        return self._cache[key]


def label_image(components: list[np.ndarray], shape) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int16)
    for index, mask in enumerate(components, start=1):
        labels[mask] = index
    return labels


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------

def select_anchors(source, components, fwhm_pix: float, einstein_radius_pix: float):
    """Return (anchor list, per-component records)."""
    signal = (source.data - source.background) / source.sigma
    cx, cy = source.centre_crop_xy
    row_c = int(np.clip(round(cy), 0, source.data.shape[0] - 1))
    col_c = int(np.clip(round(cx), 0, source.data.shape[1] - 1))
    core_radius = ANCHOR_CORE_RADIUS_FWHM * fwhm_pix
    anchors: list[tuple[int, int]] = []
    owner: dict[tuple[int, int], int] = {}
    records = []
    for index, mask in enumerate(components, start=1):
        usable = mask & source.valid
        row, col = np.unravel_index(np.argmax(np.where(usable, signal, -np.inf)), signal.shape)
        row, col = int(row), int(col)
        r_peak = float(np.hypot(col - cx, row - cy))
        record = {
            "component_index_1based": index,
            "area_pix": int(mask.sum()),
            "bbox": bbox(mask, source),
            "peak_crop_x": col, "peak_crop_y": row,
            "peak_full_x": col + source.x0, "peak_full_y": row + source.y0,
            "peak_residual_snr": float(signal[row, col]),
            "component_integrated_snr": integrated_snr(mask, signal),
            "peak_radius_pix": r_peak,
            "peak_radius_over_einstein_radius": r_peak / einstein_radius_pix,
            "contains_lens_centre_pixel": bool(mask[row_c, col_c]),
        }
        if r_peak < core_radius:
            record["fate"] = "skipped"
            record["reason"] = (
                f"peak lies in the lens core: r_peak={r_peak:.2f} px < "
                f"{ANCHOR_CORE_RADIUS_FWHM} FWHM = {core_radius:.2f} px"
            )
        elif (row, col) in owner:
            record["fate"] = "merged"
            record["reason"] = (
                f"peak pixel identical to the anchor of component {owner[(row, col)]}"
            )
        elif record["area_pix"] < ANCHOR_MIN_COMPONENT_AREA_PIX:
            record["fate"] = "rejected"
            record["reason"] = (
                f"component area {record['area_pix']} px < "
                f"{ANCHOR_MIN_COMPONENT_AREA_PIX} px (lens-subtraction residual)"
            )
        else:
            record["fate"] = "anchor"
            record["reason"] = "protected source peak"
            anchors.append((row, col))
            owner[(row, col)] = index
        records.append(record)
    return anchors, records


# ---------------------------------------------------------------------------
# Source candidates
# ---------------------------------------------------------------------------

def grow(seed: np.ndarray, signal: np.ndarray, valid: np.ndarray, level: float,
         centre_xy: tuple[float, float] | None = None,
         ) -> tuple[np.ndarray, bool]:
    """Hysteresis growth with an empty lens mask.

    The seed grows into the whole faint connected component it touches,
    without an area limit. Growth is rejected only if it creates a new
    self-reflection overlap. Contaminating components are then removed by
    carving, and the S/N-retention gate decides whether the remainder is usable.
    """
    faint, _ = label((signal >= level) & valid)
    touched = np.unique(faint[seed])
    touched = touched[touched > 0]
    if len(touched) == 0:
        grown = seed.copy()
    else:
        grown = np.isin(faint, touched) | seed
    kept_seed = binary_fill_holes(seed) & valid
    grown = binary_fill_holes(grown) & valid
    if centre_xy is not None and reflected_cell_hits(grown, grown, centre_xy).any() \
            and not reflected_cell_hits(kept_seed, kept_seed, centre_xy).any():
        return kept_seed, False
    return grown, True


def bridge_union(members: list[np.ndarray]) -> np.ndarray:
    """Union of components joined by shortest pixel lines (merge groups)."""
    merged = members[0].copy()
    for component in members[1:]:
        distance, nearest = distance_transform_edt(~merged, return_indices=True)
        yy, xx = np.nonzero(component)
        k = int(np.argmin(distance[yy, xx]))
        ty, tx = int(yy[k]), int(xx[k])
        sy, sx = int(nearest[0, ty, tx]), int(nearest[1, ty, tx])
        by, bx = line(sy, sx, ty, tx)
        merged[by, bx] = True
        merged |= component
    return merged


def rs_classification(raw_crop: np.ndarray, lens_product, source) -> dict:
    """Point-reflection source fraction r_s on every pixel."""
    cx, cy = source.centre_crop_xy
    from scipy.ndimage import map_coordinates

    yy, xx = np.indices(raw_crop.shape, dtype=float)
    reflected = map_coordinates(
        raw_crop, [2 * cy - yy, 2 * cx - xx],
        order=1, mode="constant", cval=np.nan, prefilter=False,
    )
    difference = raw_crop - reflected
    denominator = lens_product.data + difference
    ratio = np.full(raw_crop.shape, np.nan)
    ok = np.isfinite(difference) & np.isfinite(denominator) & (denominator != 0)
    ratio[ok] = difference[ok] / denominator[ok]
    floor = 3.0 * np.sqrt(2.0) * float(source.sigma)
    reliable = ok & (difference >= 0) & (denominator > floor) & (ratio >= 0) & (ratio <= 1)
    return {"ratio": ratio, "reliable": reliable, "valid_ratio": ok}


def build_candidate(name, members, seed, components, grown_by_component, signal,
                    source, lens_product, lens_mask, satellite, rs, centre_xy) -> dict:
    """Grow/r_s/carve/gate one candidate.  ``members`` are 0-based indices."""
    level = max(1.5, SOURCE_GROW_SIGMA_RATIO * ANCHOR_DETECT_SIGMA)
    grown, did_grow = grow(seed, signal, source.valid, level, centre_xy)
    # Blockers: every other component's uncarved 4-sigma segmentation.  Their
    # grown 2-sigma wings are deliberately not blockers (use the original component footprint):
    # for counter-images they removed most of the brighter image while the
    # measured wing contamination of the background is ~2% of the source flux.
    del grown_by_component
    others = np.zeros_like(seed)
    for index, component in enumerate(components):
        if index not in members:
            others |= component
    overlap = grown & lens_mask & source.valid & lens_product.valid
    accepted = overlap & rs["reliable"] & (rs["ratio"] >= RS_THRESHOLD)
    after_rs = grown & (~(grown & lens_mask) | accepted)
    record = {
        "candidate": name,
        "members_1based": [m + 1 for m in members],
        "seed_area_pix": int(seed.sum()),
        "grown": bool(did_grow),
        "grown_area_pix": int(grown.sum()),
        "lens_overlap_pix": int(overlap.sum()),
        "rs_reliable_overlap_pix": int(np.sum(overlap & rs["reliable"])),
        "rs_accepted_overlap_pix": int(accepted.sum()),
        "area_before_carving_pix": int(after_rs.sum()),
        "integrated_snr_before_carving": integrated_snr(after_rs, signal),
    }
    blocker = others | satellite
    carved = after_rs & ~blocker
    record["carved_direct_pix"] = int(after_rs.sum() - carved.sum())
    mirror_hits = reflected_cell_hits(carved, blocker, centre_xy)
    record["carved_mirror_on_others_or_satellite_pix"] = int(mirror_hits.sum())
    carved &= ~mirror_hits
    self_hits = reflected_cell_hits(carved, carved, centre_xy)
    record["carved_self_overlap_pix"] = int(self_hits.sum())
    carved &= ~self_hits
    piece, pieces = largest_piece_4(carved)
    record["pieces_after_carving"] = pieces
    record["area_after_carving_pix"] = int(piece.sum())
    record["integrated_snr_after_carving"] = integrated_snr(piece, signal)
    before = record["integrated_snr_before_carving"]
    record["snr_retention"] = (
        record["integrated_snr_after_carving"] / before if before > 0 else 0.0
    )
    holes = hole_mask(piece) if piece.any() else piece
    record["hole_pixel_count"] = int(holes.sum())
    reasons = []
    if not piece.any():
        reasons.append("nothing left after carving")
    else:
        if record["snr_retention"] < MIN_CARVE_SNR_RETENTION:
            reasons.append(
                f"carving kept {record['snr_retention']:.2f} of the integrated S/N "
                f"(< {MIN_CARVE_SNR_RETENTION})"
            )
        if record["area_after_carving_pix"] < MIN_SOURCE_AREA_PIX:
            reasons.append(
                f"area {record['area_after_carving_pix']} px < {MIN_SOURCE_AREA_PIX} px "
                f"after carving (before: {record['area_before_carving_pix']} px)"
            )
        if record["integrated_snr_after_carving"] < MIN_SOURCE_INTEGRATED_SNR:
            reasons.append(
                f"integrated S/N {record['integrated_snr_after_carving']:.1f} < "
                f"{MIN_SOURCE_INTEGRATED_SNR}"
            )
        cx, cy = centre_xy
        if piece[int(np.clip(round(cy), 0, piece.shape[0] - 1)),
                 int(np.clip(round(cx), 0, piece.shape[1] - 1))]:
            reasons.append("contains the lens centre")
        # Assertions guaranteed by construction.
        assert not (piece & blocker).any()
        assert not reflected_cell_hits(piece, blocker, centre_xy).any()
        assert not reflected_cell_hits(piece, piece, centre_xy).any()
    record["passes_hard_gates"] = not reasons
    record["fate"] = "passed" if not reasons else "discarded"
    record["reasons"] = reasons
    return {"record": record, "mask": piece, "accepted": accepted & piece,
            "holes": holes, "grown": grown}


class SourceSelectionError(RuntimeError):
    """No candidate passed; carries every candidate record for the report."""

    def __init__(self, message: str, candidates: list[dict]):
        super().__init__(message)
        self.candidates = candidates


def select_source(module, source, lens_product, raw_crop, components, anchor_records,
                  lens_mask, satellite, merge_group_1based=None) -> dict:
    """Single source selection; returns masks and full records."""
    centre_xy = source.centre_crop_xy
    signal = (source.data - source.background) / source.sigma
    rs = rs_classification(raw_crop, lens_product, source)
    level = max(1.5, SOURCE_GROW_SIGMA_RATIO * ANCHOR_DETECT_SIGMA)
    grown_by_component = None

    group = [int(i) - 1 for i in (merge_group_1based or [])]
    if group:
        if any(i < 0 or i >= len(components) for i in group):
            raise ValueError(f"invalid merge group {merge_group_1based} for {len(components)} components")
        not_anchor = [i + 1 for i in group if anchor_records[i]["fate"] != "anchor"]
        if not_anchor:
            raise RuntimeError(f"merge-group components {not_anchor} are not anchor components")
    plan = []
    if group:
        plan.append(("merge_group_" + "+".join(str(i + 1) for i in group), group,
                     bridge_union([components[i] for i in group])))
    for index, component in enumerate(components):
        if index not in group:
            plan.append((f"component_{index + 1}", [index], component))

    built = [
        build_candidate(name, members, seed, components, grown_by_component, signal,
                        source, lens_product, lens_mask, satellite, rs, centre_xy)
        for name, members, seed in plan
    ]
    # Ranking metric: effective S/N with the lens-light leakage term.  Region
    # statistics come from the same routine that measures the final source.
    for b in built:
        if not b["mask"].any():
            continue
        measured: dict = {}
        module.refresh_selected_measurements(
            measured, b["mask"], source, lens_product, lens_mask & ~b["mask"],
            EFFECTIVE_SNR_EPSILON,
        )
        for key in ("effective_snr", "difference_flux", "statistical_flux_error",
                    "lens_model_flux", "region_flux", "lens_model_flux_fraction",
                    "mirror_asymmetry"):
            b["record"][key] = measured[key]
        b["record"]["effective_snr_epsilon"] = EFFECTIVE_SNR_EPSILON
    passing = [b for b in built if b["record"]["passes_hard_gates"]]
    if not passing:
        raise SourceSelectionError(
            "No source candidate passed the hard gates: "
            + "; ".join(f"{b['record']['candidate']}: {b['record']['reasons']}" for b in built),
            [b["record"] for b in built],
        )
    winner = max(passing, key=lambda b: b["record"]["effective_snr"])
    for b in built:
        if b is winner:
            b["record"]["fate"] = "selected"
        elif b["record"]["passes_hard_gates"]:
            b["record"]["fate"] = "passed_not_selected"

    # Final lens: remove the winner's accepted r_s pixels and their reflections,
    # then restore exact (discrete) point symmetry by intersection.
    accepted = winner["accepted"]
    cut = module.symmetric_closure(accepted, centre_xy, source.valid) if accepted.any() else accepted
    lens_final = lens_mask & ~cut
    converged = False
    for iteration in range(MAX_LENS_SYMMETRISATION_ITERATIONS):
        reflected = module._reflect_mask(lens_final, centre_xy) & source.valid
        updated = lens_final & reflected
        if np.array_equal(updated, lens_final):
            converged = True
            break
        lens_final = updated
    if not converged:
        raise RuntimeError("final lens aperture did not converge to point symmetry")
    source_mask = winner["mask"] & ~lens_final
    if not np.array_equal(source_mask, winner["mask"]):
        raise RuntimeError("selected source overlaps the final lens aperture")

    secondary = np.zeros_like(source_mask)
    winner_members = set(winner["record"]["members_1based"])
    for index, component in enumerate(components):
        if index + 1 not in winner_members:
            secondary |= component
    return {
        "source_mask": source_mask,
        "lens_final": lens_final,
        "accepted": accepted,
        "holes": winner["holes"],
        "secondary": secondary,
        "winner": winner["record"],
        "winner_members_1based": sorted(winner_members),
        "candidates": [b["record"] for b in built],
        "candidate_masks": {b["record"]["candidate"]: b["mask"] for b in built},
        "rs_ratio": rs["ratio"],
        "lens_symmetrisation_iterations": iteration + 1,
    }


def neighbour_flux_diagnostic(module, cache, source, lens_product, raw_crop, lens_mask,
                              satellite, winner_members_1based, components_4sigma) -> dict:
    """Diagnostic only: rerun the winner at the neighbouring thresholds."""
    signal = (source.data - source.background) / source.sigma
    rs = rs_classification(raw_crop, lens_product, source)
    winner_seed = np.zeros_like(source.valid)
    for i in winner_members_1based:
        winner_seed |= components_4sigma[i - 1]

    def flux(mask):
        return float(np.sum(source.data[mask] - source.background))

    out = {}
    for sigma in DIAGNOSTIC_SIGMAS:
        comps = cache.components(sigma)
        if not comps:
            out[f"{sigma:.2f}"] = {"error": "no component"}
            continue
        overlaps = [int(np.sum(c & winner_seed)) for c in comps]
        members = [i for i, o in enumerate(overlaps) if o > 0] or [int(np.argmax(overlaps))]
        level = max(1.5, SOURCE_GROW_SIGMA_RATIO * sigma)
        grown_by = None
        seed = bridge_union([comps[i] for i in members]) if len(members) > 1 else comps[members[0]]
        built = build_candidate("diagnostic", members, seed, comps, grown_by, signal, source,
                                lens_product, lens_mask, satellite, rs, source.centre_crop_xy)
        out[f"{sigma:.2f}"] = {
            "area_pix": built["record"]["area_after_carving_pix"],
            "region_flux": flux(built["mask"]),
            "integrated_snr": built["record"]["integrated_snr_after_carving"],
        }
    return out


# ---------------------------------------------------------------------------
# Exact pixel-cell region files
# ---------------------------------------------------------------------------
# The legacy writer traces the 0.5 iso-contour and simplifies it with a 0.35 px
# tolerance, so a written polygon differs from the union of its pixel cells.
# Two masks whose cells only touch (a source and its own reflection after
# carving) then overlap by a sub-pixel area once written, and photometry
# rejects every exposure.  The writer below emits the exact cell outline, so
# the continuous reflection of the source polygon is exactly the reflected
# cells and the cell-overlap guarantees used in carving carry over.

def cell_outline_loops(mask: np.ndarray) -> list[np.ndarray]:
    """Counter-clockwise outer loops (x, y in crop pixel units) of pixel cells.

    Hole loops (clockwise) are not returned; holes are reported separately.
    Diagonal pinch points are resolved by turning left, which keeps cells that
    only touch at a corner in separate loops.
    """
    mask = np.asarray(mask, dtype=bool)
    rows, cols = mask.shape
    padded = np.pad(mask, 1)
    edges = {}
    for r, c in zip(*np.nonzero(mask)):
        pr, pc = r + 1, c + 1
        x0, x1, y0, y1 = c - 0.5, c + 0.5, r - 0.5, r + 0.5
        if not padded[pr - 1, pc]:
            edges.setdefault((x0, y0), []).append((x1, y0))
        if not padded[pr, pc + 1]:
            edges.setdefault((x1, y0), []).append((x1, y1))
        if not padded[pr + 1, pc]:
            edges.setdefault((x1, y1), []).append((x0, y1))
        if not padded[pr, pc - 1]:
            edges.setdefault((x0, y1), []).append((x0, y0))
    loops = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        current, previous = start, None
        while True:
            options = edges[current]
            if len(options) == 1 or previous is None:
                nxt = options[0]
            else:
                dx, dy = current[0] - previous[0], current[1] - previous[1]
                # Prefer a left turn, then straight, then right.
                left = (current[0] - dy, current[1] + dx)
                straight = (current[0] + dx, current[1] + dy)
                nxt = left if left in options else (straight if straight in options else options[0])
            options.remove(nxt)
            if not options:
                del edges[current]
            previous, current = current, nxt
            if current == start and (start not in edges):
                break
            loop.append(current)
            if current == start:
                break
        xy = np.asarray(loop, dtype=float)
        # Drop collinear vertices.
        keep = []
        for i in range(len(xy)):
            a, b, c = xy[i - 1], xy[i], xy[(i + 1) % len(xy)]
            if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-12:
                keep.append(b)
        xy = np.asarray(keep)
        area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
        if area > 0:
            loops.append(xy)
    return loops


def _ds9_polygon(product, xy_full: np.ndarray, label: str, color: str) -> str:
    sky = product.wcs.celestial.pixel_to_world(xy_full[:, 0], xy_full[:, 1]).icrs
    coordinates = ",".join(
        f"{value:.10f}" for pair in zip(sky.ra.deg, sky.dec.deg) for value in pair
    )
    return f"polygon({coordinates}) # color={color} text={{{label}}}\n"


DS9_HEADER = (
    "# Region file format: DS9 version 4.1\n"
    "global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" "
    "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
    "icrs\n"
)


def write_exact_cell_region(path: Path, mask: np.ndarray, product, label: str,
                            color: str = "green") -> None:
    lines = [DS9_HEADER]
    for xy in cell_outline_loops(mask):
        lines.append(_ds9_polygon(product, xy + [product.x0, product.y0], label, color))
    if len(lines) == 1:
        raise RuntimeError(f"Cannot write an empty region: {path}")
    Path(path).write_text("".join(lines))


def write_exact_reflected_region(path: Path, mask: np.ndarray, product,
                                 centre_full_xy: tuple[float, float], label: str,
                                 color: str = "cyan") -> None:
    """Continuous point reflection of the exact cell outline of ``mask``."""
    lines = [DS9_HEADER]
    for xy in cell_outline_loops(mask):
        full = xy + [product.x0, product.y0]
        reflected = np.column_stack((2.0 * centre_full_xy[0] - full[:, 0],
                                     2.0 * centre_full_xy[1] - full[:, 1]))
        lines.append(_ds9_polygon(product, reflected, label, color))
    if len(lines) == 1:
        raise RuntimeError(f"Cannot write an empty region: {path}")
    Path(path).write_text("".join(lines))
