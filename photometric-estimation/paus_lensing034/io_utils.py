from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from .config import PipelineConfig


def read_2d_celestial_image(path: Path) -> tuple[np.ndarray, WCS, int]:
    """Read the first two-dimensional FITS HDU with a valid celestial WCS."""
    with fits.open(path, memmap=False) as hdul:
        for index, hdu in enumerate(hdul):
            if hdu.data is None or np.asarray(hdu.data).ndim != 2:
                continue
            celestial = WCS(hdu.header).celestial
            if (
                celestial.has_celestial
                and celestial.pixel_n_dim == 2
                and celestial.world_n_dim == 2
            ):
                return np.asarray(hdu.data, dtype=float), celestial, index
    raise ValueError(f"No 2-D celestial image HDU found: {path}")


def ordered_nb_files(load_dir: Path) -> list[Path]:
    available = {
        path.name
        for path in load_dir.glob("*.fits")
        if not path.name.endswith(".weight.fits") and "_weight_" not in path.name
    }
    image_list = load_dir / "img_list.txt"
    if image_list.exists():
        listed = [
            Path(line.strip()).name
            for line in image_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [load_dir / name for name in listed if name in available]
    return [load_dir / name for name in sorted(available)]


def one_bb_file(system_dir: Path, band: str) -> Path:
    if band.startswith("subaru_"):
        filter_name = f"HSC-{band.removeprefix('subaru_').upper()}"
    elif band.startswith("cfht_"):
        filter_name = f"CFHT-{band.removeprefix('cfht_').upper()}"
    else:
        raise ValueError(f"Unknown BB band naming convention: {band}")
    path = system_dir / "science" / f"{filter_name}.fits"
    if not path.is_file():
        raise FileNotFoundError(f"Missing BB science cutout for {band!r}: {path}")
    return path


def load_array(path: Path, count: int, label: str) -> np.ndarray | None:
    if not path.exists():
        return None
    values = np.atleast_1d(np.loadtxt(path)).astype(float)
    if values.size == 1:
        return np.repeat(values, count)
    if values.size != count:
        raise ValueError(f"{label} has {values.size} values for {count} images: {path}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_input_manifest(config: PipelineConfig) -> Path:
    roles = {
        "catalogue": config.catalog_path,
        "source_region": config.source_region_path,
        "background_region": config.background_region_path,
        "lens_region": config.lens_region_path,
    }
    if config.nb_aperture_centre_path is not None:
        roles["nb_aperture_centre_table"] = config.nb_aperture_centre_path
    rows = []
    for role, path in roles.items():
        stat = path.stat()
        rows.append({
            "role": role,
            "relative_path": config.relative_to_project(path),
            "sha256": sha256(path),
            "mtime": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat(),
            "size_bytes": stat.st_size,
        })
    output = config.result_dir / "input_manifest.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
