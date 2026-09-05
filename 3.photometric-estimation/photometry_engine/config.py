from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration shared by the two thin Photo-lens notebooks."""

    lens_id: str
    field: str = "W1"
    aperture_subpixels: int = 81
    convergence_subpixels: int = 161
    coadd_error_factor: float = 1.5
    hsc_zero_point: float = 27.0
    cfht_zero_point: float = 30.0
    paus_zero_point: float = 26.0
    calibrate_bcnz: bool = False
    nb_weights: tuple[float, ...] = (1.0,)
    official_nb_weight: float = 1.0
    include_hsc_g: bool = False

    catalog_rel: Path = Path("data/catalogue/targets.csv")
    nb_cutout_rel: Path = Path("data/nb/single_epoch")
    nb_coadd_rel: Path = Path("data/nb/coadd")
    bb_cutout_rel: Path = Path("data/bb")
    source_region_rel: Path = Path("2.region-detection/regions")
    background_region_rel: Path = Path("2.region-detection/regions")
    lens_region_rel: Path = Path("2.region-detection/regions")
    model_rel: Path = Path("3.photometric-estimation/model")
    hsc_g_model_rel: Path = Path("3.photometric-estimation/hsc-g-response/model-g")
    output_rel: Path = Path("3.photometric-estimation/results")
    source_region_override: Path | None = None
    lens_region_override: Path | None = None
    background_region_override: Path | None = None
    nb_aperture_centre_table: Path | None = None

    @property
    def excluded_bands(self) -> tuple[str, ...]:
        import json
        stable = json.loads((self.project_root / "config/photometry.json").read_text())
        return tuple(stable.get("system_band_exclusions", {}).get(str(self.lens_id), []))

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def output_root(self) -> Path:
        return self.project_root / self.output_rel

    @property
    def result_dir(self) -> Path:
        return self.output_root / str(self.lens_id)

    @property
    def catalog_path(self) -> Path:
        return self.project_root / self.catalog_rel

    @property
    def nb_cutout_dir(self) -> Path:
        return self.project_root / self.nb_cutout_rel

    @property
    def nb_coadd_dir(self) -> Path:
        return (self.project_root / self.nb_coadd_rel).resolve()

    @property
    def bb_cutout_dir(self) -> Path:
        return self.project_root / self.bb_cutout_rel

    @property
    def source_region_path(self) -> Path:
        if self.source_region_override is not None:
            return self.source_region_override.resolve()
        return (
            self.project_root / self.source_region_rel / f"{self.lens_id}"
            / f"{self.lens_id}_source_point_fraction040.reg"
        )

    @property
    def background_region_path(self) -> Path:
        if self.background_region_override is not None:
            return self.background_region_override.resolve()
        return (
            self.project_root / self.background_region_rel / f"{self.lens_id}"
            / f"{self.lens_id}_background_point_fraction040.reg"
        )

    @property
    def lens_region_path(self) -> Path:
        if self.lens_region_override is not None:
            return self.lens_region_override.resolve()
        return (
            self.project_root / self.lens_region_rel / f"{self.lens_id}"
            / f"{self.lens_id}_lens_point_fraction040.reg"
        )

    @property
    def bb_system_dir(self) -> Path:
        return self.bb_cutout_dir / str(self.lens_id)

    @property
    def bb_survey(self) -> str:
        science = self.bb_system_dir / "science"
        if any(science.glob("HSC-*.fits")):
            return "HSC"
        if any(science.glob("CFHT-*.fits")):
            return "CFHT"
        raise FileNotFoundError(f"No HSC/CFHT science files in {science}")

    @property
    def bb_zero_point(self) -> float:
        return self.hsc_zero_point if self.bb_survey == "HSC" else self.cfht_zero_point

    @property
    def nb_aperture_centre_path(self) -> Path | None:
        if self.nb_aperture_centre_table is None:
            return None
        return self.nb_aperture_centre_table.resolve()

    @property
    def model_dir(self) -> Path:
        return self.project_root / self.model_rel

    @property
    def hsc_g_model_dir(self) -> Path:
        return self.project_root / self.hsc_g_model_rel

    @property
    def nb_names(self) -> list[str]:
        return [f"NB{x}" for x in np.arange(455, 846, 10)]

    @property
    def bb_names(self) -> list[str]:
        if self.bb_survey == "HSC":
            available = {p.stem.removeprefix("HSC-").lower() for p in (self.bb_system_dir / "science").glob("HSC-*.fits")}
            supported = ("g", "r", "i", "z", "y") if self.include_hsc_g else ("r", "i", "z", "y")
            return [f"subaru_{band}" for band in supported if band in available]
        available = {p.stem.removeprefix("CFHT-").lower() for p in (self.bb_system_dir / "science").glob("CFHT-*.fits")}
        return [f"cfht_{band}" for band in ("u", "g", "r", "i", "z") if band in available]

    @property
    def ordered_bands(self) -> list[str]:
        return self.nb_names + self.bb_names

    def relative_to_project(self, path: Path) -> str:
        return Path(os.path.relpath(path.resolve(), self.project_root.resolve())).as_posix()

    def validate(self) -> None:
        required = {
            "catalogue": self.catalog_path,
            "NB cutout root": self.nb_cutout_dir,
            f"{self.bb_survey} system directory": self.bb_system_dir,
            "source region": self.source_region_path,
            "background region": self.background_region_path,
            "lens region": self.lens_region_path,
            "BCNz model directory": self.model_dir,
        }
        if self.include_hsc_g:
            required["HSC-G BCNz model directory"] = self.hsc_g_model_dir
        missing = [f"{role}: {path}" for role, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing Photo-lens input(s):\n" + "\n".join(missing))
        if self.aperture_subpixels < 1 or self.convergence_subpixels < 1:
            raise ValueError("Subpixel counts must be positive integers")
        if self.official_nb_weight not in self.nb_weights:
            raise ValueError("official_nb_weight must be included in nb_weights")
        if (
            self.nb_aperture_centre_path is not None
            and not self.nb_aperture_centre_path.is_file()
        ):
            raise FileNotFoundError(
                "Missing NB aperture-centre table: "
                f"{self.nb_aperture_centre_path}"
            )
