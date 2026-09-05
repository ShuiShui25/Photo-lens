from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    system: str
    science_cutout: Path
    psf: Path
    positions_file: Path
    output_dir: Path
    z_lens: float
    z_source: float
    lens_ra_deg: float
    lens_dec_deg: float
    variance_cutout: Path | None = None
    weight_cutout: Path | None = None
    pixel_scale: float = 0.168
    science_hdu: int = 1
    variance_hdu: int = 2
    weight_hdu: int = 0
    light_centre_sigma: float = 0.2
    mass_centre_sigma: float = 0.2
    lens_gaussians_per_basis: int = 15
    lens_bases: int = 2
    source_gaussians: int = 10
    n_live: int = 75
    n_eff: int = 100
    n_batch: int = 5
    number_of_cores: int = 1
    psf_size: int = 31
    over_sample: bool = True
    # NumPy/FNNLS remains the portable fallback. JAX is valid only when the
    # AutoArray FFT crop fix documented in autoarray_jax_crop_patch is present.
    use_jax: bool = False
    positions_threshold: float = 0.5
    satellite_light_centre_sigma: float = 0.08
    einstein_radius_mean: float = 1.19
    stage4_mesh_shape: tuple[int, int] = (20, 20)
    stage4_use_jax: bool = True
    # ``constant`` reproduces the original stage 4.  ``constant_brightness_zeroth``
    # adds a signal-weighted zeroth-order term, which needs the stage-3 lensed
    # source model image as an adapt image.
    stage4_regularization: str = "constant"
    stage4_signal_scale: float = 1.0
    # Stage 4 may write elsewhere than stages 1-3, so a regularization variant can
    # be run without disturbing the validated baseline.  Stage 3 is still read
    # from ``output_dir``.
    stage4_output_dir: Path | None = None
    bb_survey: str = "HSC"
    bb_band: str = ""


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    for key in (
        "science_cutout", "variance_cutout", "weight_cutout", "psf", "positions_file",
        "output_dir", "stage4_output_dir",
    ):
        if raw.get(key) is None:
            continue
        value = Path(raw[key])
        raw[key] = (base / value).resolve() if not value.is_absolute() else value
    return RunConfig(**raw)
