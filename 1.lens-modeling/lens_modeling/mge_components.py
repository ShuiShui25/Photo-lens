from __future__ import annotations

from pathlib import Path

import autolens as al

from .config import load_config
from .data import load_dataset
from .products import write_wcs_products_from_profiles


def _stage_tracer_path(output_dir: Path, stage: int) -> Path:
    matches = list(output_dir.glob(f"stage{stage}_*/**/files/tracer.json"))
    if not matches:
        raise FileNotFoundError(f"No stage-{stage} tracer.json under {output_dir}")
    if len(matches) != 1:
        raise RuntimeError(
            f"Ambiguous stage-{stage} tracer selection under {output_dir}: "
            + ", ".join(map(str, matches))
        )
    return matches[0]


def export_mge_component_cube(
    config_path: Path, products_dir: Path, stage: int | None = None
) -> Path:
    """Re-export the complete full-detector product set from an exact tracer."""
    config = load_config(config_path)
    products_dir = products_dir.resolve()
    if stage is None:
        stage = 4 if (config.output_dir / "stage4_run_summary.json").is_file() else 3
    bundle = load_dataset(config)
    tracer_path = _stage_tracer_path(config.output_dir, stage)
    tracer = al.from_json(file_path=tracer_path)
    main_lenses = [galaxy for galaxy in tracer.galaxies if hasattr(galaxy, "mass")]
    if len(main_lenses) != 1:
        raise ValueError(f"Expected one mass-bearing main lens, found {len(main_lenses)}")
    lens = main_lenses[0]
    output_name = str(products_dir.relative_to(config.output_dir))
    products = write_wcs_products_from_profiles(
        profiles=list(lens.bulge.profile_list),
        einstein_radius=float(lens.mass.einstein_radius),
        bundle=bundle,
        config=config,
        stage=stage,
        output_name=output_name,
        tracer_path=tracer_path,
    )
    return products["mge_components"]
