from __future__ import annotations

from pathlib import Path
import autolens as al

from .config import load_config
from .data import load_dataset
from .products import write_consistent_fit_products
from .saved_fit import stage_tracer, maximum_likelihood_fit


def export_mge_component_cube(
    config_path: Path, products_dir: Path, stage: int | None = None
) -> Path:
    """Export MGE components using the same grid and PSF as the saved fit."""
    config = load_config(config_path)
    if stage is None:
        stage = 4 if (config.output_dir / "stage4_run_summary.json").is_file() else 3
    bundle = load_dataset(config)
    tracer_path = stage_tracer(config.output_dir, stage)
    tracer = al.from_json(file_path=tracer_path)
    fit = maximum_likelihood_fit(config, bundle, tracer, stage)
    products = write_consistent_fit_products(
        fit, tracer, bundle, config, stage=stage,
        output_dir=products_dir.resolve(), tracer_path=tracer_path)
    return products["mge_components"]
