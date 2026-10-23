#!/usr/bin/env python3
"""Export exact FitImaging plane products and independent satellite models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import autolens as al

SCRIPT_DIR = Path(__file__).resolve().parent
MODELING_ROOT = SCRIPT_DIR.parent / "1.lens-modeling"
sys.path.insert(0, str(MODELING_ROOT))

from lens_modeling.config import load_config  # noqa: E402
from lens_modeling.data import load_dataset  # noqa: E402
from lens_modeling.saved_fit import stage_tracer, maximum_likelihood_fit  # noqa: E402



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--products-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    bundle = load_dataset(config)
    tracer_path = stage_tracer(config.output_dir, args.stage)
    tracer = al.from_json(file_path=tracer_path)
    fit = maximum_likelihood_fit(config, bundle, tracer, args.stage)
    from lens_modeling.products import write_consistent_fit_products
    write_consistent_fit_products(fit, tracer, bundle, config, stage=args.stage,
                                  output_dir=args.products_dir.resolve(), tracer_path=tracer_path)
    print(args.products_dir.resolve())


if __name__ == "__main__":
    main()
