#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from lens_modeling.mge_components import export_mge_component_cube


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 30 fitted lens MGE components")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--products-dir", required=True, type=Path)
    parser.add_argument("--stage", type=int, choices=(3, 4), default=None)
    args = parser.parse_args()
    print(export_mge_component_cube(args.config, args.products_dir, stage=args.stage))


if __name__ == "__main__":
    main()
