#!/usr/bin/env python3
"""Run unified PHOTO-LENS03 HSC/CFHT photometry and BCNz estimation."""
from __future__ import annotations

import argparse
from pathlib import Path

from paus_lensing034 import PipelineConfig, run_pipeline


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, type=int)
    parser.add_argument("--source-region", required=True, type=Path)
    parser.add_argument("--lens-region", required=True, type=Path)
    parser.add_argument("--background-region", required=True, type=Path)
    parser.add_argument("--nb-centres", required=True, type=Path)
    parser.add_argument(
        "--include-hsc-g", action="store_true",
        help="Use the separately generated official hscPipe HSC-G model response",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "photometric-estimation/results",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    try:
        output_rel = output_root.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("PHOTO-LENS03 outputs must remain inside PHOTO-LENS03") from exc

    config = PipelineConfig(
        lens_id=args.system,
        aperture_subpixels=81,
        convergence_subpixels=161,
        source_region_override=args.source_region,
        lens_region_override=args.lens_region,
        background_region_override=args.background_region,
        nb_aperture_centre_table=args.nb_centres,
        include_hsc_g=args.include_hsc_g,
        output_rel=output_rel,
    )
    result = run_pipeline(config)
    excluded = result.band_status.loc[~result.band_status.effective_common]
    print("Effective common bands:", len(result.common_bands), result.common_bands)
    print("Excluded bands:", len(excluded))
    if not excluded.empty:
        print(excluded[["band", "kind", "exclusion_reason"]].to_string(index=False))
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
