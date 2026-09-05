#!/usr/bin/env python3
"""Recompute stable photometry and photo-z without overwriting the snapshot."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'photometric-estimation'))
from paus_lensing034 import PipelineConfig, run_pipeline

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', type=int, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs'))
    args = parser.parse_args()
    if args.output.is_absolute() or '..' in args.output.parts or args.output.parts[0] != 'runs':
        parser.error('--output must be runs or a directory beneath runs/')
    cfg = PipelineConfig(lens_id=args.system, output_rel=args.output)
    cfg = PipelineConfig(lens_id=args.system, output_rel=args.output,
        include_hsc_g=cfg.bb_survey == 'HSC',
        nb_aperture_centre_table=ROOT / f'photometric-estimation/nb-centres/system_{args.system}/system{args.system}_nb_antisymmetry_centres.csv')
    result = run_pipeline(cfg)
    print(result.summary.to_string(index=False))

if __name__ == '__main__':
    main()
