#!/usr/bin/env python
"""Run only stage 4 from a completed on-disk stage-3 posterior."""

from __future__ import annotations

import argparse
from pathlib import Path

from lens_modeling.config import load_config
from lens_modeling.pipeline import run_stage4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed-mesh pixel-source stage without rerunning stages 1-3."
    )
    parser.add_argument("config", type=Path, help="System config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    result, summary_path = run_stage4(config)
    print("Stage 4 complete")
    print("log evidence:", float(result.samples.log_evidence))
    print(
        "reduced chi-squared:",
        float(result.max_log_likelihood_fit.reduced_chi_squared),
    )
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
