#!/usr/bin/env python
"""Run any configured three-stage model.

JAX device initialization remains available for compatible models, while each
JSON config decides whether AutoLens evaluates the likelihood with JAX or with
the NumPy/FNNLS backend.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "J022511-045433" / "config.json",
        help="Path to a system_<id>/config.json file (default: J022511-045433).",
    )
    parser.add_argument(
        "--jax-platform", choices=("gpu", "cpu", "auto"), default="gpu"
    )
    return parser.parse_args()


def configure_jax(platform: str) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".runtime/matplotlib"))
    if platform == "gpu":
        # JAX's concrete NVIDIA backend is named ``cuda``. Using the generic
        # word ``gpu`` also probes an installed ROCm registration first in this
        # environment, which is both unnecessary and incompatible.
        os.environ["JAX_PLATFORMS"] = "cuda"
    elif platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"


def main():
    args = parse_args()
    configure_jax(args.jax_platform)
    import jax

    from lens_modeling.config import load_config
    from lens_modeling.pipeline import run

    print("JAX devices:", jax.devices())
    if args.jax_platform == "gpu" and not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("--jax-platform gpu was requested but no JAX GPU is visible")
    config = load_config(args.config)
    results, products = run(config)
    print("Completed stages:", len(results))
    print("WCS products:", {key: str(value) for key, value in products.items()})


if __name__ == "__main__":
    main()
