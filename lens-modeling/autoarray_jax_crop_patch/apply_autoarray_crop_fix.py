#!/usr/bin/env python
"""Apply the System-8 validated AutoArray JAX FFT crop fix idempotently."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


def default_target() -> Path:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise RuntimeError("CONDA_PREFIX is unset; activate or use conda run -n autolens-jax")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return (
        Path(prefix)
        / "lib"
        / version
        / "site-packages"
        / "autoarray"
        / "structures"
        / "arrays"
        / "kernel_2d.py"
    )


def block(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, help="Override kernel_2d.py path")
    args = parser.parse_args()
    target = (args.target or default_target()).resolve()
    # Universal-newline reading deliberately normalizes CRLF/LF mixtures.  The
    # installed source encountered during development contained both after a
    # local patch, and semantic block matching must remain reliable.
    content = target.read_text(encoding="utf-8")

    replacements = [
        (
            block(
                [
                    "        mask_shape = (",
                    "            (y_max + pad_y // 2) - (y_min - pad_y // 2),",
                    "            (x_max + pad_x // 2) - (x_min - pad_x // 2),",
                    "        )",
                ]
            ),
            block(
                [
                    "        mask_shape = (",
                    "            (y_max + pad_y // 2) - (y_min - pad_y // 2) + 1,",
                    "            (x_max + pad_x // 2) - (x_min - pad_x // 2) + 1,",
                    "        )",
                ]
            ),
        ),
        (
            block(
                [
                    "        out_shape_full = mask_shape",
                    "",
                    "        # Crop back to mask_shape",
                    "        start_indices = tuple(",
                    "            (full_size - out_size) // 2",
                    "            for full_size, out_size in zip(full_shape, mask_shape)",
                    "        )",
                ]
            ),
            block(
                [
                    "        # Keep every sample from the PSF-centred offset through the end of the",
                    "        # FFT work array.  Cropping to ``mask_shape`` here is too small for the",
                    "        # original native mask indices used below.  NumPy would flag those",
                    "        # indices as out of bounds, whereas JAX silently clamps them to the",
                    "        # final row / column, producing duplicated right and lower edges.",
                    "        start_indices = tuple(",
                    "            (full_size - out_size) // 2",
                    "            for full_size, out_size in zip(full_shape, mask_shape)",
                    "        )",
                    "        out_shape_full = tuple(",
                    "            fft_size - start",
                    "            for fft_size, start in zip(fft_shape, start_indices)",
                    "        )",
                ]
            ),
        ),
        (
            block(
                [
                    "        out_shape_full = mask_shape + (blurred_mapping_matrix_full.shape[2],)",
                ]
            ),
            block(
                [
                    "        out_shape_full = tuple(",
                    "            fft_size - start",
                    "            for fft_size, start in zip(fft_shape, start_indices[:2])",
                    "        ) + (blurred_mapping_matrix_full.shape[2],)",
                ]
            ),
        ),
    ]

    changed = False
    for old, new in replacements:
        if old in content:
            if content.count(old) != 1:
                raise RuntimeError(f"expected one match in {target}, found {content.count(old)}")
            content = content.replace(old, new)
            changed = True
        elif new not in content:
            raise RuntimeError(f"neither original nor patched block found in {target}")

    if not changed:
        print(f"Already patched: {target}")
        return
    backup = target.with_suffix(target.suffix + ".before_jax_crop_fix")
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(content, encoding="utf-8", newline="\n")
    print(f"Patched: {target}")
    print(f"Backup:  {backup}")


if __name__ == "__main__":
    main()
