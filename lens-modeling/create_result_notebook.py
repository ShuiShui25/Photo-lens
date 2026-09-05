#!/usr/bin/env python
"""Create a compact, already-executed notebook that displays all four stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("system")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    system_dir = root / f"system_{args.system}"
    config = json.loads((system_dir / args.config).read_text())
    output_dir = config.get("output_dir", "output")
    label = f" ({args.tag})" if args.tag else ""
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "autolens-jax", "language": "python", "name": "python3"
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            f"# System {args.system}{label}: PHOTO-LENS03 four-stage modeling\n\n"
            f"BB backend: **{config['bb_survey']} {config['bb_band']}**. "
            "Stage 4 uses `ConstantPlusBrightnessZeroth(signal_scale=0.5)`."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nfrom IPython.display import Image, display\n"
            "root = Path.cwd()\nout = root / 'system_" + args.system + "' / "
            + repr(output_dir) + "\n"
            "for stage in (1,2,3,4):\n"
            "    path = out / 'stage_previews' / f'stage{stage}_fit.png'\n"
            "    print(f'Stage {stage}: {path}')\n    display(Image(filename=str(path)))"
        ),
        nbf.v4.new_code_cell(
            "import json\nsummary=json.loads((out/'stage4_run_summary.json').read_text())\nsummary"
        ),
    ]
    tag = f"_{args.tag}" if args.tag else ""
    source = root / f"{args.system}{tag}_four_stage_modeling.ipynb"
    nbf.write(nb, source)
    print(source)


if __name__ == "__main__":
    main()
