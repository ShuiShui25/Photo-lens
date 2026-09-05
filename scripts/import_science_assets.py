#!/usr/bin/env python3
"""Copy full-resolution science assets from an existing PHOTO-LENS03 tree.

These large assets are not needed by tests/smoke_test.py. No symlinks to the
original workspace are used. The destination is ignored by Git.
"""
import argparse
from pathlib import Path
import shutil
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    paths = ['data', 'photometric-estimation/model', 'photometric-estimation/hsc-g-response/model-g']
    for name in paths:
        if not (source / name).is_dir():
            parser.error(f'Missing asset directory: {source / name}')
        if (ROOT / name).exists():
            parser.error(f'Destination already exists: {ROOT / name}; use a fresh checkout')
    for name in paths:
        shutil.copytree(source / name, ROOT / name)
        print(f'Copied {name}')
if __name__ == '__main__':
    main()
