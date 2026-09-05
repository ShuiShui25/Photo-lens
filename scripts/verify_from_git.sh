#!/usr/bin/env bash
# Create a clean environment, clone GitHub, install the clone, and run its test.
set -euo pipefail
VERIFY_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/photo-lens-clone-XXXXXX")"
VERIFY_PYTHON="${PHOTO_LENS_BASE_PYTHON:-python3.10}"
VERIFY_URL="${PHOTO_LENS_GIT_URL:-git@github.com:ShuiShui25/Photo-lens.git}"
"$VERIFY_PYTHON" -m venv "$VERIFY_ROOT/env"
git clone "$VERIFY_URL" "$VERIFY_ROOT/repo"
cd "$VERIFY_ROOT/repo"
if [[ -n "${PHOTO_LENS_GIT_REF:-}" ]]; then git checkout "$PHOTO_LENS_GIT_REF"; fi
"$VERIFY_ROOT/env/bin/python" -m pip install -r requirements.txt
"$VERIFY_ROOT/env/bin/python" -m pip check
MPLBACKEND=Agg MPLCONFIGDIR="$VERIFY_ROOT/matplotlib" OPENBLAS_NUM_THREADS=1 \
  "$VERIFY_ROOT/env/bin/python" -u tests/smoke_test.py
printf 'Verified commit: '
git rev-parse HEAD
printf 'Validation directory: %s\n' "$VERIFY_ROOT"
