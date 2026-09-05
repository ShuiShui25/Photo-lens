#!/usr/bin/env bash
# Start Photo-lens from any working directory; no conda activation is needed.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for interpreter in python3 python; do
    if command -v "$interpreter" >/dev/null 2>&1; then
        exec "$interpreter" "$repo_dir/scripts/launch_dashboard.py" "$@"
    fi
done
echo 'Python is required. Install the environments described in README.md first.' >&2
exit 1
