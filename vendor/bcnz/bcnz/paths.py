"""Resolve optional external datasets without machine-specific defaults."""
import os
from pathlib import Path


def input_path(value, environment, *, directory=False):
    """Explicit input takes precedence over an environment variable.

    These external files are not included in BCNz. Relative paths are resolved
    against the caller's working directory; home-directory expansion is allowed.
    """
    value = value if value is not None else os.environ.get(environment)
    if value is None or not str(value).strip():
        raise ValueError(
            f"External data is required: pass its path or set {environment}."
        )
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{environment}: {kind} not found: {path}")
    return path
