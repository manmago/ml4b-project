"""Project configuration loaded from environment variables.

Loads a .env file from the project root (if present) and exposes
all configurable paths as pathlib.Path objects. Every path is resolved
relative to the project root so the package works regardless of where
it is installed or invoked.

Typical usage:
    from ml4b.utils.config import PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, MODELS_DIR
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Derived from __file__ — reliable in all contexts (installed package, editable install,
# Jupyter kernel regardless of CWD, VS Code, command line).
# src/ml4b/utils/config.py → utils → ml4b → src → project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Find the project root by walking up from the current working directory.

    Designed for use in Jupyter notebooks, scripts, and any context where
    ``__file__`` is unavailable or points to a temporary location. Works
    identically on WSL, macOS, and Windows.

    Args:
        marker: Filename that must exist in the project root directory.
            Defaults to ``"pyproject.toml"``.

    Returns:
        Absolute Path to the project root directory.

    Raises:
        FileNotFoundError: If no parent directory contains the marker file.
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Could not find project root (no {marker} found starting from {current})"
    )


def _resolve(env_var: str, default: str) -> Path:
    """Resolve an env var to an absolute Path relative to the project root.

    Args:
        env_var: Name of the environment variable to read.
        default: Fallback path string (relative to project root) when the
            variable is not set.

    Returns:
        An absolute Path. If the env var holds a relative path it is
        anchored to the project root; absolute paths are used as-is.
    """
    raw = os.getenv(env_var, default)
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


DATA_RAW: Path = _resolve("ML4B_DATA_RAW", "data/raw")
DATA_PROCESSED: Path = _resolve("ML4B_DATA_PROCESSED", "data/processed")
# User-correction feedback for continual learning (DECISIONS.md §8). Not in git — it is
# the end user's own labelled corrections, persisted locally and fed back into
# retraining.
DATA_FEEDBACK: Path = _resolve("ML4B_DATA_FEEDBACK", "data/feedback")
MODELS_DIR: Path = _resolve("ML4B_MODELS_DIR", "models/saved")
REPORTS_DIR: Path = _resolve("ML4B_REPORTS_DIR", "reports/figures")
