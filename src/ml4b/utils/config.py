"""Project configuration loaded from environment variables.

Loads a .env file from the project root (if present) and exposes
all configurable paths as pathlib.Path objects. Every path is resolved
relative to the project root so the package works regardless of where
it is installed or invoked.

Typical usage:
    from ml4b.utils.config import DATA_RAW, DATA_PROCESSED, MODELS_DIR
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: src/ml4b/utils/config.py → utils → ml4b → src → root
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

load_dotenv(_PROJECT_ROOT / ".env")


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
    return p if p.is_absolute() else _PROJECT_ROOT / p


DATA_RAW: Path = _resolve("ML4B_DATA_RAW", "data/raw")
DATA_PROCESSED: Path = _resolve("ML4B_DATA_PROCESSED", "data/processed")
MODELS_DIR: Path = _resolve("ML4B_MODELS_DIR", "models/saved")
