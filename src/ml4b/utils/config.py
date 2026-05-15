"""Project configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve project root as the directory three levels above this file:
# src/ml4b/utils/config.py -> src/ml4b/utils -> src/ml4b -> src -> project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(_PROJECT_ROOT / ".env")


def _resolve(env_var: str, default: str) -> Path:
    """Return an absolute path from an env var, relative to project root."""
    raw = os.getenv(env_var, default)
    p = Path(raw)
    return p if p.is_absolute() else _PROJECT_ROOT / p


DATA_RAW: Path = _resolve("ML4B_DATA_RAW", "data/raw")
DATA_PROCESSED: Path = _resolve("ML4B_DATA_PROCESSED", "data/processed")
