#!/usr/bin/env bash
# One-click launcher for the ML4B Streamlit app (macOS / Linux / WSL).
#
# Usage:  ./run_app.sh
#
# Requires only `uv` to be installed (see the Setup guides). `uv run`
# automatically creates the virtual environment and installs every dependency
# from pyproject.toml / uv.lock on first launch — no manual `uv sync`, no pip,
# no conda, no separate Python install needed. The trained model is committed,
# so no dataset download is required just to run the app.

set -euo pipefail

# Resolve the project root from this script's location so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Fail early with a friendly message if uv is missing.
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is not installed."
  echo "Install it (macOS/Linux/WSL):  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "Then re-run:  ./run_app.sh"
  exit 1
fi

echo "Starting the ML4B app — open http://localhost:8501 in your browser."
exec uv run streamlit run app/streamlit_app.py
