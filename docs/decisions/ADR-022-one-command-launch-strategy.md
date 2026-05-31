# ADR-022: One-Command Launch Strategy
**Status:** Accepted
**Date:** 2026-05-31

## Context
The project must be handover-ready: a brand-new team should run the app with the
fewest possible commands and no fragile manual setup. Earlier guides instructed
users to run `uv sync` before `uv run streamlit ...` and to install Python
separately, which is more steps than necessary and a common source of "it
doesn't work on my machine".

## Decision
Standardise on a **single command** to launch the app from a fresh clone:

```bash
git clone <repo> && cd ml4b-project && uv run streamlit run app/streamlit_app.py
```

`uv run` automatically creates the virtual environment and installs every
dependency from `pyproject.toml` / `uv.lock` on first run — so **no separate
`uv sync`, no `pip`, no `conda`, and no manual Python install** are needed (uv
fetches the right Python too). `uv.lock` is committed for reproducible, fast
installs. The trained model and metrics are committed (ADR-011), so **no dataset
download** is required to run the app.

Convenience entry points wrap this so users need not memorise it:
- `make run` / `make train` / `make test` / `make lint` (Makefile),
- `./run_app.sh` (macOS/Linux/WSL) and `run_app.bat` (Windows) — one-click
  launchers that check for `uv` and start the app.

## Alternatives Considered
- **`uv sync` then `uv run`** — an extra, redundant step; `uv run` already syncs.
- **`pip install -r requirements.txt` in a manual venv** — slower, less
  reproducible, requires a pre-installed Python, more steps.
- **conda environment** — heavier, slower, another tool to install.
- **Docker** — fully reproducible but adds a large dependency (Docker Desktop)
  and is overkill for a Streamlit app a student team must run quickly.

## Rationale
`uv run` collapses environment creation, dependency install and launch into one
command that is identical on all three OSes, which is the minimal, most
reliable path to a running app. The Makefile and one-click scripts make it
approachable for non-CLI users without hiding what actually runs.

## Consequences
- **Positive:** clone-to-running in one command on WSL, macOS and Windows; no
  Python/conda/pip prerequisites; reproducible via `uv.lock`; the app needs no
  dataset.
- **Negative:** `uv` itself must be installed once (a single documented
  command per OS); the first `uv run` is slower while it provisions the
  environment (subsequent runs are instant).
