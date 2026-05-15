# ADR-001: Python Package Manager — uv

**Status:** Accepted  
**Date:** 2026-05-15  
**Deciders:** Anshul Agrawal

---

## Context

The project must run identically on WSL (Windows), native Windows, and macOS across multiple team members. We need:

- A reproducible, lockfile-based dependency resolution
- Fast environment setup (important for onboarding and CI)
- A single configuration file (`pyproject.toml`) as the source of truth
- No dependency on Anaconda/conda, which carries licensing and size overhead

Candidates evaluated: `conda`, `pip + venv`, `poetry`, `uv`.

---

## Decision

We use **uv** (by Astral) as the sole package and environment manager.

- `pyproject.toml` declares all dependencies (runtime + dev extras)
- `uv.lock` is committed and pins every transitive dependency exactly
- `uv sync` is the single command to reproduce the full environment
- `uv run <cmd>` executes tools (ruff, pytest, streamlit) inside the managed venv without manual activation

---

## Consequences

**Positive:**
- 10–100× faster installs compared to pip/conda
- Deterministic environments via lockfile (no "works on my machine")
- Single `pyproject.toml` replaces `requirements.txt`, `setup.py`, `setup.cfg`, `environment.yml`
- Supported on all target platforms (WSL, macOS, Windows)

**Negative / Trade-offs:**
- `uv` is newer than pip/conda — team members may be unfamiliar
- Some complex conda-only packages (e.g. CUDA-linked binaries) require workarounds
- The lockfile (`uv.lock`) must be updated and committed whenever dependencies change (`uv add <pkg>` does this automatically)

**Neutral:**
- `conda` is not used; the `Course_Files/` conda environment files are for reference only and must not be imported into this project
