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

## Alternatives Considered

| Option | Why not chosen |
|--------|----------------|
| **conda / mamba** | Heavyweight, slower, licensing overhead for the default channels, and a separate `environment.yml` to maintain. |
| **pip + venv** | No built-in lockfile for transitive pins; reproducibility requires extra tooling (`pip-tools`) and manual venv activation. |
| **poetry** | Solid lockfile support but noticeably slower resolves/installs than uv, and an extra tool layer on top of `pyproject.toml`. |

## Rationale

uv reads the standard `pyproject.toml`, produces a fully-pinned `uv.lock`, and
resolves/installs 10–100× faster than pip/poetry — which matters most for
onboarding and CI. `uv run` executes tools inside the managed venv without
manual activation, so the same commands work identically on WSL, macOS, and
Windows. This directly satisfies the cross-platform reproducibility requirement
with the least friction.

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
- `conda` is not used in this project
