# ADR-011: Commit the Trained Model and Feature Names to Git

**Status:** Accepted
**Date:** 2026-05-29
**Deciders:** Anshul Agrawal

---

## Context

The final deliverable is a Streamlit app that a new team must be able to run
with three commands after cloning, **without** downloading the 2.5 GB RecoFit
dataset (see project_overview.md). The app needs two artifacts at runtime:

- `models/saved/best_model.joblib` — the trained Random Forest (~32 MB)
- `data/processed/feature_names.txt` — the ordered 47-feature list the model expects

The project's baseline `.gitignore` rules exclude `models/saved/`, all
`*.joblib` files, and everything under `data/processed/` — so by default the app
would not work after a fresh clone.

---

## Decision

Re-include exactly three files in git via `.gitignore` negations:

```gitignore
!models/saved/best_model.joblib
!models/saved/random_forest.joblib
!data/processed/feature_names.txt
```

To make the negations effective, `models/saved/` was changed to
`models/saved/*` (git cannot re-include a file whose parent directory is
excluded). All other binaries and processed CSVs remain ignored.

---

## Alternatives Considered

| Option | Why not chosen |
|--------|----------------|
| **Git LFS** | Adds a setup dependency (`git lfs install`) and a remote LFS budget; breaks the "clone + uv sync + run" promise for a new team. |
| **Model registry / cloud download** | Requires hosting, credentials, and network access at startup; overkill for a student project and a hand-over deliverable. |
| **Regenerate on first run** | Requires the 2.5 GB dataset and minutes of training before the app can start — defeats the zero-friction goal. |
| **Don't commit (status quo)** | App is unusable after a fresh clone without the dataset. |

## Rationale

The model is small (~32 MB), changes rarely, and is the single thing that makes
the app usable without the dataset. Committing it (plus the tiny feature list)
is the lowest-friction way to guarantee the 3-command quickstart and a
genuinely handover-ready repository. Reproducibility is preserved separately:
`scripts/train_model.py` regenerates the identical model (`random_state=42`)
from the dataset when needed.

---

## Consequences

**Positive:**
- App runs immediately after `git clone` + `uv sync` — no dataset, no extra tooling.
- Fully handover-ready; a new team needs nothing beyond the repository.
- The committed `random_forest.joblib` is an explicit archive copy of the chosen model.

**Negative / Trade-offs:**
- Adds ~64 MB of binaries to the repository history.
- The committed model can drift from the code if someone retrains but forgets to
  re-commit; mitigated by `scripts/train_model.py` being deterministic and documented.

**Neutral:**
- All other model binaries and processed CSVs remain git-ignored as before.
