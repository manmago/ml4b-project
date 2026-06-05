# Night Log — Autonomous Overnight Session (2026-06-05 → 06-06)

Worked autonomously overnight on three independent tasks, each on its **own
feature branch off `develop`**. **Nothing is merged to `main` or `develop`.**
This copy lives on `feature/continuous-learning`; its Branch 3 section is the
authoritative record for this branch.

## Overview of the three branches

| Branch | Task | Status |
|--------|------|--------|
| `feature/uncertain-overall-result` | Report **uncertain** as the overall/per-set result when it is the most common label (rest excluded) instead of forcing a minority exercise. | ✅ Done |
| `feature/consolidate-decisions` | Replace the 24 fine-grained ADRs with **one `docs/DECISIONS.md`**, update every reference. | ✅ Done |
| `feature/continuous-learning` | Human-in-the-loop **correction + continual-learning loop**. | ✅ Done (this branch) |

How to review:
```bash
git log develop..feature/<name>      # commits added by that branch
git diff develop...feature/<name>
```

---

## Branch 3 — `feature/continuous-learning`

### Idea
Let the user upload data, see predictions, **correct** them, and have the model
**learn** from those corrections (your idea — confirmed as the right approach).
Implemented as a proper human-in-the-loop loop following best practices.

### Design decisions (full rationale in `docs/decisions/ADR-027`)
- **Why not online `partial_fit`:** the shipped model is a Random Forest (no
  incremental fit), and single-sample updates invite catastrophic forgetting.
  The robust, standard pattern is a **feedback-augmented retrain**: rebuild from
  the base data **plus** corrections through the *same* training pipeline.
- **Store raw windows, not features:** corrections survive a feature-set change
  and are re-featurised through the shared pipeline at retrain time.
- **Capture is decoupled from training:** corrections are always saved, even when
  the base dataset isn't present (a fresh clone) — so user effort is never lost.
- **New exercises supported:** correcting to a new label trains a new class.

### What changed
- **`src/ml4b/feedback/` (new package)**
  - `store.py` — persist/load corrections to `data/feedback/feedback.jsonl`
    (raw window channels + corrected label + metadata); `build_records`,
    `append`, `load`, `stats`, `to_window_df`, `clear`.
  - `retrain.py` — `retrain_model()` rebuilds from base Kaggle data + repeated,
    augmented corrections (same windowing → augmentation → invariant features →
    Random Forest, seed 42); backs up the shipped model to
    `best_model_base.joblib`, writes `model_manifest.json`; `base_available()`,
    `restore_base_model()`.
- **`src/ml4b/data/apple_watch_loader.py`** — `predict_from_sensor_logger` gains
  `return_windows=True` to also return the raw windows (needed to store
  corrections). Default behaviour unchanged (all existing callers unaffected).
- **`app/pages/prediction.py`** — new **✏️ Correct & Improve** section: a
  per-window editor to set the correct label (incl. a new exercise), **Save**
  (persists corrections), and **🔁 Retrain model with my corrections** (rebuilds
  and hot-swaps the model). Disabled with a clear message if the base dataset is
  absent; corrections are still collected.
- **`scripts/update_model.py`** — CLI for the retrain (`--repeat`, `--n-augment`,
  `--restore-base`).
- **`src/ml4b/utils/config.py`** — new `DATA_FEEDBACK` path.
- **`.gitignore`** — ignore `data/feedback/` (the user's own data).
- **Tests** — `tests/test_feedback.py` (9): store round-trip, stats, schema,
  confidence serialisation, and that corrections flow through the real
  augmentation + invariant-feature pipeline.
- **Docs** — `ADR-027`, plus README, STRUCTURE, architecture, crisp_dm_log and
  data_dictionary updated (feedback store schema + continual-learning flow).

### Verification
- `uv run ruff format/check` — clean. `uv run pytest` — **66 passed** (9 new).
- **End-to-end smoke test on a real sample**: predict (598 windows, raw windows
  aligned) → correct → store → stats — all OK.
- **Full retrain validated once on real data**, including teaching a brand-new
  class `squat` (manifest classes became
  `[bicep_curl, row, squat, tricep_extension]`; the reloaded model knew `squat`).
  The committed `best_model.joblib` / `feature_names.txt` were **backed up and
  restored** afterwards, so this branch leaves the shipped model untouched.

### Notes / limitations (also in ADR-027)
- Retraining needs the base dataset (`data/raw/kaggle_gym_imu/`) and takes a few
  minutes — it is explicit, not real-time.
- A user-retrained `best_model.joblib` differs from the committed one (shows as a
  local git change — intentional, not auto-committed; `--restore-base` undoes it).
- The **novelty detector** (ADR-024) is not refit by this loop, so a freshly
  added class may be flagged `unknown` until `scripts/fit_novelty_detector.py` is
  re-run or the detector removed.

### Cross-branch note
This branch uses **ADR-027** (Branch 1 uses ADR-026), so merging both causes no
file collision. If you adopt Branch 2's consolidated `DECISIONS.md`, fold the
ADR-026/027 decisions into it and delete the ADR files.
