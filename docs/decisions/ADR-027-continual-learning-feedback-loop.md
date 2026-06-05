# ADR-027: Human-in-the-Loop Correction & Continual Learning
**Status:** Accepted
**Date:** 2026-06-05

## Context
The project's central limitation is that the model is trained on a **single
subject** (ADR-016, ADR-021), so it transfers imperfectly to a new person — the
documented fix is "a small number of the user's own labelled recordings"
(ADR-013/014). Nothing in the app captured that signal: a user could see a wrong
prediction but had no way to correct it or improve the model. We want the app to
get better the more it is used, by letting the user correct predictions and fold
those corrections back into the model.

## Decision
Add a **human-in-the-loop feedback loop** in a new package `src/ml4b/feedback/`:

1. **Capture (app).** On the Predict page, an "✏️ Correct & Improve" editor lets
   the user set the correct label per window (choosing a known class, `rest`, or
   typing a **new** exercise). Corrections are saved immediately.
2. **Store (`feedback/store.py`).** Each correction persists the **raw window
   samples** (six canonical-unit channels) + the corrected label + light metadata
   to `data/feedback/feedback.jsonl` (append-only, dependency-free, git-ignored).
   Raw windows — not features — are stored, so corrections survive a feature-set
   change and are re-featurised through the shared pipeline at retrain time.
3. **Retrain (`feedback/retrain.py`, `scripts/update_model.py`).** Rebuild the
   model from the **base Kaggle data + accumulated corrections** through the
   *identical* windowing → augmentation → invariant-feature → Random Forest
   pipeline as initial training. Corrections are repeated (`feedback_repeat`) and
   augmented like all windows so a handful of examples carry real weight. New
   labels become new classes automatically. The originally-shipped model is
   backed up to `best_model_base.joblib` (restorable) and a `model_manifest.json`
   records what went into each retrain.

Capture is **decoupled** from retraining: corrections are always stored, and if
the base dataset is absent (a fresh handover clone) the app disables the retrain
button but still collects feedback for later (`scripts/update_model.py`).

## Alternatives Considered
- **True online learning (`partial_fit`)** — would require replacing the tuned
  Random Forest with an SGD/incremental model. Rejected: it discards the chosen
  model and a few single-sample updates invite catastrophic forgetting.
- **Train a separate per-user personalisation model and blend** — more moving
  parts and fragile with very few samples; a feedback-augmented full retrain is
  simpler and reuses the existing, tested pipeline.
- **Store extracted features instead of raw windows** — smaller, but breaks if
  the feature set ever changes and violates the single-pipeline rule. Rejected.
- **Auto-retrain on every correction** — slow and unpredictable; retraining is an
  explicit, user-triggered action instead.

## Consequences
- **Positive:** directly attacks the single-subject limitation; the app improves
  with use; users can add **new exercises**; reuses the exact training pipeline
  (no model-contract change); fully reproducible (`scripts/update_model.py`,
  seed 42); base model is always recoverable.
- **Negative:** retraining needs the base dataset and takes a few minutes (it is
  not real-time). A user-retrained `best_model.joblib` differs from the committed
  one (shown as a local git change — intentional, not auto-committed). The
  **novelty detector** (ADR-024) is *not* refit by this loop, so a freshly added
  class may be flagged `unknown` until `scripts/fit_novelty_detector.py` is re-run
  or the detector is removed.
