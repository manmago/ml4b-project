"""Human-in-the-loop feedback and continual learning for ML4B (DECISIONS.md §8).

This package implements the correction → persist → retrain loop:

* :mod:`ml4b.feedback.store`   — persist the user's per-window label corrections
  (raw window samples + the corrected label) to a local feedback file, and read
  them back as a windowed DataFrame compatible with the training pipeline.
* :mod:`ml4b.feedback.retrain` — rebuild the model from the base Kaggle data
  *plus* the accumulated corrections, reusing the exact same windowing →
  augmentation → invariant-feature → Random Forest pipeline as initial training.

The design keeps **capture decoupled from training**: corrections are always
saved (so user effort is never lost), even on a fresh clone where the base
dataset needed for retraining is absent.
"""
