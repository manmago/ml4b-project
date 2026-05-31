# ADR-018: Device-Invariant Feature Set
**Status:** Accepted
**Date:** 2026-05-31

## Context
The model trains on one person's Apple Watch (Kaggle anchor, ADR-016) but must
generalise to other people's watches. Raw per-axis statistics (e.g. `az_mean`)
depend on exactly how the watch sits on the wrist — handedness, strap tightness,
forearm rotation — which differs between people and devices. The earlier
per-axis feature set (`ml4b.data.features`) was sensitive to this and contributed
to bicep/triceps confusion on real recordings.

## Decision
Use a new feature module `src/ml4b/data/features_invariant.py` (39 features)
built from quantities that are robust to watch orientation and per-device
scale/offset:
- **Magnitude features** on `|accel|` and `|gyro|` — mean, std, min, max, range,
  RMS, MAD, zero-crossing rate, dominant frequency, spectral energy. Magnitudes
  are invariant to device rotation.
- **Per-window z-normalized shape features** — each axis is standardized within
  the window before computing zero-crossing rate and dominant frequency, so only
  the *pattern* of motion survives, not its absolute scale/offset.
- **Axis-pair correlations** — coordination structure that is invariant to
  per-axis offset and gain.
- **Gyro/accel ratio** — how rotational vs translational the movement is.

The identical module is used in training and in the app (the CLAUDE.md reuse
rule).

## Alternatives Considered
- **Keep the per-axis `features.py` (47 features)** — orientation-sensitive;
  retained only for the legacy MM-Fit pipeline, not used by the final model.
- **Deep learning on raw windows (CNN)** — would learn its own invariances but
  needs far more (and more diverse) data than a single subject provides, and is
  heavier to deploy. Rejected for this dataset size.
- **Hand-craft only magnitude features** — loses the coordination/shape cues that
  help separate the three classes; the z-normalized shape and correlation
  features add discrimination without re-introducing scale sensitivity.

## Rationale
Invariant features are the cheapest, most reliable way to transfer a
single-subject model to new users without collecting target-domain data. They
pair with rotation/mirror augmentation (ADR-019), which further teaches
orientation invariance.

## Consequences
- **Positive:** more robust cross-device/cross-user generalisation; smaller,
  interpretable feature set; shared by training and inference.
- **Negative:** discards some absolute-orientation information that *could* help
  in-domain; the honest leave-one-set-out score (ADR-021) reflects this trade-off
  and is the number we report.
