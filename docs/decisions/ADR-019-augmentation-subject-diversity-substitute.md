# ADR-019: Data Augmentation as a Subject-Diversity Substitute
**Status:** Accepted
**Date:** 2026-05-31
**Supersedes:** ADR-014 (rotation augmentation rejected — see Rationale)

## Context
The final model trains on a **single subject** (Kaggle Apple-Watch anchor,
ADR-016) but must generalise to other people. We cannot collect more subjects
(no access to additional Apple-Watch recordings). The variability a model
normally gets from many subjects — different watch orientations, handedness, rep
tempos, body sizes — is absent.

## Decision
Synthesise that variability with composed augmentation
(`src/ml4b/data/augmentation.py`, `augment_windows`). For each training window,
5 augmented copies are generated (→ 6× data), each applying in order:
- **Random 3-D rotation** (shared by accel + gyro) — different watch orientation
  / mounting and arm-posture handedness.
- **Time-warp** (tempo factor 0.8–1.2×) — faster/slower repetition speeds.
- **Axis mirror** (probability 0.5) — wearing the watch on the other wrist (the
  Kaggle data is left-wrist only).
- **Per-axis Gaussian jitter** (5% of each axis's std) — sensor/body variation.

Only training windows are augmented; held-out sets stay pristine, and augmented
copies of a held-out set are excluded from its training fold (ADR-021), so
metrics remain leak-free.

## Alternatives Considered
- **No augmentation** — leaves the model overfit to one person's style.
- **Rotation only** — what ADR-014 evaluated and rejected on the *MM-Fit*
  pipeline. Insufficient alone for the single-subject case.
- **Collect real multi-subject Apple-Watch data** — not possible (no access).

## Rationale
ADR-014 rejected rotation augmentation, but that decision was made on the
*multi-subject MM-Fit* dataset, where it hurt in-domain F1 and the bicep/triceps
case. The situation has fundamentally changed: the anchor is now **single
subject**, so synthetic variability is the only available stand-in for missing
subjects, and the invariant features (ADR-018) make the augmentations act on the
right quantities. Composing rotation with time-warp, mirror and jitter covers
the dominant real-world nuisance factors. This is a standard, documented HAR
technique when target-domain data cannot be collected.

## Consequences
- **Positive:** effectively multiplies "subjects" ~6×; improves robustness to
  orientation, handedness and tempo without any new data.
- **Negative:** augmented data is synthetic, not real — it cannot fully replace
  true subject diversity, so real-world accuracy will still trail the in-domain
  test numbers (documented as a limitation). Training is ~6× slower.
