# ADR-014: Rotation Augmentation — Implemented but Disabled by Default

**Status:** Accepted (augmentation **off** by default)
**Date:** 2026-05-29
**Deciders:** Anshul Agrawal

---

## Context

After switching to MM-Fit (ADR-013), real Apple Watch recordings showed a
residual domain gap: the **resting orientation of the watch on the wrist**
differs between the Apple Watch and MM-Fit's TicWatch, which shifts the per-axis
gravity distribution (top out-of-distribution features were `az_mean`,
`ay_mean`). The textbook no-new-data fix for an orientation gap is **rotation
augmentation** (domain randomization): rotate each training window's 3-axis
accelerometer and gyroscope vectors by random 3-D rotations so the model becomes
orientation-robust.

## Decision

Implement rotation augmentation as a reusable, tested module
(`src/ml4b/data/augmentation.py`, `--augment N` flag in
`scripts/build_mmfit_dataset.py`) but **leave it disabled by default**
(`--augment 0`). The committed model is trained **without** augmentation.

## Alternatives Considered

| Option | Result |
|--------|--------|
| **No augmentation (chosen)** | Test macro F1 **0.961**; `push_up` recognized on real Apple Watch; `bicep_curl` confused with `tricep_extension`. |
| **3× rotation augmentation (rejected)** | Test macro F1 dropped to **0.945**, and on the real bicep samples it made the error *worse* (tricep confidence 52% → 74%). |

## Rationale

The evidence contradicted the hypothesis. Full 3-D rotation augmentation forces
the model onto the rotation-**invariant** *shape* of the motion — but bicep curl
and tricep extension have almost the same wrist-motion shape (both are elbow
rotations). Removing orientation cues therefore removed the very signal that
separated them, making the confusion worse while also costing in-domain
accuracy. Augmentation did not help the one case it was meant to fix.

## Consequences

**Positive:**
- The capability is preserved, documented and unit-tested, so a future team can
  re-enable or refine it (e.g. constrained wrist-only rotations) by running
  `uv run python scripts/build_mmfit_dataset.py --augment N`.
- The decision is evidence-based and reproducible.

**Negative:**
- The residual orientation gap is not closed by augmentation; the robust fix for
  the bicep/tricep case remains a small set of user-recorded labelled sessions
  (see ADR-013).
