# ADR-013: Switch Training Dataset from RecoFit to MM-Fit

**Status:** Accepted
**Date:** 2026-05-29
**Deciders:** Anshul Agrawal

---

## Context

The end goal is recognizing exercises from an **Apple Watch worn on the wrist**.
The original model was trained on **RecoFit** (Microsoft). After fixing the
Sensor Logger column mapping, the 100 → 50 Hz sampling rate, and aligning units
(ADR-012), real Apple Watch bicep-curl recordings were **still** misclassified.

A feature-distribution diagnostic (`scripts/test_apple_watch_prediction.py`)
showed the per-axis features were out of the training distribution even after
units matched. The root cause is a **sensor-placement domain gap**:

- **RecoFit** recorded with an inertial **armband on the forearm**
  (SparkFun IMU) — confirmed from the dataset's documentation.
- An **Apple Watch sits on the wrist.** During arm exercises the wrist adds
  supination/pronation and a different gravity orientation that a forearm
  sensor near the elbow never sees.

No amount of preprocessing can close a *placement* gap — the signals describe
different body segments. The fix must be a dataset recorded at the **wrist**.

## Decision

Replace RecoFit with the **MM-Fit** dataset (Strömbäck, Huang & Radu,
UbiComp/ISWC 2020, CC-BY-4.0) as the training source.

- Use only the **smartwatch accelerometer + gyroscope** modalities, from
  **both wrists** (`sw_l_*` and `sw_r_*`) — matches the deployment device and,
  because left/right wrist axes are mirror images, makes the model agnostic to
  which wrist the Apple Watch is on.
- New loader `src/ml4b/data/mmfit_loader.py` emits the **same long-format
  schema** as `load_recofit_raw`, so windowing + feature extraction are reused
  unchanged (the project's shared-pipeline rule).
- New builder `scripts/build_mmfit_dataset.py` writes the standard processed
  CSVs, so `scripts/train_model.py`, the notebooks and the app are unchanged.
- **Classes:** keep the original 6 and add **`push_up`** (MM-Fit `pushups`),
  giving **7 classes**. MM-Fit exercises with no ML4B equivalent (lunges,
  sit-ups, dumbbell rows, jumping jacks) are dropped.
- **Split:** MM-Fit's official workout-id partition (train 10 / val 3 / test 3),
  a session/subject-level split — no leakage.
- **Units (canonical = MM-Fit):** accelerometer **m/s² including gravity**,
  gyroscope **rad/s**. The Apple Watch loader is re-aligned to *these* units:
  `ax = (accelerationX + gravityX) * 9.80665` (g → m/s²) and the gyroscope is
  left in rad/s (no rad/s → deg/s conversion — that ADR-012 step is reverted).
- 100 → 50 Hz decimation is retained so a 100-sample window stays 2 s.

## Alternatives Considered

| Option | Why not chosen |
|--------|----------------|
| **Keep RecoFit, only fix preprocessing** | Proven insufficient — the forearm/wrist placement gap is physical, not a units/rate bug. |
| **Collect a large labelled Apple-Watch dataset and train on it** | Not feasible: the team cannot record the hundreds of sessions needed. |
| **Orientation-invariant (magnitude-only) features on RecoFit** | Magnitude-only features discard the *direction* information that distinguishes arm exercises (bicep vs lateral vs shoulder), so it cannot separate them. |
| **Combine RecoFit + MM-Fit** | Mixing forearm and wrist placements dilutes the wrist signal; deferred — MM-Fit alone already matches the device. |

## Rationale

MM-Fit is the closest public dataset to the deployment setup: **wrist-worn
consumer smartwatch, the same gym exercises**, already segmented and labelled.
Switching the dataset attacks the actual root cause (placement) instead of
adding more preprocessing on top of a mismatched source.

## Consequences

**Positive:**
- In-domain performance is strong on held-out workouts: **Val macro F1 0.880 /
  acc 0.948**, **Test macro F1 0.961 / acc 0.985** (vs RecoFit Test ≈ 0.80).
- **`push_up` is now a class and is correctly recognized on a real Apple Watch
  recording** — previously impossible.
- Units genuinely match the training distribution (diagnostic z-scores dropped
  from > 10 to < 2).

**Negative / Trade-offs:**
- MM-Fit uses Wear-OS (TicWatch) smartwatches, not Apple Watch, so a smaller
  residual device/orientation gap remains.
- The 1.7 GB dataset must be downloaded to rebuild from scratch (the trained
  model is committed, so the app runs without it).

**Honest limitation (residual domain gap):** on real Apple Watch samples,
`push_up` is recognized, but `bicep_curl` is still confused with
`tricep_extension` (and quiet recordings with `rest`). Two reasons: (1) a
residual Apple-Watch-vs-TicWatch orientation difference, and (2) bicep curl and
tricep extension are **biomechanically near-identical at the wrist** (both are
elbow rotations) — even the in-domain test confuses them most (`bicep_curl`
F1 0.89, the lowest class). Rotation augmentation was tried to close the
orientation part and **rejected** — see ADR-014. Closing the bicep/tricep gap
specifically would need a small number of the user's own labelled recordings.
