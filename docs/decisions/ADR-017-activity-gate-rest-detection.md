# ADR-017: Energy-Threshold Activity Gate for Rest Detection
**Status:** Accepted
**Date:** 2026-05-31

## Context
Real gym recordings contain long pauses between sets. Previous models learned a
`rest` class from the training data and then **over-predicted rest** on real
Apple-Watch uploads: a new user's idle behaviour (fidgeting, adjusting the
watch, walking, drinking) does not resemble the training set's clean rest, so
the learned rest boundary transfers badly and swallows genuine activity. The new
Kaggle training anchor (ADR-016) also has **no rest data at all**, so a learned
rest class is not even an option.

## Decision
Detect rest with a simple, device-agnostic **energy threshold** instead of a
trained class (`src/ml4b/data/activity_gate.py`). A window is "active" if
*either*:
- the standard deviation of the accelerometer magnitude > `0.08 g`, **or**
- the mean gyroscope magnitude > `0.30 rad/s`.

Windows that clear neither threshold are labelled `rest` and never reach the
model. The same gate is used in training-data exploration and in the app, so the
three trained classes only ever compete on genuine movement.

## Alternatives Considered
- **Learned `rest` class** — what previous versions did; over-predicted on real
  data and impossible here (no rest in the Kaggle anchor). Rejected.
- **Single-signal gate (accel only or gyro only)** — less robust; some exercises
  are rotation-dominant (curl) and others translation-dominant (row), so an OR
  over both signals catches all three. Chosen.
- **Learned one-class / novelty detector for rest** — more moving parts and
  needs rest examples to tune; the threshold is simpler and transparent.

## Rationale
Rest is fundamentally a *low-energy physical state*, so a threshold on movement
energy is far more transferable across devices and people than a learned
boundary. Calibration on the Kaggle exercise windows shows ~90% of genuine
exercise windows clear the thresholds with a wide margin (the gated ~10% are
genuine low-motion moments at set start/end and between reps), while truly still
pauses fall below — exactly the desired behaviour.

## Consequences
- **Positive:** directly fixes rest over-prediction; needs no rest training
  data; transparent and easy to recalibrate; shared by training and app.
- **Negative:** the thresholds are dataset-calibrated constants — a very gentle,
  slow exercise could occasionally be gated as rest; the ~10% of low-energy
  exercise windows that are gated slightly reduce recall. Thresholds are
  module-level constants so they can be tuned if real-world data warrants.
