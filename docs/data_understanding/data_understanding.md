# Data Understanding — ML4B Gym Exercise Recognition

> CRISP-DM Phase 2 — short narrative. Last updated: 2026-06-17

## Goal of this phase

Find sensor data that lets a model recognise three gym exercises (bicep curl,
triceps extension, row) from a **wrist-worn Apple Watch** and understand its
properties well enough to build the preprocessing pipeline.

## What we learned

The decisive finding of this phase is a **domain lesson, not a dataset score**:
a model only transfers to the Apple Watch if it is trained on data from the
**same sensor placement and device family**. In-sample accuracy on a mismatched
sensor is misleading.

- We first anchored on **RecoFit** (forearm armband) — a placement mismatch that
  no preprocessing could close.
- We then switched to **MM-Fit** (wrist, but a non-Apple smartwatch) — still a
  device-domain gap to the Apple Watch.
- We finally settled on the **Kaggle "Gym Workout IMU" dataset**, recorded *on an
  Apple Watch SE* at 100 Hz, and complement it with **our own committed
  Apple-Watch recordings** in `data/Testdaten/`.

## Conclusion — the data we use

**Kaggle Apple-Watch anchor + our own Testdaten.** Both are Apple Watch /
CoreMotion data, so they share one device domain. The single-subject Kaggle
anchor is the central limitation; our own recordings are the concrete mitigation.

## Where the detail lives

- **Dataset comparison & rationale** → [`../data/dataset_evaluation.md`](../data/dataset_evaluation.md)
- **Columns, units & engineered features** → [`../data/data_dictionary.md`](../data/data_dictionary.md)
- **All major decisions** → [`../DECISIONS.md`](../DECISIONS.md)
- **Exploration notebook** → `notebooks/02_data_understanding.ipynb`
