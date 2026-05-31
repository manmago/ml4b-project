# ADR-023: Notebooks Migrated to the 3-Class Apple-Watch Pipeline
**Status:** Accepted
**Date:** 2026-06-01

## Context
The source modules, the Streamlit app, and all prose documentation were migrated
to the final **3-class Apple-Watch** pipeline (Kaggle anchor; ADR-016). The
Jupyter notebooks, however, still documented the **old 6-class RecoFit** pipeline
(and its later MM-Fit iteration). This was the last remaining inconsistency: a
new team reading the notebooks would see a different dataset, class set, feature
set, and evaluation method than the code actually uses.

## Decision
Rewrite every notebook to reflect the current pipeline, importing from
`src/ml4b/` and orchestrating (never duplicating pipeline logic):

- `02_data_understanding.ipynb` — Kaggle Apple-Watch dataset exploration: 21
  abbreviations → full names, the 3 target classes and set counts, 100 Hz
  confirmation, single-subject + sensor-lag notes, example signals.
- `03_data_preparation.ipynb` — load (lag-trimmed, canonical units) → sliding
  window 200 @ 100 Hz → activity gate → 39 invariant features → augmentation →
  leave-one-set-out grouping.
- `04_modeling.ipynb` — Random Forest with leave-one-set-out CV (reproduces the
  committed macro F1 0.776), confusion matrix, per-class F1; does **not**
  overwrite `best_model.joblib`.
- `05_evaluation.ipynb` — honest metrics from the committed `model_metrics.json`,
  limitations, and the one-shot sanity check on `test_samples` (reported as-is).
- `06_streamlit_demo.ipynb` — end-to-end `predict_from_sensor_logger` demo that
  mirrors the app (including gated `rest` and `uncertain`).

There is no `01_business_understanding.ipynb` or `06_deployment.ipynb` in the
repo; the business-understanding and deployment phases are documented in
`docs/` instead. Each notebook is executed end-to-end and committed with outputs
stripped.

## Alternatives Considered
- **Delete the notebooks** — loses the CRISP-DM phase narrative expected by the
  course; rejected.
- **Mark them as historical and leave the RecoFit content** — keeps a visible
  contradiction with the code/app/docs; rejected (the task is consistency).
- **Keep outputs in the committed notebooks** — bloats diffs (the old `02` was
  ~800 KB of embedded plots); rejected in favour of stripped outputs.

## Rationale
The notebooks are part of the handover deliverable and must agree with the code
they document. Importing from `src/ml4b/` guarantees they stay correct as the
modules evolve, and executing them in CI-like fashion (`nbconvert --execute`)
proves they run against the real dataset and modules.

## Consequences
- **Positive:** notebooks now match the app, source, and docs (single source of
  truth); they run top-to-bottom on the Kaggle data; clean diffs.
- **Negative:** `04_modeling.ipynb` runs a full 75-fold leave-one-set-out CV
  (~5 minutes); this is the price of an honest, reproducible evaluation in the
  notebook. Running the notebooks requires the Kaggle dataset
  (`data/raw/kaggle_gym_imu/`); `05`/`06` need only the committed model + samples.
