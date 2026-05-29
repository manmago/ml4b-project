# CRISP-DM Progress Log

Project: ML4B SoSe 2026 — Gym Exercise Recognition  
Team: Anshul Agrawal  
Last updated: 2026-05-29
Phase 1 completed: 2026-05-15
Phase 2 completed: 2026-05-22
Phase 3 completed: 2026-05-28
Phase 4 completed: 2026-05-28
Phase 5 started:   2026-05-28
Phase 5 completed: 2026-05-28
Phase 6 started:   2026-05-28
Phase 6 completed: 2026-05-29

---

## Phase Overview

| # | Phase | Status | Responsible | Notebook | Notes |
|---|-------|--------|-------------|----------|-------|
| 1 | Business Understanding | done | Anshul Agrawal | `notebooks/01_business_understanding.ipynb` | Research question defined, exercise classes defined (refined to 6 in Phase 2), two-dataset validation strategy decided, personal test data via Sensor Logger app on Apple Watch |
| 2 | Data Understanding | done | Anshul Agrawal | `notebooks/02_data_understanding.ipynb` | 75 exercise classes found in RecoFit. 6 final classes selected data-driven based on subject coverage (>30 participants threshold). Class mapping confirmed. See ADR-005. |
| 3 | Data Preparation | done | Anshul Agrawal | `notebooks/03_data_preparation.ipynb` | Pipeline modules created: `loader.py`, `windowing.py`, `features.py`, `splitting.py`. ADR-006 (window 2 s, 50% overlap) and ADR-007 (subject-based split) accepted. Notebook ready to run end-to-end. |
| 4 | Modeling | done | Anshul Agrawal | `notebooks/04_modeling.ipynb` | Random Forest selected as best model (macro F1 = 0.8136 on val). XGBoost = 0.8057, SVM = 0.7478. ADR-009, ADR-010 accepted. best_model.joblib saved. |
| 5 | Evaluation | done | Anshul Agrawal | `notebooks/05_evaluation.ipynb` | Test Macro F1 = 0.8006 ✅ target met. Generalization gap 1.3%. No iterative improvement needed. See notebooks/05_evaluation.ipynb for full results. |
| 6 | Deployment | done | Anshul Agrawal | `notebooks/06_deployment.ipynb` | Full Streamlit app (3 pages) live; model + feature names committed to git; `scripts/train_model.py` added; Sensor Logger CSV/ZIP upload pipeline working. App runs with 3 commands, no dataset needed. |
| ↻ | **Iteration 2 (CRISP-DM loop): dataset switch + tuning** | done | Anshul Agrawal | `scripts/build_mmfit_dataset.py` | Deployment revealed RecoFit (forearm-worn) does not generalize to the wrist-worn Apple Watch. Re-ran Data Understanding → Modeling → Evaluation on **MM-Fit** (wrist-worn smartwatch). 7 classes (added `push_up`). Then regularized RF + rebalanced `rest` to cut over-fit trees and `rest` over-prediction. **Val macro F1 = 0.866, Test macro F1 = 0.944.** ADR-013 (dataset), ADR-014 (augmentation rejected), ADR-015 (regularization + rest rebalancing). |

---

## Detailed Notes

### Phase 1 — Business Understanding
- **Goal:** Define what "gym exercise recognition" means precisely (which exercises? which users? what accuracy threshold?)
- **Deliverable:** Project charter, success criteria, CRISP-DM plan

### Phase 2 — Data Understanding
- **Goal:** Load raw sensor data, visualise per-axis signals per exercise, identify quality issues
- **Deliverable:** EDA notebook, `docs/data/data_dictionary.md` filled in

### Phase 3 — Data Preparation
- **Goal:** Sliding-window segmentation, feature extraction, label encoding, train/test split
- **Deliverable:** `data/processed/{train,val,test}_features.csv` + `feature_names.txt`, preprocessing pipeline in `src/ml4b/data/`
- **Status (2026-05-28):**
  - `src/ml4b/data/loader.py` — RecoFit `.mat` parser, filters to 6 target classes via `EXERCISE_MAPPING`
  - `src/ml4b/data/windowing.py` — 100-sample (2 s) sliding windows with 50% overlap, never crossing subject/recording boundaries (ADR-006)
  - `src/ml4b/data/features.py` — 47 features per window: 7 statistics × 6 axes + 3 magnitude + 2 FFT features
  - `src/ml4b/data/splitting.py` — subject-disjoint train/val/test split (ADR-007)
  - `notebooks/03_data_preparation.ipynb` — orchestrates the full pipeline and asserts no subject overlap / no NaN-inf
  - ADRs accepted: `ADR-006-sliding-window-parameters.md`, `ADR-007-subject-based-train-test-split.md`
  - **Class imbalance detected:** `rest` = 88.8% of windows. Fixed with `undersample_majority_class(multiplier=2.0)` on train set only. `class_weight='balanced'` planned for all Phase 4 models. Primary metric: macro-averaged F1. See ADR-008.

### Phase 4 — Modeling
- **Goal:** Train and compare three classical ML classifiers; select best by macro F1 on validation set
- **Deliverable:** Trained pipelines saved to `models/saved/`, comparison table in notebook
- **Status (2026-05-28):**
  - `src/ml4b/models/train.py` — `train_random_forest()`, `train_xgboost()`, `train_svm()` (SVM wrapped in sklearn Pipeline with StandardScaler)
  - `src/ml4b/models/evaluate.py` — `evaluate_model()`, `compare_models()`, `save_model()`; primary metric is macro F1; confusion matrices saved to `reports/figures/`
  - `notebooks/04_modeling.ipynb` — full pipeline from data loading to best model serialisation; ready to run
  - ADR accepted: `ADR-009-model-selection-rationale.md` — RF baseline, XGBoost, SVM selected; Deep Learning, k-NN, Logistic Regression rejected with rationale
  - `xgboost>=2.0` added to `pyproject.toml`; `uv sync` updated
  - **Results (validation set):** Random Forest macro F1 = 0.8136 (best), XGBoost = 0.8057, SVM = 0.7478
  - ADR accepted: `ADR-010-random-forest-as-final-model.md`; `best_model.joblib` saved

### Phase 5 — Evaluation
- **Goal:** Final unbiased test set evaluation; Apple Watch generalization test (pending data collection)
- **Deliverable:** `notebooks/05_evaluation.ipynb`, `src/ml4b/data/apple_watch_loader.py`, data collection guide
- **Status (2026-05-28 — COMPLETE):**
  - `src/ml4b/data/apple_watch_loader.py` — `load_sensor_logger_csv()`, `predict_from_sensor_logger()` — handles Sensor Logger CSV format variations, unit conversion (m/s² → g), dummy metadata for inference
  - `notebooks/05_evaluation.ipynb` — test set evaluation, val vs test comparison, error analysis, Apple Watch placeholder (Cell 14)
  - `docs/project/apple_watch_data_collection_guide.md` — recording protocol, export instructions, troubleshooting
  - `data/raw/apple_watch/` — directory created for future Apple Watch CSV files
  - Apple Watch generalization test: **PENDING** — data not yet collected
  - **Final results:** Test Accuracy = 0.9630, Test Macro F1 = 0.8006 ✅ (target ≥ 0.80 met)
  - **Generalization gap:** 1.3% (val 0.8136 → test 0.8006) — model does not overfit
  - **Best class:** rest (F1 = 0.98); **Weakest class:** lateral_raise (F1 = 0.55)
  - **Decision:** No iterative improvement — target met, Phase 6 is higher priority

### Phase 6 — Deployment
- **Goal:** Wrap model in Streamlit app, demo live predictions from uploaded CSV window
- **Deliverable:** Working `app/streamlit_app.py`, final demo recording
- **Status (2026-05-29 — COMPLETE):**
  - `app/streamlit_app.py` — entry point: cached model loading, sidebar navigation, routes to 3 pages
  - `app/pages/home.py` — overview, headline metrics, Sensor Logger collection instructions
  - `app/pages/prediction.py` — `WristMotion.csv` **and** ZIP upload, full pipeline, timeline chart, distribution pie, results table, CSV download, graceful error handling
  - `app/pages/model_performance.py` — Phase 5 metrics, model comparison, per-class F1 bar chart, row-normalized confusion matrix, key findings
  - `app/__init__.py`, `app/pages/__init__.py`, `.streamlit/config.toml` (single clean sidebar nav)
  - `src/ml4b/data/apple_watch_loader.py` — rewritten: auto-detects 4 Sensor Logger column formats, ZIP support, `predict_from_sensor_logger()` shared with training
  - `scripts/train_model.py` — reproducible one-shot training (load → window → features → split → train → save)
  - `models/saved/best_model.joblib`, `models/saved/random_forest.joblib`, `data/processed/feature_names.txt` — committed to git so the app runs after a fresh clone with **no dataset download**
  - All 3 pages verified via Streamlit `AppTest` (zero exceptions); `uv run ruff check .` passes
  - **Final results (unchanged from Phase 5):** Test Accuracy = 0.9630, Test Macro F1 = 0.8006 ✅, generalization gap 1.3%

---

## Iteration 2 — Dataset Switch to MM-Fit (2026-05-29)

**Trigger (CRISP-DM is a loop):** During Deployment, real Apple Watch recordings
were misclassified. After fixing the Sensor Logger column mapping, the 100→50 Hz
rate and the units (ADR-012), bicep curls were *still* wrong. Diagnosis with
`scripts/test_apple_watch_prediction.py` showed the cause was not a bug but a
**sensor-placement domain gap**: RecoFit's sensor was a **forearm** armband,
while the Apple Watch is on the **wrist**.

**Action — re-ran the relevant CRISP-DM phases on a wrist-worn dataset:**
- **Data Understanding/Preparation:** adopted **MM-Fit** (wrist-worn smartwatch,
  CC-BY-4.0). New `src/ml4b/data/mmfit_loader.py` emits the same long-format
  schema, so windowing + features are reused unchanged. New
  `scripts/build_mmfit_dataset.py` writes the standard processed CSVs. Both
  wrists used; 100→50 Hz decimation kept; 7 classes (added `push_up`); MM-Fit's
  official workout-id split. See ADR-013.
- **Modeling/Evaluation:** Random Forest retained (ADR-010), then regularized +
  `rest` rebalanced (ADR-015) to reduce over-fit trees and `rest`
  over-prediction. **Val macro F1 = 0.866 / acc 0.938; Test macro F1 = 0.944 /
  acc 0.978** on held-out workouts — all 7 classes ≥ 0.84 F1.
- **Unit re-alignment:** `apple_watch_loader.py` now matches MM-Fit units
  (accel m/s² incl. gravity, gyro rad/s); diagnostic z-scores fell from >10 to <2.
- **Rotation augmentation** tried to close the residual orientation gap but
  **rejected** by evidence (hurt the bicep/tricep case, lowered in-domain F1) —
  ADR-014. Module kept, off by default.

**Real Apple Watch result:** `push_up` now recognized correctly (was impossible
before); `bicep_curl` still confused with `tricep_extension` — a residual
device-orientation gap plus the biomechanical similarity of the two movements at
the wrist. Robust fix would require a few of the user's own labelled recordings.

---

## Final Project Summary (2026-05-29)

All six CRISP-DM phases are complete, plus a second iteration that switched the
training dataset to MM-Fit (ADR-013). The deliverable is a working Streamlit app
that runs with three commands (`git clone`, `uv sync`, `uv run streamlit run
app/streamlit_app.py`) and needs no dataset download. The best model (Random
Forest, trained on MM-Fit, **Test Macro F1 = 0.944**) is committed alongside the
feature list and reproducible build + training scripts. Sensor Logger (Apple
Watch) exports are accepted as either `WristMotion.csv` or a full ZIP. The
project is handover-ready: every decision is documented in `docs/decisions/`
(ADR-001–015), and OS-specific setup guides cover WSL, macOS, and Windows.
