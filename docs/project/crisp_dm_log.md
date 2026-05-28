# CRISP-DM Progress Log

Project: ML4B SoSe 2026 — Gym Exercise Recognition  
Team: Anshul Agrawal  
Last updated: 2026-05-28
Phase 1 completed: 2026-05-15
Phase 2 completed: 2026-05-22
Phase 3 completed: 2026-05-28
Phase 4 completed: 2026-05-28
Phase 5 started:   2026-05-28
Phase 5 completed: 2026-05-28
Phase 6 started:   2026-05-28

---

## Phase Overview

| # | Phase | Status | Responsible | Notebook | Notes |
|---|-------|--------|-------------|----------|-------|
| 1 | Business Understanding | done | Anshul Agrawal | `notebooks/01_business_understanding.ipynb` | Research question defined, exercise classes defined (refined to 6 in Phase 2), two-dataset validation strategy decided, personal test data via Sensor Logger app on Apple Watch |
| 2 | Data Understanding | done | Anshul Agrawal | `notebooks/02_data_understanding.ipynb` | 75 exercise classes found in RecoFit. 6 final classes selected data-driven based on subject coverage (>30 participants threshold). Class mapping confirmed. See ADR-005. |
| 3 | Data Preparation | done | Anshul Agrawal | `notebooks/03_data_preparation.ipynb` | Pipeline modules created: `loader.py`, `windowing.py`, `features.py`, `splitting.py`. ADR-006 (window 2 s, 50% overlap) and ADR-007 (subject-based split) accepted. Notebook ready to run end-to-end. |
| 4 | Modeling | done | Anshul Agrawal | `notebooks/04_modeling.ipynb` | Random Forest selected as best model (macro F1 = 0.8136 on val). XGBoost = 0.8057, SVM = 0.7478. ADR-009, ADR-010 accepted. best_model.joblib saved. |
| 5 | Evaluation | done | Anshul Agrawal | `notebooks/05_evaluation.ipynb` | Test Macro F1 = 0.8006 ✅ target met. Generalization gap 1.3%. No iterative improvement needed. See notebooks/05_evaluation.ipynb for full results. |
| 6 | Deployment | in progress | Anshul Agrawal | `notebooks/06_deployment.ipynb` | Streamlit app, model serialisation, demo |

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
- **Status (2026-05-28 — IN PROGRESS):**
  - `app/streamlit_app.py` — full implementation: model + feature names loaded at startup via `st.cache_resource`, page routing to three pages
  - `app/pages/home.py` — Home page: project overview, key metrics, step-by-step instructions, per-class F1 cards
  - `app/pages/prediction.py` — Predict Exercise: CSV upload → `predict_from_sensor_logger()` → workout summary metrics + timeline chart + results table
  - `app/pages/model_performance.py` — Model Performance: overall metrics, model comparison table, per-class F1 bar chart, confusion matrix heatmap, key findings
