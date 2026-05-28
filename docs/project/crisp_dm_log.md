# CRISP-DM Progress Log

Project: ML4B SoSe 2026 — Gym Exercise Recognition  
Team: Anshul Agrawal  
Last updated: 2026-05-28
Phase 1 completed: 2026-05-15
Phase 2 completed: 2026-05-22
Phase 3 started:   2026-05-28

---

## Phase Overview

| # | Phase | Status | Responsible | Notebook | Notes |
|---|-------|--------|-------------|----------|-------|
| 1 | Business Understanding | done | Anshul Agrawal | `notebooks/01_business_understanding.ipynb` | Research question defined, exercise classes defined (refined to 6 in Phase 2), two-dataset validation strategy decided, personal test data via Sensor Logger app on Apple Watch |
| 2 | Data Understanding | done | Anshul Agrawal | `notebooks/02_data_understanding.ipynb` | 75 exercise classes found in RecoFit. 6 final classes selected data-driven based on subject coverage (>30 participants threshold). Class mapping confirmed. See ADR-005. |
| 3 | Data Preparation | in progress | Anshul Agrawal | `notebooks/03_data_preparation.ipynb` | Pipeline modules created: `loader.py`, `windowing.py`, `features.py`, `splitting.py`. ADR-006 (window 2 s, 50% overlap) and ADR-007 (subject-based split) accepted. Notebook ready to run end-to-end. |
| 4 | Modeling | todo | — | `notebooks/04_modeling.ipynb` | Baseline + tuned classifiers, cross-validation |
| 5 | Evaluation | todo | — | `notebooks/05_evaluation.ipynb` | Confusion matrix, per-class metrics, error analysis |
| 6 | Deployment | todo | — | `notebooks/06_deployment.ipynb` | Streamlit app, model serialisation, demo |

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

### Phase 4 — Modeling
- **Goal:** Fit baseline (k-NN or Decision Tree), then tune (Random Forest, SVM, XGBoost)
- **Deliverable:** Trained pipelines saved to `models/saved/`, comparison table in notebook

### Phase 5 — Evaluation
- **Goal:** Evaluate best model on held-out test set, analyse confusion matrix, per-exercise F1
- **Deliverable:** Evaluation report in notebook, updated `docs/architecture/architecture.md`

### Phase 6 — Deployment
- **Goal:** Wrap model in Streamlit app, demo live predictions from uploaded CSV window
- **Deliverable:** Working `app/main.py`, final demo recording
