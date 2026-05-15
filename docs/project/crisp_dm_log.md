# CRISP-DM Progress Log

Project: ML4B SoSe 2026 — Gym Exercise Recognition  
Team: Anshul Agrawal  
Last updated: 2026-05-15
Phase 1 completed: 2026-05-15

---

## Phase Overview

| # | Phase | Status | Responsible | Notebook | Notes |
|---|-------|--------|-------------|----------|-------|
| 1 | Business Understanding | done | Anshul Agrawal | `notebooks/01_business_understanding.ipynb` | Research question defined, 7 exercise classes selected, two-dataset validation strategy decided, personal test data via Sensor Logger app on Apple Watch |
| 2 | Data Understanding | in progress | Anshul Agrawal | `notebooks/02_data_understanding.ipynb` | Dataset evaluation complete. RecoFit selected (wrist, 50Hz, 200+ participants, acc+gyro). Full class list pending notebook exploration. scipy added as dependency. |
| 3 | Data Preparation | todo | — | `notebooks/03_data_preparation.ipynb` | Windowing, feature engineering, train/test split |
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
- **Deliverable:** `data/processed/features.csv`, preprocessing pipeline in `src/ml4b/data/`

### Phase 4 — Modeling
- **Goal:** Fit baseline (k-NN or Decision Tree), then tune (Random Forest, SVM, XGBoost)
- **Deliverable:** Trained pipelines saved to `models/saved/`, comparison table in notebook

### Phase 5 — Evaluation
- **Goal:** Evaluate best model on held-out test set, analyse confusion matrix, per-exercise F1
- **Deliverable:** Evaluation report in notebook, updated `docs/architecture/architecture.md`

### Phase 6 — Deployment
- **Goal:** Wrap model in Streamlit app, demo live predictions from uploaded CSV window
- **Deliverable:** Working `app/main.py`, final demo recording
