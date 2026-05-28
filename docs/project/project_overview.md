# Project Overview — ML4B Gym Exercise Recognition

> **Read this first.** This document explains what this project does, how it works,
> what decisions were made and why, and where to find everything in the repository.
> A new team member should be able to understand the full project from this document alone.

---

## 1. What Does This Project Do?

This project automatically recognizes gym exercises from wrist-worn sensor data
(Apple Watch accelerometer and gyroscope) using machine learning.

**In plain language:**
You go to the gym, record your workout with the Sensor Logger app on your Apple Watch,
upload the CSV file to our Streamlit web app, and the app tells you which exercise
you performed in each time window — with a confidence score.

**The 6 exercises the model can recognize:**
1. Bicep Curl
2. Shoulder Press
3. Squat
4. Tricep Extension
5. Lateral Raise
6. Rest / No Exercise

---

## 2. Project Context

- **Course:** ML4B (Machine Learning for Business), SoSe 2026
- **University:** FAU Nürnberg, Lehrstuhl für Wirtschaftsinformatik
- **Methodology:** CRISP-DM (Cross-Industry Standard Process for Data Mining)
- **Team:** Anshul Agrawal + 2 teammates
- **Final deliverable:** Streamlit web application presented at Schaeffler, Nürnberg

---

## 3. How to Run the Project (Quickstart)

```bash
# 1. Clone the repo
git clone git@github.com:AnshulAgrawal7/ml4b-project.git
cd ml4b-project

# 2. Install dependencies (one command)
uv sync

# 3. Download the dataset (see data/raw/recofit/README.md)
# Place .mat files in data/raw/recofit/

# 4. Run the Streamlit app
uv run streamlit run app/streamlit_app.py
```

For OS-specific setup: see `Setup_WSL_Windows.md`, `Setup_macOS.md`, `Setup_Windows.md`

---

## 4. How the ML Pipeline Works

The pipeline has 5 steps — each step has a corresponding notebook and source module:

**Step 1 — Load raw data**
- Input:  `data/raw/recofit/exercise_data.50.0000_singleonly.mat` (2.5 GB)
- Code:   `src/ml4b/data/loader.py`
- Output: pandas DataFrame with 7.96M sensor samples

**Step 2 — Sliding Window Segmentation**
- Input:  Raw DataFrame
- Code:   `src/ml4b/data/windowing.py`
- Params: window_size=100 (2 seconds at 50Hz), overlap=50%
- Output: 155,598 windows
- Why:    Each 2-second window becomes one training sample

**Step 3 — Feature Extraction**
- Input:  Windows
- Code:   `src/ml4b/data/features.py`
- Output: 47 features per window (mean, std, min, max, RMS, FFT per axis)
- Why:    ML models need fixed-size vectors, not raw time series

**Step 4 — Train/Val/Test Split**
- Input:  Feature matrix (155,598 × 47)
- Code:   `src/ml4b/data/splitting.py`
- Method: Subject-based split (no subject appears in both train and test)
- Output: train (21,490), val (13,888), test (30,096)
- Why:    Subject-based prevents data leakage — see ADR-007

**Step 5 — Model Training**
- Input:  `train_features.csv`
- Code:   `src/ml4b/models/train.py`
- Models: Random Forest ✅ (best), XGBoost, SVM
- Output: `models/saved/best_model.joblib`

**Step 6 — Prediction (App)**
- Input:  New CSV from Sensor Logger (Apple Watch)
- Code:   `src/ml4b/data/apple_watch_loader.py`
- Output: Predicted exercise per 2-second window + confidence

---

## 5. Key Results

| Metric | Value |
|--------|-------|
| Best model | Random Forest |
| Test Macro F1 | 0.8006 ✅ (target: ≥ 0.80) |
| Test Accuracy | 96.3% |
| Generalization gap | 1.3% (very stable) |
| Best class | rest (F1 = 0.98) |
| Weakest class | lateral_raise (F1 = 0.55) |
| Apple Watch test | ⏳ Pending — data not yet collected |

---

## 6. Key Decisions Made (and Why)

Every decision has a full ADR in `docs/decisions/`. Here is a plain-language summary:

| Decision | What we chose | Why | ADR |
|----------|--------------|-----|-----|
| Package manager | uv | Faster than pip/conda, reproducible across all OS | ADR-001 |
| ML framework | scikit-learn | Sufficient for tabular features, easy to use | ADR-002 |
| Multi-agent AI setup | data_scientist + documenter + reviewer agents | Better quality, less token waste | ADR-003 |
| Code documentation standard | Google docstrings + inline comments | New team must understand code without asking | ADR-004 |
| Exercise class selection | 6 classes (not original 7) | Data-driven: Bench Press + Deadlift had <30 participants | ADR-005 |
| Sliding window size | 2 seconds (100 samples) | Captures one full rep phase; consistent with literature | ADR-006 |
| Train/test split method | Subject-based | Prevents data leakage between train and test | ADR-007 |
| Class imbalance fix | Undersampling rest to 2× largest class | rest was 89% of data — model would always predict rest | ADR-008 |
| Final model | Random Forest | Best macro F1 (0.8136 val), fast, interpretable | ADR-010 |

---

## 7. Where to Find Everything

| What you need | Where to look |
|--------------|---------------|
| Project goals and research question | `docs/business_understanding/business_understanding.md` |
| Dataset evaluation and selection | `docs/data_understanding/dataset_evaluation.md` |
| All technical decisions with rationale | `docs/decisions/ADR-001` through `ADR-010` |
| CRISP-DM progress log | `docs/project/crisp_dm_log.md` |
| System architecture | `docs/architecture/architecture.md` |
| How to collect Apple Watch data | `docs/project/apple_watch_data_collection_guide.md` |
| Folder and file descriptions | `STRUCTURE.md` |
| OS-specific setup guides | `Setup_WSL_Windows.md`, `Setup_macOS.md`, `Setup_Windows.md` |
| Data exploration | `notebooks/02_data_understanding.ipynb` |
| Data preparation pipeline | `notebooks/03_data_preparation.ipynb` |
| Model training and comparison | `notebooks/04_modeling.ipynb` |
| Final evaluation results | `notebooks/05_evaluation.ipynb` |
| Reusable ML code | `src/ml4b/` |
| Trained model | `models/saved/best_model.joblib` |
| Streamlit app | `app/streamlit_app.py` |

---

## 8. What Still Needs to Be Done

| Task | Status | Notes |
|------|--------|-------|
| Streamlit App | ✅ Phase 6 complete | Home, Predict Exercise, Model Performance pages implemented |
| Apple Watch generalization test | ⏳ Pending | Record gym session with Sensor Logger |
| WristMotion.csv column mapping | ⏳ Pending | Need to verify Sensor Logger column names match pipeline |

---

## 9. Dataset

**RecoFit (Microsoft Research)**

- 200+ participants, wrist-worn sensor, 50Hz
- Download: https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors
- Citation: Morris et al. (2014), CHI Conference
- NOT included in git (2.5 GB) — see `data/raw/recofit/README.md`
