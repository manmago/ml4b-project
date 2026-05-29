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

## 3. How to Run the Project (Quickstart — 3 Commands)

```bash
git clone git@github.com:AnshulAgrawal7/ml4b-project.git
cd ml4b-project
uv sync
uv run streamlit run app/streamlit_app.py
```

→ Open **http://localhost:8501**

**No dataset download needed** — the trained model
(`models/saved/best_model.joblib`) and feature list
(`data/processed/feature_names.txt`) are committed to git. The dataset is only
required if you want to **retrain** the model (`uv run python
scripts/train_model.py`).

For OS-specific setup: see `docs/setup/Setup_WSL_Windows.md`,
`docs/setup/Setup_macOS.md`, `docs/setup/Setup_Windows.md`.

---

## 4. How the ML Pipeline Works

The pipeline has 5 steps — each step has a corresponding notebook and source module:

**Step 1 — Load raw data**
- Input:  `data/raw/mm-fit/` smartwatch `.npy` files, both wrists (~1.7 GB) — ADR-013
- Code:   `src/ml4b/data/mmfit_loader.py` (+ `scripts/build_mmfit_dataset.py`)
- Output: long-format DataFrame (same schema as the original RecoFit loader)
- Note:   RecoFit (`loader.py`) was the original Phase 1–5 source, superseded
          because its sensor was forearm-worn while the Apple Watch is wrist-worn

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
| Training dataset | MM-Fit (wrist-worn smartwatch — ADR-013) |
| Test Macro F1 | 0.944 ✅ (target: ≥ 0.80) |
| Test Accuracy | 97.8% |
| Val Macro F1 | 0.866 |
| Best class | push_up (F1 = 1.00), rest (0.99) |
| Weakest class | squat (0.84) / bicep_curl (0.86) |
| Apple Watch test | push_up recognized ✅; bicep_curl still confused with tricep_extension (ADR-013/014) |

---

## 6. Key Decisions Made (and Why)

Every decision has a full ADR in `docs/decisions/`. Here is a plain-language summary:

| Decision | What we chose | Why | ADR |
|----------|--------------|-----|-----|
| Package manager | uv | Faster than pip/conda, reproducible across all OS | ADR-001 |
| ML framework | scikit-learn | Sufficient for tabular features, easy to use | ADR-002 |
| Multi-agent AI setup | data_scientist + documenter + reviewer agents | Better quality, less token waste | ADR-003 |
| Code documentation standard | Google docstrings + inline comments | New team must understand code without asking | ADR-004 |
| Exercise class selection | 7 classes (6 from RecoFit + push_up from MM-Fit) | Data-driven coverage (ADR-005); push_up added with MM-Fit | ADR-005, ADR-013 |
| Training dataset switch | RecoFit → MM-Fit | RecoFit was forearm-worn; Apple Watch is wrist-worn — MM-Fit matches the device | ADR-013 |
| Rotation augmentation | Rejected (kept off) | Hurt the bicep/tricep case and in-domain F1 | ADR-014 |
| RF regularization + rest rebalancing | depth≤20, leaf=4, rest mult 1.5 | Reduce overfitting (full-depth trees) and over-prediction of `rest` | ADR-015 |
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
| OS-specific setup guides | `docs/setup/Setup_WSL_Windows.md`, `docs/setup/Setup_macOS.md`, `docs/setup/Setup_Windows.md` |
| Data exploration | `notebooks/02_data_understanding.ipynb` |
| Data preparation pipeline | `notebooks/03_data_preparation.ipynb` |
| Model training and comparison | `notebooks/04_modeling.ipynb` |
| Final evaluation results | `notebooks/05_evaluation.ipynb` |
| Reusable ML code | `src/ml4b/` |
| Trained model | `models/saved/best_model.joblib` |
| Streamlit app | `app/streamlit_app.py` |

---

## 8. Project Status

| Task | Status | Notes |
|------|--------|-------|
| Streamlit App | ✅ Done | 3 pages: Home, Predict Exercise, Model Performance. Accepts `WristMotion.csv` and ZIP uploads. Verified via Streamlit `AppTest`. |
| WristMotion.csv column mapping | ✅ Done | Loader auto-detects 4 Sensor Logger formats and ZIP exports — see Section 8b. |
| Trained model in git | ✅ Done | `best_model.joblib` + `feature_names.txt` committed — app runs without the dataset. |
| Apple Watch generalization test | ⏳ Pending | Optional: record a real gym session with Sensor Logger and evaluate. |

---

## 8b. Sensor Logger Data Format

The app accepts exports from the **Sensor Logger** iOS app (free). Upload either:

- the single **`WristMotion.csv`** file, **or**
- the **full ZIP** of the export (the app finds `WristMotion.csv` inside).

`src/ml4b/data/apple_watch_loader.py` auto-detects the column format and
normalizes everything to `[timestamp, ax, ay, az, gx, gy, gz]`. Supported
source formats:

| Format | Columns |
|--------|---------|
| A — default WristMotion.csv | `time, seconds_elapsed, x, y, z, roll, pitch, yaw` |
| B — pre-normalized | `timestamp, ax, ay, az, gx, gy, gz` |
| C — seconds_elapsed variant | `seconds_elapsed, x, y, z, roll, pitch, yaw` |
| D — DeviceMotion export | `time, accelerationX/Y/Z, rotationRateX/Y/Z` |

Full collection + export protocol:
`docs/project/apple_watch_data_collection_guide.md`.

---

## 9. Dataset

**MM-Fit (current training source — ADR-013)**

- Wrist-worn smartwatch (Mobvoi TicWatch Pro), acc+gyro at 100 Hz, both wrists
- 20 workouts, 10 gym exercises (7 mapped to our classes + `push_up`)
- Download: https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip (~1.7 GB)
- Project page: https://mmfit.github.io/ · Citation: Strömbäck, Huang & Radu (2020), UbiComp/ISWC · CC-BY-4.0
- NOT included in git — unzip to `data/raw/mm-fit/`

**RecoFit (Microsoft Research) — original source, superseded (ADR-013)**

- 200+ participants, **forearm**-worn sensor, 50 Hz
- Download: https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors
- Citation: Morris et al. (2014), CHI · Replaced because the Apple Watch is wrist-worn
