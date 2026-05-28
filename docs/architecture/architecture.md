# Architecture Documentation — ML4B Gym Exercise Recognition
> arc42 lightweight format — mandatory sections: 1, 2, 3, 5, 6, 8

---

## 1. Goals and Requirements

### Primary Research Question
> "Can machine learning models trained on publicly available wrist-worn sensor data accurately classify gym exercises, and how well do these models generalize to new data collected from an Apple Watch during real workout sessions?"

### Primary Goal
Classify gym exercises from Apple Watch sensor streams (accelerometer + gyroscope) using supervised Machine Learning. The model must distinguish between **6 exercise classes**: bicep_curl, shoulder_press, squat, tricep_extension, lateral_raise, rest. Classes were selected data-driven based on subject coverage in the RecoFit dataset (minimum 30 participants threshold). See `docs/decisions/ADR-005-exercise-class-selection.md` for the selection rationale.

### Two-Dataset Validation Strategy
| Phase | Data Source | Purpose |
|-------|------------|---------|
| Training & validation | RecoFit (Microsoft Research) — MATLAB .mat, 50 Hz, wrist-worn, 200+ subjects | Build and validate the full ML pipeline |
| Generalization test | Self-recorded Apple Watch data via Sensor Logger app | Measure transfer to a new individual and device |

Only **Wrist Motion** (accelerometer + gyroscope) and optionally Heart Rate are used as features. Location, Barometer, Magnetometer, and Compass channels from Sensor Logger are discarded.

### Quality Goals
| Priority | Quality Goal | Scenario |
|----------|-------------|----------|
| 1 | Reproducibility | Any team member can re-run the full pipeline on a new machine with `uv sync` + `uv run` |
| 2 | Modularity | Data loading, feature engineering, and modelling are decoupled Python modules |
| 3 | Explainability | Model predictions can be explained to non-ML stakeholders via the Streamlit app |

### Performance Targets
| Criterion | Target |
|-----------|--------|
| Classification accuracy (public test set) | ≥ 80% macro-averaged |
| Generalization accuracy (Apple Watch data) | ≥ 65% macro-averaged |

### Stakeholders
| Role | Concern |
|------|---------|
| FAU ML4B Course | Correct application of CRISP-DM methodology |
| Development team | Clear project structure, reproducible environment |
| Demo audience | Usable Streamlit UI for live predictions |

---

## 2. Constraints

### Technical Constraints
| Constraint | Rationale |
|-----------|-----------|
| Python 3.11 | Course requirement; available on all target platforms |
| `uv` as package manager | Reproducible lockfile, fast installs, no conda overhead |
| Streamlit for UI | Simple, Python-native, zero frontend knowledge required |
| scikit-learn for ML | Well-documented, suited to tabular/sensor data, beginner-friendly |
| Jupyter for exploration | Standard in data science; one notebook per CRISP-DM phase |

### Organisational Constraints
| Constraint | Rationale |
|-----------|-----------|
| University project (SoSe 2026) | Scope limited to semester timeline |
| No cloud infrastructure | All processing runs locally (WSL/macOS/Windows) |

---

## 3. System Context

```
┌──────────────────────────────────────────────────────────────────┐
│                          External World                           │
│                                                                  │
│  RecoFit (Microsoft) ──► MATLAB .mat file (~2.5 GB)             │
│    200+ subjects, 50 Hz, wrist-worn acc+gyro                     │
│    Loaded via: scipy.io.loadmat(path, simplify_cells=True)       │
│                                                                  │
│  Apple Watch (Sensor Logger app) ──► CSV export                  │
│    Personal recordings for generalization test                   │
│    Channels used: Wrist Motion (acc+gyro), optionally Heart Rate │
└──────────────────────┬───────────────────────────────────────────┘
                       │ data/raw/recofit/   (training)
                       │ data/raw/personal/  (generalization test)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                        ML4B System                                │
│                                                                  │
│  notebooks/  ──► exploration & CRISP-DM documentation            │
│  src/ml4b/   ──► reusable pipeline package                       │
│  app/        ──► Streamlit prediction UI                          │
│  models/     ──► saved model artefacts (.joblib)                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
         Streamlit App (localhost:8501)
         User uploads sensor window CSV →
         receives predicted exercise label + confidence
```

**Primary training input:** RecoFit MATLAB `.mat` file parsed with `scipy.io.loadmat`. Sampling rate: 50 Hz. Sensors: accelerometer (g) + gyroscope (dps), 3 axes each. Dataset contains 75 exercise classes total; 6 target classes selected based on subject coverage analysis (>30 participants threshold — see ADR-005).  
**Generalization test input:** Self-recorded Apple Watch data via Sensor Logger app (Wrist Motion channel only).  
**Outputs:** Trained scikit-learn Pipeline (`.joblib`) + Streamlit web app for exercise prediction.

---

## 5. Building Block View

```
ml4b-project/
├── agents/                     # Claude Code specialist agent instruction files
├── app/                        # Streamlit application
│   └── streamlit_app.py        #   entry point: uv run streamlit run app/streamlit_app.py
├── src/ml4b/                   # Installable Python package
│   ├── data/                   #   data preparation pipeline (Phase 3)
│   │   ├── loader.py           #     RecoFit .mat → long-format DataFrame (filters to 6 target classes)
│   │   ├── windowing.py        #     Sliding-window segmentation (2 s, 50% overlap — ADR-006)
│   │   ├── features.py         #     Per-window statistical + FFT features (47 dims)
│   │   └── splitting.py        #     Subject-based train/val/test split (ADR-007)
│   ├── models/                 #   model training & inference (Phase 4)
│   │   ├── train.py            #     train_random_forest(), train_xgboost(), train_svm() (ADR-009)
│   │   └── evaluate.py         #     evaluate_model(), compare_models(), save_model()
│   └── utils/
│       └── config.py           #   env-based path configuration
│                               #     constants: PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR
├── notebooks/                  # One notebook per CRISP-DM phase
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment.ipynb
├── data/                       # NOT in git
│   ├── raw/                    #   original sensor recordings
│   └── processed/              #   feature-engineered datasets
├── models/
│   └── saved/                  #   serialised model files (.joblib)
├── tests/                      # pytest unit tests
└── docs/                       # Project documentation (this folder)
```

### Key Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/ml4b/data/loader.py` | Read RecoFit `.mat` via scipy.io.loadmat, flatten to a long-format DataFrame, filter to the 6 target classes via `EXERCISE_MAPPING` |
| `src/ml4b/data/windowing.py` | Segment continuous recordings into 100-sample (2 s) windows with 50% overlap, never crossing subject/exercise/recording boundaries (ADR-006) |
| `src/ml4b/data/features.py` | Extract 47 features per window: 7 statistics × 6 axes, 3 magnitude features, 2 FFT features |
| `src/ml4b/data/splitting.py` | Partition by `subject_id` into disjoint train/val/test (ADR-007); undersample `rest` class in train to `2×` largest exercise class to fix 89% imbalance (ADR-008) |
| `src/ml4b/models/train.py` | Train Random Forest, XGBoost, SVM classifiers; all use `class_weight='balanced'`; SVM wrapped in `Pipeline` with `StandardScaler` (ADR-009) |
| `src/ml4b/models/evaluate.py` | Compute accuracy, macro F1, per-class F1, confusion matrix; save plots to `reports/figures/`; serialise best model with `joblib.dump` |
| `src/ml4b/utils/config.py` | Centralised path resolution via env vars (PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR) |
| `app/streamlit_app.py` | Streamlit UI: file upload → feature extraction → prediction |
| `notebooks/` | CRISP-DM phase documentation and exploratory analysis |
| `agents/` | Claude Code specialist agent instruction files |
| `reports/figures/` | Generated evaluation plots and result figures |

---

## 6. Runtime View

### Training Pipeline (Phase 3 → Phase 4)
```
data/raw/recofit/exercise_data.50.0000_singleonly.mat   (2.5 GB MATLAB file)
    │
    ▼ src/ml4b/data/loader.py  — load_recofit_raw()
    │   scipy.io.loadmat(path, simplify_cells=True)
    │   Iterate (n_subjects × n_exercises) cell matrix
    │   Apply EXERCISE_MAPPING → keep only the 6 target classes
Long DataFrame: subject_id, exercise_name, recording_id, timestamp, ax, ay, az, gx, gy, gz
    │
    ▼ src/ml4b/data/windowing.py  — apply_sliding_window(size=100, overlap=0.5)
    │   Per (subject_id, exercise_name, recording_id) group
    │   2 s windows at 50 Hz, 50% overlap (ADR-006)
Window DataFrame: one row per window, raw_* columns hold lists of 100 samples
    │
    ▼ src/ml4b/data/features.py  — extract_features()
    │   Per axis: mean, std, min, max, range, RMS, zero-crossing rate
    │   Magnitudes: accel_magnitude_{mean,std}, gyro_magnitude_mean
    │   FFT on accel magnitude: dominant_frequency, spectral_energy
Feature matrix: 47 numeric features + (subject_id, exercise_name, window_id)
    │
    ▼ src/ml4b/data/splitting.py  — subject_based_split(test=0.2, val=0.1, seed=42)
    │   No subject appears in more than one split (ADR-007)
    │
    ▼ src/ml4b/data/splitting.py  — undersample_majority_class(multiplier=2.0)  [TRAIN ONLY]
    │   rest class capped at 2× largest exercise class to fix 89% imbalance (ADR-008)
    │   Val and test keep original distribution for honest evaluation
Three CSVs in data/processed/: train_features.csv (balanced), val_features.csv, test_features.csv
                                + feature_names.txt
    │
    ▼ Phase 4 — src/ml4b/models/train.py  — train_random_forest() / train_xgboost() / train_svm()
    │   Three classifiers compared by macro F1 on val set (ADR-009)
    │   SVM wrapped in sklearn Pipeline with StandardScaler
Trained classifiers evaluated via src/ml4b/models/evaluate.py — evaluate_model() / compare_models()
    │
    ▼ models/saved/best_model.joblib  (joblib.dump — loaded by Streamlit app)
```

### Prediction Flow (Streamlit)
```
User uploads CSV window via browser
    │
    ▼ app/main.py — feature extraction (same as training)
Feature vector (1 × n_features)
    │
    ▼ model.predict()
Exercise label + confidence score
    │
    ▼ Streamlit UI — result displayed to user
```

---

## 8. Cross-Cutting Concepts

### Reproducibility
- `uv.lock` pins every transitive dependency exactly.
- `uv sync` restores the environment in full on any machine.
- Random seeds are set explicitly in all model training code.

### Path Handling
- All file paths are `pathlib.Path` objects resolved relative to `_PROJECT_ROOT`.
- `src/ml4b/utils/config.py` is the single source of truth for paths.
- No hardcoded absolute paths anywhere in the codebase.

### Environment Configuration
- `.env` (never committed) overrides defaults for data/model directories.
- `.env.example` documents all available variables.

### Code Quality
- Formatter: `ruff format` (88-char line length).
- Linter: `ruff check` (pycodestyle + pyflakes + isort).
- Type checker: `mypy`.
- All public functions and classes have Google-style docstrings and type hints.

### Testing
- `pytest` with test files mirroring the `src/ml4b/` structure.
- No mocking of file I/O — tests use `tmp_path` fixtures with real files.
