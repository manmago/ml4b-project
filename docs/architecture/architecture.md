# Architecture Documentation — ML4B Gym Exercise Recognition
> arc42 lightweight format — mandatory sections: 1, 2, 3, 5, 6, 8

---

## 1. Goals and Requirements

### Primary Research Question
> "Can machine learning models trained on publicly available wrist-worn sensor data accurately classify gym exercises, and how well do these models generalize to new data collected from an Apple Watch during real workout sessions?"

### Primary Goal
Classify gym exercises from Apple Watch sensor streams (accelerometer + gyroscope) using supervised Machine Learning. The model distinguishes **7 exercise classes**: bicep_curl, shoulder_press, squat, tricep_extension, lateral_raise, push_up, rest. The original 6 classes were selected from RecoFit by subject coverage (ADR-005); `push_up` was added when the training dataset switched to MM-Fit (ADR-013). See `docs/decisions/ADR-005-exercise-class-selection.md` and `ADR-013-switch-training-dataset-to-mmfit.md`.

### Two-Dataset Validation Strategy
| Phase | Data Source | Purpose |
|-------|------------|---------|
| Training & validation | MM-Fit — wrist-worn smartwatch, 100→50 Hz, acc+gyro (ADR-013); originally RecoFit (forearm), superseded | Build and validate the full ML pipeline |
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
│  MM-Fit ──► NumPy .npy per workout (~1.7 GB)  [CURRENT — ADR-013]│
│    wrist-worn smartwatch, 100→50 Hz, acc+gyro, both wrists       │
│    Loaded via: src/ml4b/data/mmfit_loader.py                     │
│    (RecoFit forearm .mat = original Phase 1–5 source, superseded)│
│                                                                  │
│  Apple Watch (Sensor Logger app) ──► WristMotion.csv or ZIP      │
│    Live app input AND personal generalization test               │
│    Channels used: Wrist Motion (acc+gyro), optionally Heart Rate │
└──────────────────────┬───────────────────────────────────────────┘
                       │ data/raw/mm-fit/    (training — ADR-013)
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

**Primary training input:** MM-Fit smartwatch `.npy` files parsed with `src/ml4b/data/mmfit_loader.py` (both wrists). Recorded at 100 Hz, decimated to 50 Hz. Sensors: accelerometer (m/s², gravity incl.) + gyroscope (rad/s), 3 axes each. 7 target classes (6 original + `push_up`). Sensor placement (wrist) matches the Apple Watch — see ADR-013. The original RecoFit `.mat` source (forearm, ADR-005) was superseded.  
**Live app input:** Sensor Logger (Apple Watch) export — either `WristMotion.csv` or the full ZIP — uploaded on the Streamlit Predict page. The loader auto-detects 4 column formats and normalizes to `[timestamp, ax, ay, az, gx, gy, gz]`. The same files double as the self-recorded generalization-test input.  
**Outputs:** Trained scikit-learn Pipeline (`.joblib`) + Streamlit web app for exercise prediction.

---

## 5. Building Block View

```
ml4b-project/
├── agents/                     # Claude Code specialist agent instruction files
├── app/                        # Streamlit application
│   ├── streamlit_app.py        #   entry point: uv run streamlit run app/streamlit_app.py
│   └── pages/                  #   render() modules imported by the entry point
│       ├── home.py             #     overview, metrics, Sensor Logger instructions
│       ├── prediction.py       #     CSV/ZIP upload → predictions, charts, download
│       └── model_performance.py #    metrics, model comparison, confusion matrix
├── scripts/                    # Stand-alone scripts
│   ├── build_mmfit_dataset.py  #   build processed CSVs from MM-Fit (ADR-013)
│   └── train_model.py          #   reproducible training from processed CSVs (load→…→save)
├── src/ml4b/                   # Installable Python package
│   ├── data/                   #   data preparation pipeline (Phase 3)
│   │   ├── loader.py           #     RecoFit .mat → long-format DataFrame (original source, superseded)
│   │   ├── mmfit_loader.py     #     MM-Fit .npy (both wrists) → long-format DataFrame, 7 classes (ADR-013)
│   │   ├── windowing.py        #     Sliding-window segmentation (2 s, 50% overlap — ADR-006)
│   │   ├── features.py         #     Per-window statistical + FFT features (47 dims)
│   │   ├── augmentation.py     #     Rotation augmentation (off by default — ADR-014)
│   │   ├── splitting.py        #     Subject-based split + rest undersampling (ADR-007/008)
│   │   └── apple_watch_loader.py #   Sensor Logger CSV → window → features → predict; MM-Fit unit alignment (ADR-013)
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
| `src/ml4b/data/mmfit_loader.py` | **(current source — ADR-013)** Read MM-Fit smartwatch `.npy` (both wrists), pair acc+gyr, label by frame, decimate 100→50 Hz, map to 7 classes; emits the same long-format schema as `loader.py` |
| `src/ml4b/data/loader.py` | (original source, superseded) Read RecoFit `.mat` via scipy.io.loadmat, flatten to a long-format DataFrame, filter to 6 classes via `EXERCISE_MAPPING` |
| `src/ml4b/data/augmentation.py` | Rotation augmentation for orientation robustness; implemented + tested but disabled by default (ADR-014) |
| `src/ml4b/data/windowing.py` | Segment continuous recordings into 100-sample (2 s) windows with 50% overlap, never crossing subject/exercise/recording boundaries (ADR-006) |
| `src/ml4b/data/features.py` | Extract 47 features per window: 7 statistics × 6 axes, 3 magnitude features, 2 FFT features |
| `src/ml4b/data/splitting.py` | Partition by `subject_id` into disjoint train/val/test (ADR-007); undersample `rest` class in train to `2×` largest exercise class to fix 89% imbalance (ADR-008) |
| `src/ml4b/models/train.py` | Train Random Forest, XGBoost, SVM classifiers; all use `class_weight='balanced'`; SVM wrapped in `Pipeline` with `StandardScaler` (ADR-009) |
| `src/ml4b/models/evaluate.py` | Compute accuracy, macro F1, per-class F1, confusion matrix; save plots to `reports/figures/`; serialise best model with `joblib.dump` |
| `src/ml4b/data/apple_watch_loader.py` | Load Sensor Logger `WristMotion.csv` or ZIP → auto-detect 4 column formats → sliding window → `extract_features()` → `model.predict()`; `predict_from_sensor_logger()` is the app's entry point |
| `scripts/train_model.py` | Reproduce the trained model end-to-end (load → window → features → split → train → save `best_model.joblib`); the Jupyter-free path |
| `app/pages/*.py` | `render()` functions for the Home, Predict Exercise, and Model Performance pages |
| `src/ml4b/utils/config.py` | Centralised path resolution via env vars (PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR) |
| `app/streamlit_app.py` | Streamlit UI: file upload → feature extraction → prediction |
| `notebooks/` | CRISP-DM phase documentation and exploratory analysis |
| `agents/` | Claude Code specialist agent instruction files |
| `reports/figures/` | Generated evaluation plots and result figures |

---

## 6. Runtime View

### Training Pipeline (Phase 3 → Phase 4)
```
data/raw/mm-fit/w{ID}/  — smartwatch .npy (both wrists) + labels.csv  (~1.7 GB)
    │
    ▼ src/ml4b/data/mmfit_loader.py  — load_mmfit_split()   [scripts/build_mmfit_dataset.py]
    │   Pair acc+gyr by timestamp, label by pose frame, decimate 100→50 Hz
    │   Map MMFIT_TO_ML4B → keep the 7 target classes (ADR-013)
Long DataFrame: subject_id, exercise_name, recording_id, timestamp, ax, ay, az, gx, gy, gz
    │
    ▼ src/ml4b/data/windowing.py  — apply_sliding_window(size=100, overlap=0.5)
    │   Per (subject_id, exercise_name, recording_id) group
    │   2 s windows at 50 Hz, 50% overlap (ADR-006)
Window DataFrame: one row per window, raw_* columns hold lists of 100 samples
    │
    ▼ (optional) src/ml4b/data/augmentation.py  — rotation augmentation [OFF by default, ADR-014]
    │
    ▼ src/ml4b/data/features.py  — extract_features()
    │   Per axis: mean, std, min, max, range, RMS, zero-crossing rate
    │   Magnitudes: accel_magnitude_{mean,std}, gyro_magnitude_mean
    │   FFT on accel magnitude: dominant_frequency, spectral_energy
Feature matrix: 47 numeric features + (subject_id, exercise_name, window_id)
    │
    ▼ MM-Fit official workout-id split (train/val/test) — no leakage (ADR-013)
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

### Prediction Flow (Streamlit app)
```
User uploads WristMotion.csv OR a Sensor Logger ZIP via the browser
    │   (app/pages/prediction.py saves the upload to a temp file)
    │
    ▼ src/ml4b/data/apple_watch_loader.py — predict_from_sensor_logger()
    │   load_sensor_logger_csv() / load_sensor_logger_zip()
    │   detect_and_normalize_columns()  — auto-detect 4 formats → [timestamp,ax..gz]
    │   apply_sliding_window()  — 100 samples, 50% overlap (SAME as training)
    │   extract_features()      — same function as training, guaranteed identical
Feature matrix (n_windows × 47), reindexed to feature_names.txt order
    │
    ▼ models/saved/best_model.joblib — model.predict() + model.predict_proba()
Per-window: predicted_class, confidence, time_start_seconds
    │
    ▼ app/pages/prediction.py — summary metrics, timeline + pie charts,
      results table, and CSV download shown to the user
```

> Model loading is cached with `st.cache_resource` in `app/streamlit_app.py`, so
> `best_model.joblib` and `feature_names.txt` are read once per session. Both are
> committed to git, so the app runs after a fresh clone with no dataset.

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
