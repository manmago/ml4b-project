# Architecture Documentation — ML4B Gym Exercise Recognition
> arc42 lightweight format — mandatory sections: 1, 2, 3, 5, 6, 8

---

## 1. Goals and Requirements

### Primary Research Question
> "Can machine learning models trained on publicly available wrist-worn sensor data accurately classify gym exercises, and how well do these models generalize to new data collected from an Apple Watch during real workout sessions?"

### Primary Goal
Classify gym exercises from Apple Watch sensor streams (accelerometer + gyroscope) using supervised Machine Learning. The model must distinguish between **7 exercise classes**: Bicep Curl, Shoulder Press, Lateral Raise, Squat, Bench Press, Deadlift, and Rest/No Exercise.

### Two-Dataset Validation Strategy
| Phase | Data Source | Purpose |
|-------|------------|---------|
| Training & validation | Public wrist-sensor datasets (MMFIT, Kaggle, WHARF) | Build and validate the full ML pipeline |
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
| `Course_Files/` is read-only | University materials; must not be modified or imported |
| No cloud infrastructure | All processing runs locally (WSL/macOS/Windows) |

---

## 3. System Context

```
┌─────────────────────────────────────────────────────────┐
│                      External World                      │
│                                                         │
│  Apple Watch ──► raw CSV / sensor logs (accelerometer,  │
│                  gyroscope, timestamps, labels)         │
└───────────────────────────┬─────────────────────────────┘
                            │ data/raw/
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   ML4B System                            │
│                                                         │
│  notebooks/  ──► exploration & CRISP-DM documentation   │
│  src/ml4b/   ──► reusable pipeline package              │
│  app/        ──► Streamlit prediction UI                 │
│  models/     ──► saved model artefacts (.joblib)        │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
              Streamlit App (localhost:8501)
              User uploads sensor window →
              receives predicted exercise label
```

**Inputs:** Raw sensor data files (CSV) collected from Apple Watch during labelled exercise sessions.  
**Outputs:** Trained ML model + Streamlit web app for exercise prediction.

---

## 5. Building Block View

```
ml4b-project/
├── app/                        # Streamlit application
│   └── main.py                 #   entry point: uv run streamlit run app/main.py
├── src/ml4b/                   # Installable Python package
│   ├── data/                   #   data loading & validation
│   ├── models/                 #   model training & inference
│   └── utils/
│       └── config.py           #   env-based path configuration
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
├── docs/                       # Project documentation (this folder)
└── Course_Files/               # University materials — read-only
```

### Key Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/ml4b/data/` | Load raw CSVs, validate schema, split train/test |
| `src/ml4b/models/` | Feature engineering, model training, serialisation |
| `src/ml4b/utils/config.py` | Centralised path resolution via env vars |
| `app/main.py` | Streamlit UI: file upload → feature extraction → prediction |
| `notebooks/` | CRISP-DM phase documentation and exploratory analysis |

---

## 6. Runtime View

### Training Pipeline
```
data/raw/*.csv
    │
    ▼ src/ml4b/data/loader.py
Validated DataFrame (timestamps, ax, ay, az, gx, gy, gz, label)
    │
    ▼ src/ml4b/data/features.py
Feature matrix X, label vector y (sliding window aggregations)
    │
    ▼ src/ml4b/models/train.py
Trained scikit-learn Pipeline (scaler + classifier)
    │
    ▼ models/saved/model_<timestamp>.joblib
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
