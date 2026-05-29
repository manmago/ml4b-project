# Project Structure — ML4B Gym Exercise Recognition

> Quick reference for new team members. Read this first — you should understand where everything lives within 2 minutes.

---

## Top-Level Overview

```
ml4b-project/
│
├── agents/                 ← Claude Code specialist agent instruction files
├── app/                    ← Streamlit web application
├── data/                   ← All data (mostly NOT in git — except feature_names.txt)
├── docs/                   ← Project documentation
├── models/                 ← Trained model files (best_model.joblib IS committed)
├── notebooks/              ← Jupyter notebooks (one per CRISP-DM phase)
├── reports/                ← Generated figures and result summaries (NOT in git)
├── scripts/                ← Stand-alone scripts (train_model.py)
├── src/ml4b/               ← Reusable Python package
├── tests/                  ← Unit tests
│
├── .streamlit/             ← Streamlit config (single sidebar navigation)
├── .env                    ← Your local secrets/paths (NOT in git — copy from .env.example)
├── .env.example            ← Template showing all available env variables
├── .gitignore              ← What git ignores
├── CLAUDE.md               ← Instructions for the Claude Code AI assistant
├── Makefile                ← Shortcuts: make run / train / test / lint / format
├── STRUCTURE.md            ← This file
├── pyproject.toml          ← Project metadata, dependencies, tool config
└── uv.lock                 ← Exact pinned versions of every dependency (always commit this)
```

---

## Folder-by-Folder Breakdown

### `agents/`
Claude Code specialist agent instruction files. Each file defines a focused role with explicit responsibilities, code standards, and output requirements.

```
agents/
├── data_scientist.md       ← ML & data science work: feature engineering, modeling, notebooks
├── documenter.md           ← Documentation: arc42, ADRs, CRISP-DM log, setup guides
└── reviewer.md             ← Pre-commit review checklist for code and documentation
```

- **What goes here:** `.md` files containing Claude Code agent instructions only.
- **When to use:** Select the appropriate agent based on the task type (see CLAUDE.md for guidance).

---

### `app/`
The Streamlit web application for live exercise prediction.

```
app/
├── __init__.py             ← Makes `app` importable so the entry point can load pages
├── streamlit_app.py        ← Entry point: cached model loading + sidebar navigation
└── pages/
    ├── __init__.py         ← Package marker for the page modules
    ├── home.py             ← render(): overview, metrics, Sensor Logger instructions
    ├── prediction.py       ← render(model, feature_names): CSV/ZIP upload → predictions,
    │                          timeline + pie charts, results table, CSV download
    └── model_performance.py ← render(): test metrics, model comparison, per-class F1,
                               row-normalized confusion matrix
```

- **What goes here:** UI code only — file upload, result display, visualisations.
- **What does NOT go here:** ML logic, data loading, feature engineering — those belong in `src/ml4b/`.
- **Navigation:** a sidebar radio in `streamlit_app.py` routes to each page's `render()`.
  `.streamlit/config.toml` disables Streamlit's automatic `pages/` discovery so there is
  exactly one navigation control.

---

### `data/`
All data files. **This folder is in `.gitignore` — nothing inside it is ever committed.**

```
data/
├── raw/
│   └── recofit/            ← RecoFit .mat files (~2.5 GB, NOT in git — see README.md inside)
└── processed/              ← Output of notebooks/03_data_preparation.ipynb
    ├── .gitkeep            ← Keeps folder tracked even when empty
    ├── README.md           ← Describes the expected CSV files and how to reproduce them
    ├── train_features.csv  ← NOT in git — ~70% of subjects (features + labels)
    ├── val_features.csv    ← NOT in git — ~10% of subjects (features + labels)
    ├── test_features.csv   ← NOT in git — ~20% of subjects (features + labels)
    └── feature_names.txt   ← IN git (exception) — ordered list of 47 feature column names; the app needs it
```

**Naming conventions for data files:**
| File type | Convention | Example |
|-----------|-----------|---------|
| Raw recording | `<participant_id>_<exercise>_<session>.csv` | `p01_bicep_curl_01.csv` |
| Processed features | `features_<split>.csv` | `features_train.csv` |
| Any other file | lowercase, underscores, no spaces | `metadata_participants.csv` |

**Rules:**
- Never edit files in `raw/` — treat them as immutable originals.
- All transformations produce new files in `processed/`.

---

### `docs/`
All project documentation, organised by topic.

```
docs/
├── architecture/
│   └── architecture.md          ← arc42 architecture document (keep updated!)
├── business_understanding/
│   └── business_understanding.md ← CRISP-DM Phase 1 deliverable
├── data/
│   └── data_dictionary.md       ← Sensor columns, engineered features, label definitions
├── data_understanding/
│   └── dataset_evaluation.md    ← CRISP-DM Phase 2: dataset comparison & selection rationale
├── decisions/
│   ├── ADR-001-python-package-manager.md
│   ├── ADR-002-ml-framework.md
│   ├── ADR-003-multi-agent-documentation-strategy.md
│   ├── ADR-004-code-comment-and-documentation-standard.md
│   ├── ADR-005-exercise-class-selection.md
│   ├── ADR-006-sliding-window-parameters.md
│   ├── ADR-007-subject-based-train-test-split.md
│   ├── ADR-008-undersampling-strategy.md
│   ├── ADR-009-model-selection-rationale.md
│   └── ADR-010-random-forest-as-final-model.md
├── project/
│   ├── crisp_dm_log.md          ← CRISP-DM phase progress tracker
│   ├── project_overview.md      ← Plain-language project overview — read this first
│   └── apple_watch_data_collection_guide.md ← Recording protocol for Sensor Logger / Apple Watch data
└── setup/
    ├── Setup_macOS.md           ← Environment setup guide for macOS
    ├── Setup_Windows.md         ← Environment setup guide for Windows
    └── Setup_WSL_Windows.md     ← Environment setup guide for WSL
```

**Rules:**
- Add a new `ADR-NNN-<topic>.md` in `docs/decisions/` for every major technical choice.
- Update `docs/data/data_dictionary.md` as soon as the data schema is known.
- Update `docs/project/crisp_dm_log.md` as phases progress.

---

### `models/`
Serialised, trained model files.

```
models/
└── saved/
    ├── best_model.joblib    ← IN git (exception) — Random Forest used by the app
    └── random_forest.joblib ← IN git (exception) — archive copy of the same model
```

**Committed on purpose:** `best_model.joblib` and `random_forest.joblib` are
re-included via `.gitignore` exceptions so the Streamlit app works after a fresh
clone with no dataset download. Any *other* `*.joblib` (experiments, larger
models) stays ignored.

**Naming convention for additional model files:**
```
<algorithm>_<feature_set>_<date>.joblib
# Example: random_forest_v1_20260601.joblib
```

---

### `notebooks/`
Jupyter notebooks for exploration and CRISP-DM documentation. One notebook per phase.

```
notebooks/
├── 01_business_understanding.ipynb
├── 02_data_understanding.ipynb
├── 03_data_preparation.ipynb
├── 04_modeling.ipynb
├── 05_evaluation.ipynb
├── 06_deployment.ipynb
└── 06_streamlit_demo.ipynb   ← Self-contained demo of the app's prediction pipeline
```

**Naming convention:** `<two-digit-phase-number>_<phase_name>.ipynb` — always lowercase, underscores.

**Rules:**
- Notebooks are for exploration and storytelling, not for production code.
- Any reusable logic discovered in a notebook must be extracted into `src/ml4b/`.
- Clear all outputs before committing (`Edit → Clear All Outputs` in Jupyter).
- Never hardcode file paths — import from `ml4b.utils.config` instead.

---

### `src/ml4b/`
The installable Python package. All reusable, tested code lives here.

```
src/ml4b/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── loader.py           ← Read RecoFit .mat → long-format DataFrame, filter to 6 target classes
│   ├── windowing.py        ← Sliding-window segmentation (2 s windows, 50% overlap — ADR-006)
│   ├── features.py         ← Statistical + FFT feature extraction per window (47 features)
│   ├── splitting.py        ← Subject-based train/val/test split (ADR-007); undersample_majority_class() caps rest at 2× largest exercise class to fix 89% imbalance (ADR-008)
│   └── apple_watch_loader.py ← Sensor Logger CSV loader + predict_from_sensor_logger() for Streamlit app
├── models/
│   ├── __init__.py         ← Models subpackage marker
│   ├── train.py            ← train_random_forest(), train_xgboost(), train_svm() — see ADR-009
│   └── evaluate.py         ← evaluate_model(), compare_models(), save_model()
└── utils/
    ├── __init__.py
    └── config.py           ← Path configuration via environment variables (PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR)
```

**Naming conventions for Python files:**
| Thing | Convention | Example |
|-------|-----------|---------|
| Module (file) | `snake_case.py` | `feature_engineering.py` |
| Class | `PascalCase` | `SensorDataLoader` |
| Function / variable | `snake_case` | `load_raw_data()` |
| Constant | `UPPER_SNAKE_CASE` | `DATA_RAW` |

**Rules:**
- Every function and class needs type hints and a Google-style docstring.
- Every new module needs a corresponding test file in `tests/`.
- Run `uv run ruff format .` and `uv run ruff check .` before every commit.

---

### `scripts/`
Stand-alone, runnable scripts that are not part of the importable package.

```
scripts/
└── train_model.py          ← Reproduce the trained model end-to-end without Jupyter:
                               load → window → features → split → train → save .joblib
                               Run: uv run python scripts/train_model.py
```

- **What goes here:** one-shot operational scripts (training, data prep helpers).
- Always resolve paths via `find_project_root()` / `ml4b.utils.config` — never hardcode.

---

### `reports/`
Generated output files from analysis and model evaluation. **Not committed to git** (generated artefacts).

```
reports/
└── figures/                ← Matplotlib/seaborn plots saved during notebooks and evaluation
    └── .gitkeep            ← Keeps folder tracked by git even when empty
```

**Rules:**
- Save all plots here from notebooks and evaluation scripts.
- Never commit actual figure files — only the `.gitkeep` placeholder.
- Reference figures from notebook markdown cells using relative paths.

---

### `tests/`
Unit tests. Mirror the structure of `src/ml4b/`.

```
tests/
├── __init__.py                  ← Package marker; add test_<module>.py files alongside it
├── test_features.py             ← Feature extraction: 47 features, finite, deterministic
└── test_apple_watch_loader.py   ← Column auto-detection, ZIP, predict pipeline, error guards
```

**Naming convention:** `test_<module_name>.py` — must start with `test_` for pytest to discover it.

**Run tests with:** `uv run pytest`

---

## What NEVER Gets Committed

| What | Why |
|------|-----|
| `data/` | Data files can be large and contain personal/sensitive info |
| `models/saved/*.joblib` (except `best_model.joblib`, `random_forest.joblib`) | Binary files; large. The two committed exceptions let the app run without the dataset. |
| `.env` | Contains secrets and local paths |
| `.venv/` | Reproducible via `uv sync` — no need to commit |
| `__pycache__/`, `*.pyc` | Auto-generated bytecode |
| `.ipynb_checkpoints/` | Jupyter auto-save artefacts |
| `*.egg-info/` | Build artefacts |

All of these are already covered by `.gitignore`. If you're unsure, run `git status` before committing — if something unexpected appears, check `.gitignore` before adding it.

---

## Quick Start for New Team Members

# Run the app in 3 commands — no dataset download needed:
```bash
git clone git@github.com:AnshulAgrawal7/ml4b-project.git
cd ml4b-project
uv sync
uv run streamlit run app/streamlit_app.py   # → http://localhost:8501
```

Then, to contribute:
```bash
# Copy environment config (optional — only if your data lives elsewhere)
cp .env.example .env

# Work on a feature branch (never commit directly to main)
git checkout develop
git checkout -b feature/your-feature-name

# Run tests, format, and lint before committing
uv run pytest
uv run ruff format .
uv run ruff check .

# Retrain the model from scratch (requires the RecoFit dataset)
uv run python scripts/train_model.py
```
