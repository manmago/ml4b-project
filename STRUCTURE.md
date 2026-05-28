# Project Structure — ML4B Gym Exercise Recognition

> Quick reference for new team members. Read this first — you should understand where everything lives within 2 minutes.

---

## Top-Level Overview

```
ml4b-project/
│
├── agents/                 ← Claude Code specialist agent instruction files
├── app/                    ← Streamlit web application
├── data/                   ← All data (NOT in git — see .gitignore)
├── docs/                   ← Project documentation
├── models/                 ← Trained model files (NOT in git)
├── notebooks/              ← Jupyter notebooks (one per CRISP-DM phase)
├── reports/                ← Generated figures and result summaries (NOT in git)
├── src/ml4b/               ← Reusable Python package
├── tests/                  ← Unit tests
│
├── .env                    ← Your local secrets/paths (NOT in git — copy from .env.example)
├── .env.example            ← Template showing all available env variables
├── .gitignore              ← What git ignores
├── CLAUDE.md               ← Instructions for the Claude Code AI assistant
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
└── streamlit_app.py        ← Entry point: uv run streamlit run app/streamlit_app.py
```

- **What goes here:** UI code only — file upload, result display, visualisations.
- **What does NOT go here:** ML logic, data loading, feature engineering — those belong in `src/ml4b/`.

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
    └── feature_names.txt   ← NOT in git — ordered list of 47 feature column names
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
│   └── ADR-008-undersampling-strategy.md
├── project/
│   └── crisp_dm_log.md          ← CRISP-DM phase progress tracker
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
Serialised, trained model files. **Not committed to git** (binaries are large and environment-specific).

```
models/
└── saved/                  ← Joblib-serialised scikit-learn pipelines
```

**Naming convention for model files:**
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
└── 06_deployment.ipynb
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
│   └── splitting.py        ← Subject-based train/val/test split (ADR-007); undersample_majority_class() caps rest at 2× largest exercise class to fix 89% imbalance (ADR-008)
├── models/
│   └── __init__.py         ← Model training, evaluation, serialisation (filled in Phase 4)
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
└── __init__.py             ← Package marker; add test_<module>.py files alongside it
```

**Naming convention:** `test_<module_name>.py` — must start with `test_` for pytest to discover it.

**Run tests with:** `uv run pytest`

---

## What NEVER Gets Committed

| What | Why |
|------|-----|
| `data/` | Data files can be large and contain personal/sensitive info |
| `models/saved/*.joblib` | Binary files; environment-specific |
| `.env` | Contains secrets and local paths |
| `.venv/` | Reproducible via `uv sync` — no need to commit |
| `__pycache__/`, `*.pyc` | Auto-generated bytecode |
| `.ipynb_checkpoints/` | Jupyter auto-save artefacts |
| `*.egg-info/` | Build artefacts |

All of these are already covered by `.gitignore`. If you're unsure, run `git status` before committing — if something unexpected appears, check `.gitignore` before adding it.

---

## Quick Start for New Team Members

```bash
# 1. Clone the repo
git clone https://github.com/AnshulAgrawal7/ml4b-project.git
cd ml4b-project

# 2. Install all dependencies (creates .venv automatically)
uv sync

# 3. Copy environment config
cp .env.example .env
# Edit .env if your data lives somewhere other than data/raw/

# 4. Check out the develop branch (never work directly on main)
git checkout develop
git checkout -b feature/your-feature-name

# 5. Run the Streamlit app
uv run streamlit run app/streamlit_app.py

# 6. Run tests
uv run pytest

# 7. Format and lint before committing
uv run ruff format .
uv run ruff check .
```
