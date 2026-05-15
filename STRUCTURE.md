# Project Structure — ML4B Gym Exercise Recognition

> Quick reference for new team members. Read this first — you should understand where everything lives within 2 minutes.

---

## Top-Level Overview

```
ml4b-project/
│
├── app/                    ← Streamlit web application
├── Course_Files/           ← University materials (READ-ONLY, never touch)
├── data/                   ← All data (NOT in git — see .gitignore)
├── docs/                   ← Project documentation
├── models/                 ← Trained model files (NOT in git)
├── notebooks/              ← Jupyter notebooks (one per CRISP-DM phase)
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

### `app/`
The Streamlit web application for live exercise prediction.

```
app/
└── main.py                 ← Entry point: uv run streamlit run app/main.py
```

- **What goes here:** UI code only — file upload, result display, visualisations.
- **What does NOT go here:** ML logic, data loading, feature engineering — those belong in `src/ml4b/`.

---

### `Course_Files/`
University-provided lecture slides, datasets, and example code.

- **READ-ONLY.** Never modify, delete, or import from this folder.
- Not relevant to the actual project code — for reference only.

---

### `data/`
All data files. **This folder is in `.gitignore` — nothing inside it is ever committed.**

```
data/
├── raw/                    ← Original sensor recordings, exactly as collected
└── processed/              ← Cleaned, windowed, feature-engineered datasets
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
│   └── architecture.md     ← arc42 architecture document (keep updated!)
├── business_understanding/
│   └── business_understanding.md ← CRISP-DM Phase 1 deliverable
├── data/
│   └── data_dictionary.md  ← Sensor columns, engineered features, label definitions
├── decisions/
│   ├── ADR-001-python-package-manager.md
│   └── ADR-002-ml-framework.md
├── project/
│   └── crisp_dm_log.md     ← CRISP-DM phase progress tracker
└── setup/
    ├── Setup_macOS.md      ← Environment setup guide for macOS
    ├── Setup_Windows.md    ← Environment setup guide for Windows
    └── Setup_WSL_Windows.md ← Environment setup guide for WSL
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
│   └── __init__.py         ← Data loading, validation, train/test splitting
├── models/
│   └── __init__.py         ← Feature engineering, model training, serialisation
└── utils/
    ├── __init__.py
    └── config.py           ← Path configuration via environment variables
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

### `tests/`
Unit tests. Mirror the structure of `src/ml4b/`.

```
tests/
└── utils/
    └── test_config.py      ← Example: mirrors src/ml4b/utils/config.py
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
uv run streamlit run app/main.py

# 6. Run tests
uv run pytest

# 7. Format and lint before committing
uv run ruff format .
uv run ruff check .
```
