# Project Structure — ML4B Gym Exercise Recognition

> Quick reference for new team members. Read this first — you should understand
> where everything lives within 2 minutes.

---

## Top-Level Overview

```
ml4b-project/
│
├── app/                    ← Streamlit web application
├── data/                   ← All data (mostly NOT in git — except feature_names.txt)
├── docs/                   ← Project documentation (incl. DECISIONS.md)
├── models/                 ← Trained model + metrics (committed so the app runs)
├── notebooks/              ← Jupyter notebooks (one per CRISP-DM phase)
├── reports/                ← Generated figures and result summaries (NOT in git)
├── scripts/                ← Stand-alone scripts (train_model.py, inspect_kaggle_dataset.py)
├── src/ml4b/               ← Reusable Python package
├── tests/                  ← Unit tests
│
├── .streamlit/             ← Streamlit config (single sidebar navigation)
├── .env.example            ← Template showing all available env variables
├── .gitignore              ← What git ignores (with model/metrics exceptions)
├── Makefile                ← Shortcuts: make run / train / test / lint / format
├── run_app.sh              ← One-click app launcher (macOS/Linux/WSL)
├── run_app.bat             ← One-click app launcher (Windows)
├── STRUCTURE.md            ← This file
├── pyproject.toml          ← Project metadata, dependencies, tool config
└── uv.lock                 ← Exact pinned dependency versions (always committed)
```

---

## Folder-by-Folder Breakdown

### `app/`
The Streamlit web application for live exercise prediction.

```
app/
├── __init__.py
├── streamlit_app.py        ← Entry point: cached model loading + sidebar navigation
└── pages/
    ├── __init__.py
    ├── home.py             ← render(): overview, honest metrics, Sensor Logger steps
    ├── prediction.py       ← render(model, feature_names): CSV/ZIP upload → timeline,
    │                          pie, results table, CSV download, detected sampling rate
    └── model_performance.py ← render(): leave-one-set-out metrics from model_metrics.json,
                               per-class F1, confusion matrix, model details, limitations
```

- **What goes here:** UI code only. ML logic lives in `src/ml4b/`.
- Navigation: a sidebar radio in `streamlit_app.py` routes to each page's
  `render()`. `.streamlit/config.toml` disables automatic `pages/` discovery.

---

### `data/`
All data files. **In `.gitignore` — nothing inside is committed except the two
small text files noted below.**

```
data/
├── raw/
│   ├── kaggle_gym_imu/     ← CURRENT training source — Kaggle Gym Workout IMU
│   │                          (Apple Watch, 100 Hz, single subject; DECISIONS.md). NOT in git.
│   ├── apple_watch/        ← Real Apple Watch test samples for the sanity check (NOT in git)
│   ├── recofit/            ← ABANDONED original source (forearm-worn, DECISIONS.md). NOT in git.
│   └── mm-fit/             ← ABANDONED interim source (non-Apple smartwatch, DECISIONS.md). NOT in git.
├── processed/
│   ├── .gitkeep
│   ├── README.md
│   └── feature_names.txt   ← IN git (exception) — ordered list of the 39 invariant
│                              feature names the app/model need (DECISIONS.md)
└── feedback/              ← User label corrections for continual learning
                              (feedback.jsonl; DECISIONS.md). NOT in git — the user's own data.
```

- Never edit files in `raw/` — treat them as immutable originals.

---

### `docs/`
All project documentation, organised by topic.

```
docs/
├── architecture/architecture.md          ← arc42 architecture (keep updated)
├── business_understanding/business_understanding.md ← CRISP-DM Phase 1
├── data/data_dictionary.md               ← Sensor columns + the 39 invariant features
├── data_understanding/dataset_evaluation.md ← CRISP-DM Phase 2: dataset choice (Kaggle)
├── DECISIONS.md                           ← Consolidated decision log (every major decision)
├── project/
│   ├── crisp_dm_log.md                   ← CRISP-DM phase progress tracker
│   ├── project_overview.md               ← Plain-language overview — read this first
│   ├── apple_watch_data_collection_guide.md ← Sensor Logger recording protocol
│   └── apple_watch_validation_results.md ← Honest one-shot sanity-check results (Phase 5)
└── setup/
    ├── Setup_macOS.md
    ├── Setup_Windows.md
    └── Setup_WSL_Windows.md
```

- Record every major technical choice as a new entry in `docs/DECISIONS.md`.
- Update `data_dictionary.md` when features change, `crisp_dm_log.md` as phases
  progress.

---

### `models/`
Serialised, trained model files plus their metrics.

```
models/
└── saved/
    ├── best_model.joblib       ← IN git (exception) — Random Forest used by the app (compressed)
    ├── random_forest.joblib    ← IN git (exception) — archive copy of the same model
    ├── novelty_detector.joblib ← IN git (exception) — open-set novelty detector (DECISIONS.md)
    ├── model_metrics.json      ← IN git (exception) — honest leave-one-set-out metrics
    │                           shown on the Model Performance page
    ├── best_model_base.joblib  ← generated on first retrain — backup of the shipped
    │                             model so it can be restored (DECISIONS.md). NOT in git.
    └── model_manifest.json     ← generated by retrain — what feedback went into the
                                current model (DECISIONS.md). NOT in git.
```

Committed on purpose so the app runs after a fresh clone with no dataset
(DECISIONS.md). Other `*.joblib` files stay ignored.

---

### `notebooks/`
CRISP-DM phase notebooks, all aligned to the current 3-class Apple-Watch pipeline
(DECISIONS.md) and runnable top-to-bottom against `data/raw/kaggle_gym_imu/`:

```
notebooks/
├── 02_data_understanding.ipynb  ← Kaggle dataset exploration (21 abbrevs, 3 classes, 100 Hz)
├── 03_data_preparation.ipynb    ← load → window 200@100Hz → gate → invariant features → augment
├── 04_modeling.ipynb            ← Random Forest + leave-one-set-out CV (macro F1 0.776)
├── 05_evaluation.ipynb          ← honest metrics + limitations + sanity check on test_samples
└── 06_streamlit_demo.ipynb      ← end-to-end predict_from_sensor_logger demo (mirrors the app)
```

Exploration and storytelling only — reusable logic is imported from `src/ml4b/`,
never duplicated. Outputs are stripped before committing; never hardcode paths
(use `ml4b.utils.config`).

---

### `src/ml4b/`
The installable Python package. All reusable, tested code lives here.

```
src/ml4b/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── canonical.py          ← Shared pipeline constants + CoreMotion canonicalization
│   │                            (units, 100 Hz resample, window size) — training & app
│   ├── kaggle_loader.py      ← Load 3-class Kaggle data → long-format DataFrame (DECISIONS.md)
│   ├── windowing.py          ← Sliding-window segmentation (200 @ 100 Hz, 50% overlap; DECISIONS.md);
│   │                            carries recording_id for set-grouped evaluation
│   ├── features_invariant.py ← CURRENT features: 39 device-invariant features (DECISIONS.md)
│   ├── activity_gate.py      ← Energy-threshold rest detection — not a class (DECISIONS.md)
│   ├── novelty.py            ← Open-set novelty detection — unseen exercise → unknown (DECISIONS.md)
│   ├── session.py            ← Bout segmentation — fold windows into per-set summary (DECISIONS.md)
│   ├── augmentation.py       ← Rotation+time-warp+mirror+jitter augmentation (DECISIONS.md)
│   ├── apple_watch_loader.py ← Sensor Logger CSV/ZIP loader + predict_from_sensor_logger()
│   │                            (resample → window → gate → features → novelty → predict)
│   ├── features.py           ← LEGACY per-axis features (47) — abandoned MM-Fit pipeline
│   ├── loader.py             ← LEGACY RecoFit .mat loader — abandoned (DECISIONS.md)
│   ├── mmfit_loader.py       ← LEGACY MM-Fit loader — abandoned (DECISIONS.md)
│   └── splitting.py          ← LEGACY subject split + undersampling (DECISIONS.md)
├── models/
│   ├── __init__.py
│   ├── train.py              ← train_random_forest(), train_xgboost(), train_svm() (DECISIONS.md)
│   └── evaluate.py           ← evaluate_model(), compare_models(), save_model()
├── feedback/                 ← Human-in-the-loop continual learning (DECISIONS.md)
│   ├── __init__.py
│   ├── store.py              ← persist/load user label corrections (data/feedback/feedback.jsonl)
│   └── retrain.py            ← rebuild model from base data + corrections (same pipeline)
└── utils/
    ├── __init__.py
    ├── config.py             ← Path configuration (PROJECT_ROOT, DATA_RAW, …, find_project_root)
    └── metrics.py            ← load_model_metrics() — reads committed model_metrics.json
```

- Every function/class needs type hints and a Google-style docstring.
- Run `uv run ruff format .` and `uv run ruff check .` before every commit.

---

### `scripts/`
Stand-alone, runnable scripts (not part of the importable package).

```
scripts/
├── train_model.py            ← Train the 3-class model on Kaggle data with leave-one-set-out
│                                CV; saves best_model.joblib + model_metrics.json + feature_names.txt.
│                                Run: uv run python scripts/train_model.py
├── fit_novelty_detector.py  ← Fit the open-set novelty detector on Kaggle invariant features;
│                                saves novelty_detector.joblib (DECISIONS.md). Run: uv run python scripts/fit_novelty_detector.py
├── add_labelled_recording.py ← Add a clean, labelled recording (one set) straight to
│                                the feedback store (DECISIONS.md §8). Run: uv run python
│                                scripts/add_labelled_recording.py REC.csv --label bicep_curl
├── update_model.py          ← Continual learning: retrain on base data + user corrections (DECISIONS.md §8).
│                                Run: uv run python scripts/update_model.py  (--restore-base to undo)
├── inspect_kaggle_dataset.py ← Read-only audit of the Kaggle dataset (columns, labels,
│                                sets per exercise, sampling rate). Run: uv run python scripts/inspect_kaggle_dataset.py
├── build_mmfit_dataset.py    ← LEGACY MM-Fit feature builder — abandoned (DECISIONS.md)
└── test_apple_watch_prediction.py ← Helper: per-file predictions on real WristMotion.csv samples
```

Always resolve paths via `find_project_root()` / `ml4b.utils.config`.

---

### `reports/`
Generated figures and result summaries. **Not committed** (only `.gitkeep`
placeholders are tracked). `scripts/train_model.py` also writes
`reports/leave_one_set_out_results.json` here (the committed copy lives under
`models/saved/model_metrics.json`).

---

### `tests/`
Unit tests mirroring `src/ml4b/`.

```
tests/
├── __init__.py
├── test_features_invariant.py ← 39 features, identifier carry-through, rotation invariance
├── test_activity_gate.py      ← rest vs active gating, index-aligned mask
├── test_novelty.py            ← known vs novel detection, thresholds, serialization (DECISIONS.md)
├── test_session.py            ← bout segmentation, majority vote, unknown bouts (DECISIONS.md)
├── test_feedback.py           ← feedback store round-trip, stats, pipeline compatibility (DECISIONS.md)
├── test_augmentation.py       ← rotation + composed augmentation: size, determinism, recording_id
├── test_apple_watch_loader.py ← column auto-detect, ZIP, predict pipeline, error guards
├── test_features.py           ← LEGACY per-axis feature tests
└── test_mmfit_loader.py       ← LEGACY MM-Fit loader tests
```

**Run tests with:** `uv run pytest` (or `make test`).

---

## What NEVER Gets Committed

| What | Why |
|------|-----|
| `data/` | Large and/or personal data |
| `models/saved/*.joblib` except `best_model.joblib` / `random_forest.joblib` / `novelty_detector.joblib` | Binary; large. The committed exceptions let the app run without the dataset. |
| `.env` | Secrets and local paths |
| `.venv/` | Reproducible via `uv` |
| `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `*.egg-info/` | Auto-generated |

All covered by `.gitignore`. Run `git status` before committing if unsure.

---

## Quick Start for New Team Members

Run the app in one command — no dataset download needed:

```bash
git clone git@github.com:AnshulAgrawal7/ml4b-project.git
cd ml4b-project
make run        # → http://localhost:8501
```

To contribute:

```bash
git checkout develop
git checkout -b feature/your-feature-name
make test && make lint && make format
# Retrain (needs the Kaggle dataset in data/raw/kaggle_gym_imu/):
make train
```
