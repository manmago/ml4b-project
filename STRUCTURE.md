# Project Structure — ML4B Gym Exercise Recognition

> Quick reference for new team members. Read this first — you should understand
> where everything lives within 2 minutes.

---

## Top-Level Overview

```
ml4b-project/
│
├── app/                    ← Streamlit web application
├── data/                   ← All data (mostly NOT in git — except Testdaten/ + feature_names.txt)
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
├── Makefile                ← Shortcuts: make run / train / update / calibrate / test / lint / format
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
├── streamlit_app.py        ← Entry point: cached model loading + top-tab navigation
├── pages/
│   ├── __init__.py
│   ├── home.py             ← render(): About tab — overview, honest metrics, how-to-read results
│   ├── prediction.py       ← render(model, feature_names): Classify tab — CSV/ZIP upload →
│   │                          shared-axis scope+timeline, result (dumbbell + ring), comparison, CSV
│   └── model_performance.py ← render(): Model & Training tab — leave-one-set-out metrics,
│                               precision·recall·F1 table, confusion matrix, model details, limitations
├── ui/                     ← "Daylight" design system (presentation only, no ML logic)
│   ├── __init__.py
│   ├── theme.py            ← CSS, colour/type/status tokens, components (dumbbell icons, ring, tiles)
│   ├── viz.py              ← light Plotly figures (shared-axis scope+timeline, donut, F1, matrix)
│   ├── lottie.py           ← optional Lottie animations, auto-detected, with SVG fallback
│   └── journey.py          ← numbered onboarding walkthrough (Classify empty state)
└── assets/
    └── lottie/             ← drop-in <exercise>.json animations (README explains naming)
```

- **What goes here:** UI code only. ML logic lives in `src/ml4b/`.
- Navigation: top **tabs** in `streamlit_app.py` (Classify / Model & Training /
  About) route to each page's `render()`. The sidebar is hidden via `app/ui/theme.py`.
- Exercise animations: `app/ui/lottie.py` renders `app/assets/lottie/<exercise>.json`
  if present, else falls back to the built-in animated dumbbell SVG (DECISIONS.md §10).

---

### `data/`
All data files. **In `.gitignore` — nothing inside is committed except the two
small text files noted below.**

```
data/
├── raw/
│   ├── kaggle_gym_imu/     ← CURRENT training anchor — Kaggle Gym Workout IMU
│   │                          (Apple Watch, 100 Hz, single subject; DECISIONS.md). NOT in git.
│   ├── apple_watch/        ← Real Apple Watch test samples for the sanity check (NOT in git)
│   └── mm-fit/             ← ABANDONED interim source (non-Apple smartwatch, DECISIONS.md). NOT in git.
│                              (RecoFit, the abandoned ORIGINAL source, is no longer kept locally —
│                               its history is in docs/data/dataset_evaluation.md.)
├── Testdaten/             ← IN git — our own committed Apple-Watch recordings, one subfolder
│   │                          per category. The CANONICAL continual-learning input read by
│   │                          `make update` (DECISIONS.md §8; src/ml4b/data/testdaten.py).
│   ├── Biceps_Curls/  Rows/  Triceps_Extensions/   ← training sets (folder name = label)
│   └── Rest/  Uncertain/                            ← calibrate the gate / validate `unknown`
├── processed/
│   ├── .gitkeep
│   ├── README.md
│   └── feature_names.txt   ← IN git (exception) — ordered list of the 39 invariant
│                              feature names the app/model need (DECISIONS.md)
└── feedback/              ← Feedback store for the lower-level retrain scripts only
                              (feedback.jsonl; DECISIONS.md §8). NOT in git. The canonical
                              `make update` rebuild reads data/Testdaten/ folders directly.
```

- Never edit files in `raw/` — treat them as immutable originals.

---

### `docs/`
All project documentation, organised by topic.

```
docs/
├── architecture/architecture.md          ← arc42 architecture (keep updated)
├── business_understanding/business_understanding.md ← CRISP-DM Phase 1
├── data/
│   ├── data_dictionary.md                ← Sensor columns + the 39 invariant features
│   └── dataset_evaluation.md             ← Dataset comparison & rationale (Kaggle + Testdaten)
├── data_understanding/data_understanding.md ← CRISP-DM Phase 2: short narrative summary
├── DECISIONS.md                           ← Consolidated decision log (every major decision)
├── Handoff.md                             ← Condensed handover summary for a team presenting the project (German)
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
    ├── best_model.joblib       ← IN git — Model 2 (Kaggle + Testdaten), used by the app (compressed)
    ├── random_forest.joblib    ← IN git — archive copy of Model 2
    ├── novelty_detector.joblib ← IN git — open-set novelty detector for Model 2 (DECISIONS.md)
    ├── model_metrics.json      ← IN git — honest leave-one-set-out metrics for Model 2
    │                           shown on the Model Performance page
    ├── baseline_model.joblib            ← IN git — Model 1 (Kaggle only); the app runs both so
    │                                      the Predict page shows the effect of our data (DECISIONS.md §9)
    ├── baseline_novelty_detector.joblib ← IN git — open-set novelty detector for Model 1
    ├── baseline_metrics.json            ← IN git — leave-one-set-out metrics for Model 1
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
├── 04_modeling.ipynb            ← Random Forest + leave-one-set-out CV (Kaggle-only baseline, macro F1 0.776)
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
│   ├── testdaten.py          ← Testdaten/ folder discovery + labels (shared by rebuild + calibrate)
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
│   ├── pipeline.py           ← shared build: augment→features→RF+novelty, leave-one-set-out CV,
│   │                            metrics payload. ONE source of truth for both models (DECISIONS.md §9)
│   └── evaluate.py           ← evaluate_model(), compare_models(), save_model()
├── feedback/                 ← Continual-learning building blocks (DECISIONS.md §8). The
│   │                            canonical entry point is scripts/rebuild_from_testdaten.py.
│   ├── __init__.py
│   ├── store.py              ← persist/load labelled windows (data/feedback/feedback.jsonl)
│   └── retrain.py            ← rebuild model from base data + feedback store (same pipeline)
└── utils/
    ├── __init__.py
    ├── config.py             ← Path configuration (PROJECT_ROOT, DATA_RAW, …, find_project_root)
    └── metrics.py            ← load_model_metrics() (Model 2) + load_baseline_metrics() (Model 1)
```

- Every function/class needs type hints and a Google-style docstring.
- Run `uv run ruff format .` and `uv run ruff check .` before every commit.

---

### `scripts/`
Stand-alone, runnable scripts (not part of the importable package).

```
scripts/
├── rebuild_from_testdaten.py ← CANONICAL continual-learning rebuild (`make update`): retrains
│                                BOTH models — baseline (Kaggle only) + current (Kaggle +
│                                committed Testdaten/<Exercise>/) — refits novelty + refreshes
│                                metrics for each (DECISIONS.md §8/§9, docs/project/continual_training.md).
│                                Run: uv run python scripts/rebuild_from_testdaten.py  (--no-cv to skip CV)
├── calibrate_gate.py         ← Read-only: measure rest (Testdaten/Rest/) vs exercise (Kaggle)
│                                energy, recommend activity-gate thresholds in the gap (`make
│                                calibrate`; DECISIONS.md §5). Run: uv run python scripts/calibrate_gate.py
├── train_model.py            ← Bootstrap: train the 3-class model on Kaggle data ONLY (leave-one-set-out
│                                CV). Writes the Kaggle model to BOTH the current and baseline slots
│                                (identical until `make update` adds Testdaten). Run: uv run python scripts/train_model.py
├── fit_novelty_detector.py  ← Fit the open-set novelty detector on Kaggle invariant features;
│                                saves novelty_detector.joblib (DECISIONS.md). Run: uv run python scripts/fit_novelty_detector.py
├── add_labelled_recording.py ← Lower-level building block: add one labelled recording to
│                                the feedback store (DECISIONS.md §8). Run: uv run python
│                                scripts/add_labelled_recording.py REC.csv --label bicep_curl
├── update_model.py          ← Lower-level building block: retrain on base data + feedback store (DECISIONS.md §8).
│                                Run: uv run python scripts/update_model.py  (--restore-base to undo)
├── inspect_kaggle_dataset.py ← Read-only audit of the Kaggle dataset (columns, labels,
│                                sets per exercise, sampling rate). Run: uv run python scripts/inspect_kaggle_dataset.py
├── generate_report_figures.py ← Render the curated handover figures into reports/figures/
│                                from committed metrics + model (deterministic, no dataset).
│                                Run: uv run python scripts/generate_report_figures.py
├── build_mmfit_dataset.py    ← LEGACY MM-Fit feature builder — abandoned (DECISIONS.md)
└── test_apple_watch_prediction.py ← Helper: per-file predictions on real WristMotion.csv samples
```

Always resolve paths via `find_project_root()` / `ml4b.utils.config`.

---

### `reports/`
Generated figures and result summaries. The curated handover figure set in
`reports/figures/` is produced deterministically from the committed model +
metrics by `scripts/generate_report_figures.py` (confusion matrices for both
models, per-class/macro F1 comparison, top-15 feature importances, dataset
composition). `scripts/train_model.py` also writes
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
| `data/` **except** `data/Testdaten/` and `data/processed/feature_names.txt` | Large and/or personal data. `data/Testdaten/` (our own labelled Apple-Watch recordings) is committed so `make update` reproduces the model. |
| `models/saved/*.joblib` except `best_model.joblib` / `random_forest.joblib` / `novelty_detector.joblib` / `baseline_model.joblib` / `baseline_novelty_detector.joblib` | Binary; large. The committed exceptions (both models) let the app run + compare without the dataset. |
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
