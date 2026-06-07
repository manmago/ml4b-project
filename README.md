# ML4B — Gym Exercise Recognition

**ML4B SoSe 2026 · FAU Nürnberg · Lehrstuhl für Wirtschaftsinformatik**

Recognize **three gym exercises — bicep curl, tricep extension, row —** from
wrist-worn **Apple Watch** sensor data (accelerometer + gyroscope). Record a
workout with the free **Sensor Logger** app, upload the file to the Streamlit web
app, and get the recognized exercise per 2-second window with a confidence score.
Pauses between sets are detected automatically as **rest**, and low-confidence
windows are reported as **uncertain**.

---

## Quickstart — one command

```bash
# 1. Install uv once (macOS/Linux/WSL):
curl -LsSf https://astral.sh/uv/install.sh | sh
#    Windows (PowerShell): powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

#    IMPORTANT: the installer adds uv to your PATH, but your CURRENT shell does
#    not know it yet. Open a NEW terminal (or reload: source ~/.bashrc /
#    ~/.zshrc, Windows: close & reopen PowerShell) — otherwise `uv` is "not found".
#    Check it works:  uv --version

# 2. Clone and launch  (HTTPS works for everyone; use SSH only if you have a key)
git clone https://github.com/AnshulAgrawal7/ml4b-project.git
#    SSH alternative: git clone git@github.com:AnshulAgrawal7/ml4b-project.git
cd ml4b-project
make run        # or: ./run_app.sh   ·   run_app.bat (Windows)   ·   uv run streamlit run app/streamlit_app.py
```

→ Open **http://localhost:8501**

**No Python install, no `uv sync`, no pip, no conda, no dataset needed.** `uv run`
provisions Python and every dependency from `uv.lock` on first launch, and the
trained model is committed to the repo (DECISIONS.md). OS-specific guides:
[WSL](docs/setup/Setup_WSL_Windows.md) · [macOS](docs/setup/Setup_macOS.md) ·
[Windows](docs/setup/Setup_Windows.md).

---

## What the App Does

| Page | Purpose |
|------|---------|
| 🏠 **Home** | Project overview, honest metrics, Sensor Logger instructions |
| 🔮 **Predict Exercise** | Upload `WristMotion.csv` or a Sensor Logger ZIP → per-window timeline, **detected sets** (e.g. "2 sets of Bicep Curl"), distribution, results table, CSV download, plus ✏️ correct & improve |
| 📊 **Model Performance** | Leave-one-set-out metrics, per-class F1, confusion matrix, model details, honest limitations |

**Recognized exercises (3 classes):** Bicep Curl · Tricep Extension · Row.
Plus two non-exercise outputs produced *outside* the model: **rest** (energy
gate) and **uncertain** (low confidence).

---

## Key Results

| Metric | Value |
|--------|-------|
| Best model | Random Forest (300 trees, `class_weight='balanced'`, seed 42) |
| Training anchor | **Kaggle Gym Workout IMU** — Apple Watch, 100 Hz, single subject (DECISIONS.md) |
| Evaluation | **Leave-one-set-out** cross-validation (leakage-free; DECISIONS.md) |
| Macro F1 | **0.776** (target ≥ 0.80) |
| Accuracy | 78.2% |
| Per-class F1 | bicep curl 0.76 · row 0.76 · tricep extension 0.81 |

> ⚠️ **Honest limitation.** The training anchor is a **single subject**, so true
> cross-*person* performance cannot be measured and will be **below** these
> numbers. Augmentation (rotation/time-warp/mirror/jitter) synthesises the
> missing subject diversity (DECISIONS.md). The methodology — Apple-Watch training
> domain, leakage-free evaluation, device-invariant features, an energy gate for
> rest, and confidence-based abstention — is sound; the ceiling is data-limited.
> See the **Limitations** section of [`docs/project/project_overview.md`](docs/project/project_overview.md).

---

## Project Structure (high level)

```
app/            Streamlit web app (entry point: app/streamlit_app.py)
src/ml4b/       Reusable package: loaders, windowing, invariant features, gate, model
scripts/        train_model.py (retrain) · inspect_kaggle_dataset.py
notebooks/      One Jupyter notebook per CRISP-DM phase (01–06)
models/saved/   Trained model + model_metrics.json — committed (app runs with no dataset)
data/           Datasets (raw + processed) — NOT in git, except feature_names.txt
docs/           Architecture, DECISIONS.md, CRISP-DM log, setup guides, data dictionary
tests/          Unit tests
```

See [`STRUCTURE.md`](STRUCTURE.md) for the full breakdown and
[`docs/project/project_overview.md`](docs/project/project_overview.md) to
understand the whole project from one document.

---

## Retrain the Model (optional)

Only needed to reproduce the model from raw data — the app ships with it trained.

```bash
# 1. Download the Kaggle Gym Workout IMU dataset (Apple Watch, 100 Hz):
#    https://www.kaggle.com/datasets/shakthisairam123/gym-workout-imu-dataset
#    Unzip the CSV files into data/raw/kaggle_gym_imu/
# 2. Train:
make train        # or: uv run python scripts/train_model.py
```

This rewrites `models/saved/best_model.joblib`, `models/saved/model_metrics.json`
and `data/processed/feature_names.txt`. Uses `random_state=42` throughout. The
dataset choice (Kaggle over the abandoned RecoFit / MM-Fit) is documented in
DECISIONS.md and [`docs/data_understanding/dataset_evaluation.md`](docs/data_understanding/dataset_evaluation.md).

---

## Improve the Model with More Recordings (continual learning)

The base model is trained on a single person, so the most effective improvement is
more of **our own** labelled Apple-Watch sets. We **commit data, not models** — the
model is rebuilt from the committed recordings, so the whole team stays on one model
state (see [DECISIONS.md §8](docs/DECISIONS.md) and
[`docs/project/continual_training.md`](docs/project/continual_training.md)).

1. **Record** one exercise per file (one clean set) with Sensor Logger — see
   [`docs/project/apple_watch_data_collection_guide.md`](docs/project/apple_watch_data_collection_guide.md).
2. **Commit** it into the folder for that exercise — the **folder name** is the
   label: `Testdaten/Biceps_Curls/`, `Testdaten/Rows/`, `Testdaten/Triceps_Extensions/`.
3. **Rebuild** (one person with the Kaggle dataset, or CI):

   ```bash
   make update        # = uv run python scripts/rebuild_from_testdaten.py
   ```

   This retrains the model, refits the novelty detector and refreshes
   `model_metrics.json` from the Kaggle anchor **plus** every committed set, all
   through the same pipeline as initial training (`random_state=42`).
4. **Commit** the rebuilt `models/saved/` so everyone pulls the same model.

This is **retraining** (a full rebuild), not incremental "further training" — the
Random Forest has no `partial_fit`; rebuilding from scratch keeps it reproducible and
avoids catastrophic forgetting. Roll back with `git checkout HEAD~1 -- models/saved/`.

`Rest/` and `Uncertain/` recordings are **not** training classes — they validate the
rest gate and the open-set `unknown` rejection respectively (see the guide).

---

## Development

```bash
make test         # uv run pytest
make lint         # uv run ruff check .
make format       # uv run ruff format .
```

Branch workflow: `main → develop → feature/xxx`. Never commit directly to
`main`. Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, …) and
pair code with its documentation.

---

## Course Context

- **Course:** ML4B (Machine Learning for Business), SoSe 2026
- **Methodology:** CRISP-DM (Business Understanding → … → Deployment)
- **Dataset:** Kaggle Gym Workout IMU (Apple Watch) — DECISIONS.md; RecoFit and MM-Fit
  were evaluated and abandoned for device-domain mismatch
- **Deliverable:** Streamlit web application + presentation
