# ML4B — Gym Exercise Recognition

**ML4B SoSe 2026 · FAU Nürnberg · Lehrstuhl für Wirtschaftsinformatik**

Recognize gym exercises from wrist-worn sensor data (Apple Watch accelerometer
+ gyroscope, 50 Hz) using machine learning. Record a workout with the **Sensor
Logger** app, upload the file to the Streamlit web app, and get the recognized
exercise per 2-second window with a confidence score.

---

## Quickstart — 3 Commands

```bash
git clone git@github.com:AnshulAgrawal7/ml4b-project.git
cd ml4b-project
uv sync
uv run streamlit run app/streamlit_app.py
```

→ Open **http://localhost:8501**

**No dataset download needed** — the trained model
(`models/saved/best_model.joblib`) and feature list
(`data/processed/feature_names.txt`) are included in the repository.

OS-specific instructions: [`docs/setup/Setup_WSL_Windows.md`](docs/setup/Setup_WSL_Windows.md),
[`docs/setup/Setup_macOS.md`](docs/setup/Setup_macOS.md),
[`docs/setup/Setup_Windows.md`](docs/setup/Setup_Windows.md).

---

## What the App Does

| Page | Purpose |
|------|---------|
| 🏠 **Home** | Project overview, headline metrics, Sensor Logger instructions |
| 🔮 **Predict Exercise** | Upload `WristMotion.csv` or a Sensor Logger ZIP → per-window predictions, timeline, distribution, downloadable results |
| 📊 **Model Performance** | Test metrics, model comparison, per-class F1, confusion matrix |

**Recognized exercises (7 classes):** Bicep Curl · Shoulder Press · Squat ·
Tricep Extension · Lateral Raise · Push Up · Rest.

---

## Key Results

| Metric | Value |
|--------|-------|
| Best model | Random Forest |
| Training dataset | **MM-Fit** (wrist-worn smartwatch — see ADR-013) |
| Test Macro F1 | **0.944** ✅ (target ≥ 0.80) |
| Test Accuracy | 97.8% |
| Val Macro F1 | 0.866 |

> **Real Apple Watch note:** the model is now trained on wrist-worn smartwatch
> data (MM-Fit) instead of the forearm-worn RecoFit (ADR-013). Push-ups are
> recognized correctly on real Apple Watch recordings; bicep curls can still be
> confused with tricep extensions (residual device-orientation gap — ADR-013/014).

---

## Project Structure (high level)

```
app/            Streamlit web app (entry point: app/streamlit_app.py)
src/ml4b/       Reusable package: data loaders, windowing, features, models
scripts/        build_mmfit_dataset.py + train_model.py — reproduce the model
notebooks/      One Jupyter notebook per CRISP-DM phase (01–06)
models/saved/   Trained model (best_model.joblib) — committed
data/           Datasets (raw + processed) — NOT in git, except feature_names.txt
docs/           Architecture, ADRs, CRISP-DM log, setup guides, data dictionary
tests/          Unit tests
```

See [`STRUCTURE.md`](STRUCTURE.md) for the full breakdown and
[`docs/project/project_overview.md`](docs/project/project_overview.md) to
understand the whole project from one document.

---

## Retrain the Model (optional)

Only needed to reproduce the model from raw data. The model is trained on the
**MM-Fit** dataset (see ADR-013):

```bash
# 1. Download + unzip the MM-Fit sensor data (~1.7 GB) into data/raw/
curl -L https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip -o data/raw/mm-fit.zip
cd data/raw && unzip mm-fit.zip && cd ../..

# 2. Build the feature CSVs, then train
uv run python scripts/build_mmfit_dataset.py
uv run python scripts/train_model.py
```

The original Phase 1–5 pipeline used RecoFit (`data/raw/recofit/`); it was
superseded by MM-Fit in ADR-013 because RecoFit's sensor was forearm-worn while
the Apple Watch is wrist-worn.

---

## Development

```bash
uv run ruff format .      # format
uv run ruff check .       # lint
uv run pytest             # run tests
```

Branch workflow: `main → develop → feature/xxx`. Never commit directly to
`main`. See [`CLAUDE.md`](CLAUDE.md) for the full contribution workflow.

---

## Course Context

- **Course:** ML4B (Machine Learning for Business), SoSe 2026
- **Methodology:** CRISP-DM (Business Understanding → … → Deployment)
- **Dataset:** MM-Fit (wrist-worn smartwatch, gym exercises) — see ADR-013;
  originally RecoFit (Microsoft Research), superseded for device-match reasons
- **Deliverable:** Streamlit web application + presentation
