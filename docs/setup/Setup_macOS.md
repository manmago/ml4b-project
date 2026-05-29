# Setup Guide — macOS

> ML4B Gym Exercise Recognition | Python 3.11 · uv · VS Code
> This guide takes a brand-new Mac to a running Streamlit app.

---

## 0. Quick Start vs Full Setup

Choose your goal — most people only need the **Quick Start** path.

| Goal | Requirements |
|------|-------------|
| **Run the Streamlit app** | `git clone` + `uv sync` — **NO dataset needed** |
| **Explore the notebooks** | `git clone` + `uv sync` — **NO dataset needed** |
| **Retrain the model from scratch** | MM-Fit dataset required (~1.7 GB) — ADR-013 |

The trained model (`models/saved/best_model.joblib`) and the feature list
(`data/processed/feature_names.txt`) are committed to git, so the app works
immediately after cloning — you do **not** need to download the dataset.

---

## 1. Prerequisites

Open the **Terminal** app (or iTerm) and install the three tools below.

| Tool | Download / Install |
|------|--------------------|
| **VS Code** | https://code.visualstudio.com/download (Apple Silicon or Intel build) |
| **Git** | Comes with Xcode CLT: `xcode-select --install` |
| **uv** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` then `source ~/.zshrc` |

Verify:

```bash
git --version      # >= 2.x
uv --version       # any recent version
```

> Prefer Homebrew? `brew install uv` and `brew install git` also work.

---

## 2. Clone and Setup

```bash
git clone git@github.com:AnshulAgrawal7/ml4b-project.git
cd ml4b-project
uv sync
```

`uv sync` creates a `.venv/` and installs every pinned dependency from
`uv.lock` — reproducible and identical across all machines.

---

## 3. Run the Streamlit App

```bash
uv run streamlit run app/streamlit_app.py
```

→ Open **http://localhost:8501** in your browser (Streamlit usually opens it
for you).

You should see three pages: **🏠 Home**, **🔮 Predict Exercise**, and
**📊 Model Performance**.

---

## 4. Dataset Download (only if retraining)

Only needed if you want to reproduce the model from raw data. The model is
trained on **MM-Fit** (wrist-worn smartwatch — see ADR-013).

```bash
curl -L https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip -o data/raw/mm-fit.zip
cd data/raw && unzip mm-fit.zip && cd ../..   # creates data/raw/mm-fit/w00 … w20
```

(The original RecoFit `.mat` dataset, used in Phases 1–5, was superseded by
MM-Fit in ADR-013.)

---

## 5. Retrain the Model (only if needed)

```bash
uv run python scripts/build_mmfit_dataset.py   # MM-Fit → processed feature CSVs
uv run python scripts/train_model.py           # train + save best_model.joblib
```

This runs the full pipeline (load → window → features → split → train) and
overwrites `models/saved/best_model.joblib` and `feature_names.txt`. With the
dataset present it takes a few minutes; if processed features already exist it
skips straight to training. Uses `random_state=42` everywhere for
reproducibility.

---

## 6. VS Code Setup

1. Open the project:
   ```bash
   code .
   ```
   (If `code` is not found: in VS Code press `Cmd+Shift+P` →
   **Shell Command: Install 'code' command in PATH**.)
2. Install the recommended extensions:
   ```
   code --install-extension ms-python.python
   code --install-extension ms-toolsai.jupyter
   code --install-extension charliermarsh.ruff
   ```
3. Select the Python interpreter: `Cmd+Shift+P` → **Python: Select
   Interpreter** → choose `./.venv/bin/python`.

---

## 7. Git Setup

```bash
# Create an SSH key (press Enter to accept defaults)
ssh-keygen -t ed25519 -C "your_email@example.com"
pbcopy < ~/.ssh/id_ed25519.pub   # copies the public key to the clipboard

# Add the key at: GitHub → Settings → SSH and GPG keys → New SSH key
ssh -T git@github.com            # verify authentication

# Identify yourself for commits
git config --global user.name  "Your Name"
git config --global user.email "your_email@example.com"
```

**Branch workflow** (see CLAUDE.md): `main → develop → feature/xxx`.
Never commit directly to `main`.

```bash
git checkout develop
git checkout -b feature/your-feature-name
```

---

## 8. Sensor Logger Setup (Apple Watch data collection)

1. Install **Sensor Logger** (free) from the iOS App Store — also install it on
   your **Apple Watch**.
2. In Sensor Logger, enable the **Wrist Motion** sensor
   (accelerometer + gyroscope, 50 Hz).
3. Start a recording, perform your gym exercises, then stop.
4. Export: tap the recording → **Share / Export** → **Save to Files**
   (CSV or ZIP). On a Mac you can also AirDrop the export straight from the
   Watch/iPhone.
5. Upload it on the app's **🔮 Predict Exercise** page.

**What to upload:** either the single **`WristMotion.csv`**, or the **full ZIP**
of the export (the app finds `WristMotion.csv` inside automatically).
See `docs/project/apple_watch_data_collection_guide.md` for the full protocol.

---

## 9. Troubleshooting (macOS-specific)

| Problem | Fix |
|---------|-----|
| `uv: command not found` | Run `source ~/.zshrc` or reopen Terminal after installing uv. |
| `xcrun: error: invalid active developer path` | Run `xcode-select --install`. |
| `code: command not found` | In VS Code: `Cmd+Shift+P` → install the `code` shell command. |
| `ModuleNotFoundError: ml4b` | Run commands with `uv run ...` so the project venv is used. |
| Apple Silicon build issues | uv installs native arm64 wheels automatically — just re-run `uv sync`. |
| App says model not found | Run `uv run python scripts/train_model.py`, or re-clone to get the committed model. |
