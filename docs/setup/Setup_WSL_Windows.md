# Setup Guide — WSL (Ubuntu) on Windows

> ML4B Gym Exercise Recognition | Python 3.11 · uv · VS Code
> This guide takes a brand-new machine to a running Streamlit app.

---

## 0. Quick Start vs Full Setup

Choose your goal — most people only need the **Quick Start** path.

| Goal | Requirements |
|------|-------------|
| **Run the Streamlit app** | `git clone` + `uv sync` — **NO dataset needed** |
| **Explore the notebooks** | `git clone` + `uv sync` — **NO dataset needed** |
| **Retrain the model from scratch** | RecoFit dataset required (~2.5 GB) |

The trained model (`models/saved/best_model.joblib`) and the feature list
(`data/processed/feature_names.txt`) are committed to git, so the app works
immediately after cloning — you do **not** need to download the dataset.

---

## 1. Prerequisites

Open a **WSL (Ubuntu) terminal** and install the three tools below.

| Tool | Download / Install |
|------|--------------------|
| **VS Code** | https://code.visualstudio.com/download (install on Windows, not inside WSL) |
| **Git** | Pre-installed on Ubuntu; otherwise `sudo apt update && sudo apt install -y git` |
| **uv** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` then `source ~/.bashrc` |

Verify:

```bash
git --version      # >= 2.x
uv --version       # any recent version
```

> WSL not installed yet? In Windows PowerShell **as Administrator** run
> `wsl --install -d Ubuntu`, reboot, then create your Linux user.

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

→ Open **http://localhost:8501** in your Windows browser.
WSL forwards `localhost` automatically, so no extra configuration is needed.

You should see three pages: **🏠 Home**, **🔮 Predict Exercise**, and
**📊 Model Performance**.

---

## 4. Dataset Download (only if retraining)

Only needed if you want to reproduce the model from raw data.

Download RecoFit from:
**https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors**

Files needed:

```
exercise_data.50.0000_singleonly.mat   (~2.5 GB)
exercise_data.50.0000_multionly.mat
```

Place them in:

```
data/raw/recofit/
```

---

## 5. Retrain the Model (only if needed)

```bash
uv run python scripts/train_model.py
```

This runs the full pipeline (load → window → features → split → train) and
overwrites `models/saved/best_model.joblib` and `feature_names.txt`. With the
dataset present it takes a few minutes; if processed features already exist it
skips straight to training. Uses `random_state=42` everywhere for
reproducibility.

---

## 6. VS Code Setup

1. Install the **Remote - WSL** extension on Windows so VS Code can open the
   project inside the Linux filesystem:
   ```
   code --install-extension ms-vscode-remote.remote-wsl
   ```
2. From the WSL terminal, inside the repo, open VS Code:
   ```bash
   code .
   ```
3. Install the recommended extensions (run inside the WSL window):
   ```
   code --install-extension ms-python.python
   code --install-extension ms-toolsai.jupyter
   code --install-extension charliermarsh.ruff
   ```
4. Select the Python interpreter: `Ctrl+Shift+P` → **Python: Select
   Interpreter** → choose `./.venv/bin/python`.

---

## 7. Git Setup

```bash
# Create an SSH key (press Enter to accept defaults)
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub        # copy this output

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
   (CSV or ZIP).
5. Transfer the export to your machine and upload it on the app's
   **🔮 Predict Exercise** page.

**What to upload:** either the single **`WristMotion.csv`**, or the **full ZIP**
of the export (the app finds `WristMotion.csv` inside automatically).
See `docs/project/apple_watch_data_collection_guide.md` for the full protocol.

---

## 9. Troubleshooting (WSL-specific)

| Problem | Fix |
|---------|-----|
| `uv: command not found` | Run `source ~/.bashrc` or reopen the terminal after installing uv. |
| `localhost:8501` won't open in Windows browser | Ensure you launched the app from the WSL terminal; try `http://127.0.0.1:8501`. |
| `ModuleNotFoundError: ml4b` | Run commands with `uv run ...` so the project venv is used. |
| Slow file access / git | Keep the repo in the Linux filesystem (`~/projects/...`), **not** under `/mnt/c/`. |
| VS Code opens in Windows mode | Reopen via the green corner button → **Reopen in WSL**, or run `code .` from WSL. |
| App says model not found | Run `uv run python scripts/train_model.py`, or re-clone to get the committed model. |
