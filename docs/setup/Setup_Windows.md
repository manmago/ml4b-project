# Setup Guide — Windows (native, without WSL)

> ML4B Gym Exercise Recognition · Python managed by **uv** · one-command launch.
> Uses **PowerShell**. (Prefer Linux tooling? See `Setup_WSL_Windows.md`.)

---

## TL;DR — run the app in 3 steps

Open **PowerShell** and run:

```powershell
# 1. Install uv (one command — it provides Python and all dependencies for you)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#    Close and reopen PowerShell afterwards so the PATH update loads — otherwise
#    `uv` is "not recognized" in this window.  Check it works:  uv --version

# 2. Clone the repository (install Git from https://git-scm.com/download/win first)
#    HTTPS works for everyone — no SSH key needed:
git clone https://github.com/AnshulAgrawal7/ml4b-project.git
#    SSH alternative (only if you have an SSH key): git clone git@github.com:AnshulAgrawal7/ml4b-project.git
cd ml4b-project

# 3. Launch the app
.\run_app.bat        # or:  uv run streamlit run app/streamlit_app.py
```

(You can also just **double-click `run_app.bat`** in Explorer.)
Open **http://localhost:8501** (Streamlit usually opens it for you). That's it.

**You do NOT need to:** install Python, create a venv, run `uv sync`, run `pip`,
use conda, or download any dataset. The first `uv run` automatically provisions
the right Python and all pinned dependencies from `uv.lock`; the trained model is
committed, so the app works immediately after cloning (see ADR-022, ADR-011).

> `make` is not standard on Windows — use `.\run_app.bat` or the `uv run`
> command directly.

---

## Why only `uv`?

`uv run` reads `pyproject.toml` / `uv.lock`, creates the virtual environment,
installs the exact pinned dependencies, and runs the command — in one step. The
first launch takes ~30–60 s while it sets up; every launch afterwards is instant.
If `uv` is not recognized, close and reopen PowerShell so the PATH update loads.

---

## Optional — only if you need more

<details>
<summary><b>Retrain the model from scratch</b></summary>

The app ships with a trained model, so this is only for reproducing it.

1. Download the **Kaggle Gym Workout IMU Dataset** (Apple Watch, 100 Hz):
   https://www.kaggle.com/datasets/shakthisairam123/gym-workout-imu-dataset
2. Unzip the CSV files into `data\raw\kaggle_gym_imu\`.
3. Run `uv run python scripts/train_model.py`. This rewrites
   `models\saved\best_model.joblib`, `model_metrics.json` and
   `data\processed\feature_names.txt`. Uses `random_state=42`. See ADR-016.
</details>

<details>
<summary><b>Record your own Apple Watch data (Sensor Logger)</b></summary>

See `docs/project/apple_watch_data_collection_guide.md`. In short: record with
the free **Sensor Logger** iOS app (Device Motion enabled), export, move the
file to your PC (iCloud Drive or email), and upload the single
**`WristMotion.csv`** (or the whole ZIP) on the app's Predict page.
</details>

<details>
<summary><b>VS Code setup</b></summary>

1. Install **VS Code** (Windows User Installer) and open the project: `code .`
2. Install the **Python**, **Jupyter** and **Ruff** extensions.
3. Select the interpreter `.\.venv\Scripts\python.exe` (created on the first
   `uv run`).
</details>

<details>
<summary><b>Git / SSH setup</b></summary>

```powershell
ssh-keygen -t ed25519 -C "you@example.com"   # press Enter for defaults
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | Set-Clipboard   # add at GitHub → SSH keys
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```
Branch workflow: `main → develop → feature/xxx`; never commit to `main`.
</details>

<details>
<summary><b>Troubleshooting</b></summary>

| Symptom | Fix |
|---------|-----|
| `uv` not recognized | Close and reopen PowerShell so the updated PATH loads. |
| Script execution blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
| Port 8501 in use | `uv run streamlit run app/streamlit_app.py --server.port 8502`. |
| `ssh-keygen` not found | Enable the optional "OpenSSH Client" Windows feature, or use Git Bash. |
| Long-path errors | `git config --system core.longpaths true`. |
| First launch is slow | Expected — uv is provisioning the environment once. |
| App says model not found | Re-clone via git (the model is committed under `models\saved\`). |
</details>
