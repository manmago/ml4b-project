# Setup Guide — WSL (Ubuntu) on Windows

> ML4B Gym Exercise Recognition · Python managed by **uv** · one-command launch.
> This guide takes a brand-new machine to a running Streamlit app.

---

## TL;DR — run the app in 3 steps

Open a **WSL (Ubuntu)** terminal and run:

```bash
# 1. Install uv (one command — it provides Python and all dependencies for you)
curl -LsSf https://astral.sh/uv/install.sh | sh
#    The installer adds uv to your PATH, but the CURRENT shell won't see it yet.
#    Open a NEW terminal (or reload now: source $HOME/.local/bin/env — or
#    source ~/.bashrc), otherwise you get "uv: command not found".  Check: uv --version

# 2. Clone the repository (HTTPS works for everyone — no SSH key needed)
git clone https://github.com/AnshulAgrawal7/ml4b-project.git
#    SSH alternative (only if you have an SSH key): git clone git@github.com:AnshulAgrawal7/ml4b-project.git
cd ml4b-project

# 3. Launch the app
make run            # or:  ./run_app.sh   or:  uv run streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** (WSL forwards localhost automatically). That's it.

**You do NOT need to:** install Python, create a venv, run `uv sync`, run `pip`,
use conda, or download any dataset. The first `uv run` automatically provisions
the right Python and all pinned dependencies from `uv.lock`; the trained model is
committed, so the app works immediately after cloning (see DECISIONS.md).

> WSL not installed yet? In Windows PowerShell **as Administrator**:
> `wsl --install -d Ubuntu`, reboot, then create your Linux user.

---

## Why only `uv`?

`uv run` reads `pyproject.toml` / `uv.lock`, creates the virtual environment,
installs the exact pinned dependencies, and runs the command — in one step. The
first launch takes ~30–60 s while it sets up; every launch afterwards is instant.
If `uv` is not found after install, ensure `~/.local/bin` is on your `PATH`
(restart the terminal, or `source $HOME/.local/bin/env`).

---

## Optional — only if you need more

<details>
<summary><b>Retrain the model from scratch</b></summary>

The app ships with a trained model, so this is only for reproducing it.

1. Download the **Kaggle Gym Workout IMU Dataset** (Apple Watch, 100 Hz):
   https://www.kaggle.com/datasets/shakthisairam123/gym-workout-imu-dataset
2. Unzip the CSV files into `data/raw/kaggle_gym_imu/`.
3. Run `make train` (or `uv run python scripts/train_model.py`). This rewrites
   `models/saved/best_model.joblib`, `model_metrics.json` and
   `data/processed/feature_names.txt`. Uses `random_state=42`. See DECISIONS.md.
</details>

<details>
<summary><b>Record your own Apple Watch data (Sensor Logger)</b></summary>

See `docs/project/apple_watch_data_collection_guide.md`. In short: record with
the free **Sensor Logger** iOS app (Device Motion enabled), export, and upload
the single **`WristMotion.csv`** (or the whole ZIP) on the app's Predict page.
</details>

<details>
<summary><b>VS Code setup</b></summary>

1. Install **VS Code** on Windows and the **Remote - WSL** extension.
2. From the repo in WSL: `code .`
3. Install the **Python**, **Jupyter** and **Ruff** extensions.
4. Select the interpreter `./.venv/bin/python` (created on the first `uv run`).
5. Run dev tools with `make test`, `make lint`, `make format`.
</details>

<details>
<summary><b>Git / SSH setup</b></summary>

```bash
sudo apt update && sudo apt install -y git make
ssh-keygen -t ed25519 -C "you@example.com"   # press Enter for defaults
cat ~/.ssh/id_ed25519.pub                      # add this key at GitHub → Settings → SSH keys
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```
Branch workflow: `main → develop → feature/xxx`; never commit to `main`.
</details>

<details>
<summary><b>Troubleshooting</b></summary>

| Symptom | Fix |
|---------|-----|
| `uv: command not found` | Restart the terminal or `source $HOME/.local/bin/env`. |
| `localhost:8501` won't open in Windows | WSL2's localhost forwarding occasionally hangs. **Quick fix:** open the WSL IP instead — run `hostname -I` in WSL and browse to `http://<that-ip>:8501`. **Permanent fix:** in Windows PowerShell run `wsl --shutdown`, reopen the terminal and `make run` again; or enable mirrored networking by adding `[wsl2]` + `networkingMode=mirrored` to `C:\Users\<you>\.wslconfig`, then `wsl --shutdown`. |
| Port 8501 in use | `uv run streamlit run app/streamlit_app.py --server.port 8502`. |
| `make: command not found` | `sudo apt install -y make`, or use `./run_app.sh`. |
| Slow file access / git | Keep the repo in the Linux filesystem (`~/projects/...`), not `/mnt/c/`. |
| First launch is slow | Expected — uv is provisioning the environment once. |
| App says model not found | Re-clone via git (the model is committed under `models/saved/`). |
</details>
