# Project Overview — ML4B Gym Exercise Recognition

> **Read this first.** This document explains what this project does, how it
> works, what decisions were made and why, and where to find everything in the
> repository. A new team member should understand the full project from this
> document alone.

---

## 1. What Does This Project Do?

This project recognizes **three gym exercises** from wrist-worn **Apple Watch**
sensor data (accelerometer + gyroscope) using machine learning.

**In plain language:** you go to the gym, record your workout with the **Sensor
Logger** app on your Apple Watch, upload the `WristMotion.csv` to our Streamlit
web app, and the app tells you which exercise you performed in each 2-second
window — with a confidence score.

**The 3 exercises the model recognizes:**
1. **Bicep Curl** — elbow flexion
2. **Tricep Extension** — overhead elbow extension
3. **Row** — horizontal pull

Plus two outputs produced *outside* the model:
- **Rest** — low-motion pauses, detected by an energy gate (DECISIONS.md), not a
  trained class.
- **Uncertain** — an active window the model is not confident about (DECISIONS.md).

---

## 2. Project Context

- **Course:** ML4B (Machine Learning for Business), SoSe 2026
- **University:** FAU Nürnberg, Lehrstuhl für Wirtschaftsinformatik
- **Methodology:** CRISP-DM
- **Final deliverable:** Streamlit web application + presentation

---

## 3. How to Run (one command)

```bash
# Install uv once (macOS/Linux/WSL): curl -LsSf https://astral.sh/uv/install.sh | sh
git clone git@github.com:AnshulAgrawal7/ml4b-project.git
cd ml4b-project
make run        # or ./run_app.sh · run_app.bat · uv run streamlit run app/streamlit_app.py
```

→ Open **http://localhost:8501**. No Python install, no `uv sync`, no dataset
needed — `uv run` provisions everything and the trained model is committed
(DECISIONS.md). OS guides: `docs/setup/Setup_*.md`.

---

## 4. How the ML Pipeline Works

The **same code** runs in training and in the app (the core architectural rule).
Both the Kaggle training data and Sensor Logger uploads are Apple CoreMotion
streams, so the canonicalization is identical on both sides.

**Step 1 — Load**
- Training: `src/ml4b/data/kaggle_loader.py` reads the 3-class Kaggle files.
- App: `src/ml4b/data/apple_watch_loader.py` reads `WristMotion.csv` / ZIP.
- Both canonicalize to total acceleration in g + gyro in rad/s
  (`src/ml4b/data/canonical.py`).

**Step 2 — Resample** to 100 Hz (Apple Watch native rate; `canonical.resample_uniform`).

**Step 3 — Sliding window** — `src/ml4b/data/windowing.py`, 200 samples = 2 s,
50% overlap (DECISIONS.md). Carries `recording_id` so windows can be grouped by set.

**Step 4 — Activity gate** — `src/ml4b/data/activity_gate.py` labels low-energy
windows as `rest` so they never reach the model (DECISIONS.md).

**Step 5 — Invariant features** — `src/ml4b/data/features_invariant.py`, 39
orientation-/offset-robust features: magnitude stats + spectral, per-window
z-normalized shape features, axis-pair correlations (DECISIONS.md).

**Step 6 — Model** — Random Forest (`src/ml4b/models/train.py`),
`class_weight='balanced'`, `random_state=42`.

**Step 7 — Confidence threshold** — predictions below 0.50 top probability are
reported as `uncertain` (DECISIONS.md).

Training augments windows 6× (rotation + time-warp + mirror + jitter) to
synthesise the subject diversity a single-subject dataset lacks (DECISIONS.md).

---

## 5. Key Results

| Metric | Value |
|--------|-------|
| Best model | Random Forest |
| Training data | Kaggle Gym Workout IMU (Apple Watch, 100 Hz, single subject) + our own Testdaten (DECISIONS.md) |
| Evaluation | **Leave-one-set-out** cross-validation (leakage-free; DECISIONS.md) |
| Macro F1 | **0.792** (current, Kaggle + Testdaten) · 0.776 (Kaggle-only baseline) |
| Accuracy | 79.2% (current) · 78.2% (baseline) |
| Per-class F1 (current) | bicep curl 0.80 · row 0.78 · tricep extension 0.79 |
| Training sets | 110 (75 Kaggle + 35 Testdaten); baseline 75 (24 bicep · 21 row · 30 triceps) |

These numbers are produced by `scripts/train_model.py` and stored in
`models/saved/model_metrics.json` (shown live on the Model Performance page).

---

## 6. Limitations (read this honestly)

This is a methodologically sound project with a **data-limited ceiling**:

- **Single-subject training anchor.** The Kaggle dataset is one person on one
  Apple Watch. True leave-one-**subject**-out evaluation is impossible, so the
  reported macro F1 measures generalisation to an unseen **set**, not a new
  **person** (DECISIONS.md).
- **Real-world performance will be below the reported numbers** for a different
  user, because the model has never seen anyone else's movement style.
- **Augmentation substitutes for missing subject diversity** (DECISIONS.md) — a
  documented, standard mitigation when target-domain multi-subject data cannot
  be collected, but not a replacement for real data.
- **No target-domain multi-subject data was available**, and none could be
  collected within the project.

What *is* solid: the training domain matches deployment (Apple Watch → Apple
Watch, DECISIONS.md), evaluation is leakage-free (DECISIONS.md), features are
device-invariant (DECISIONS.md), rest is gated rather than learned (DECISIONS.md), and the
model abstains when unsure (DECISIONS.md).

---

## 7. Key Decisions Made (and Why)

Every decision is recorded in [`docs/DECISIONS.md`](../DECISIONS.md) (with the
section noted). Plain-language summary:

| Decision | What we chose | DECISIONS.md |
|----------|--------------|--------------|
| Package manager | uv | §1 |
| ML framework | scikit-learn | §2 |
| Code/doc standard | Google docstrings + inline comments | §7 |
| Sliding window | 2 s window (200 @ 100 Hz), 50% overlap | §4 |
| Commit model to git | yes — app runs with no dataset | §3 |
| Dataset journey | RecoFit → MM-Fit → **Kaggle Apple-Watch** | §3 |
| **Final 3 classes** | bicep_curl, tricep_extension, row (distinct axes, best coverage) | **§3** |
| **Activity gate** | energy-threshold rest (not a class) | **§5** |
| **Invariant features** | magnitude + z-norm shape + correlations (39) | **§4** |
| **Augmentation** | rotation+time-warp+mirror+jitter as subject-diversity substitute | **§4** |
| **Confidence threshold** | < 0.50 → uncertain | **§5** |
| **Evaluation** | leave-one-set-out (single subject limitation) | **§6** |
| **One-command launch** | `uv run` / Makefile / run_app scripts | **§1** |

The original RecoFit/MM-Fit pipeline that was superseded is summarised in
`DECISIONS.md` §3 and §6 (marked "superseded") and kept for history.

---

## 8. Where to Find Everything

| What you need | Where to look |
|--------------|---------------|
| Project goals and research question | `docs/business_understanding/business_understanding.md` |
| Dataset evaluation and selection | `docs/data/dataset_evaluation.md` |
| All technical decisions | `docs/DECISIONS.mdDECISIONS.md` … `DECISIONS.md` |
| CRISP-DM progress log | `docs/project/crisp_dm_log.md` |
| System architecture (arc42) | `docs/architecture/architecture.md` |
| Sensor columns + features | `docs/data/data_dictionary.md` |
| How to collect Apple Watch data | `docs/project/apple_watch_data_collection_guide.md` |
| Honest sanity-check results | `docs/project/apple_watch_validation_results.md` |
| Folder and file descriptions | `STRUCTURE.md` |
| OS-specific setup | `docs/setup/Setup_*.md` |
| Reusable ML code | `src/ml4b/` |
| Trained model + metrics | `models/saved/best_model.joblib`, `model_metrics.json` |
| Streamlit app | `app/streamlit_app.py` |

---

## 9. Dataset

**Kaggle "Gym Workout IMU Dataset" (current training anchor — DECISIONS.md)**
- Apple Watch SE, **left wrist**, **100 Hz**, accelerometer + gyroscope.
- 164 single-set CSV files; 75 of them map to our 3 classes (24 bicep / 21 row /
  30 triceps). Single subject.
- Download: https://www.kaggle.com/datasets/shakthisairam123/gym-workout-imu-dataset
  → unzip into `data/raw/kaggle_gym_imu/`. NOT in git.

**Abandoned sources (kept only for history):**
- **MM-Fit** — non-Apple smartwatch; good test scores but failed to transfer to
  real Apple-Watch uploads (device-domain mismatch). Superseded by Kaggle (DECISIONS.md).
- **RecoFit** (Microsoft Research) — forearm-worn sensor, 50 Hz; superseded
  because the Apple Watch is wrist-worn (DECISIONS.md).

---

## 10. Project Status

| Task | Status |
|------|--------|
| 3-class Apple-Watch model trained + committed | ✅ Done |
| Leave-one-set-out evaluation (honest metrics) | ✅ Done (macro F1 0.792 current · 0.776 baseline) |
| Streamlit app (Home / Predict / Model Performance) | ✅ Done |
| One-command launch on WSL / macOS / Windows | ✅ Done |
| Honest sanity check on real Apple Watch samples | ✅ Done — see `apple_watch_validation_results.md` |
| Cross-subject generalisation | ⚠️ Not achievable (single-subject anchor; documented) |
