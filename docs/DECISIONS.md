# Project Decisions — ML4B Gym Exercise Recognition

**Status:** Living document · **Last updated:** 2026-06-25

Single, consolidated record of the project's important decisions — *what* was
decided and *why*, grouped by theme. When a new significant decision is made,
**add or update an entry below** (don't create separate files). Keep entries to
decision + rationale + the main rejected alternative; minor implementation
choices belong in code comments.

A condensed, presentation-oriented summary for a handover/presenting team lives
in `docs/Handoff.md` — keep it in sync when a decision here changes.

---

## 1. Environment & Tooling

**Manager: `uv` (Astral).** `pyproject.toml` is the single source of truth,
`uv.lock` pins every transitive dependency, `uv sync` reproduces the environment,
`uv run <cmd>` runs tools without manual venv activation. Chosen over conda
(heavyweight/licensing), pip+venv (no lockfile) and poetry (slower) for fast,
reproducible cross-platform (WSL/macOS/Windows) setup.

**One-command launch.** From a fresh clone: `uv run streamlit run
app/streamlit_app.py` — `uv run` provisions Python and dependencies on first run
(no separate sync/pip/conda/Docker); `make run` / `run_app.sh` / `run_app.bat`
wrap it. With the trained model committed (§3) the repo is handover-ready: clone →
run, no dataset download.

## 2. ML Framework & Model

**Framework: scikit-learn.** Each sensor window becomes a fixed feature vector, so
this is classical tabular classification — scikit-learn's `fit/predict`, leak-free
`Pipeline` objects and `joblib` serialisation are the fastest path to a model the
app loads directly. Deep learning (PyTorch/TF) was rejected as overkill for
engineered features and a mixed-experience team (a future CNN/LSTM would be
recorded here).

**Final model: Random Forest.** Compared against XGBoost and SVM-RBF by macro-F1:
RF won the metric while being fastest to train/serve, natively multi-class on
string labels and interpretable via `feature_importances_`. XGBoost kept as a
documented backup; SVM rejected (lower F1, slow Platt scaling).

**RF regularisation.** `n_estimators=300, max_depth=20, min_samples_leaf=4,
min_samples_split=5, class_weight='balanced', random_state=42`. Capping depth (vs
unconstrained ~28-deep trees that memorised the training set) costs almost no
validation F1 but yields better-calibrated, less over-confident probabilities —
important because the app shows them as "confidence" and the model must transfer
across devices.

## 3. Dataset & Artifacts

**Training anchor: Kaggle "Gym Workout IMU"** — recorded on an **Apple Watch SE at
100 Hz (left wrist)**, the same device family as deployment. Matching the
deployment *device* is what made real uploads work. It is **single-subject**, the
project's central limitation (§6).

*Why not the earlier datasets:*
- **RecoFit (Microsoft)** — original anchor (6 classes), but a **forearm armband**,
  not a wrist watch: a sensor-placement gap no preprocessing could close, so real
  Apple-Watch curls were misclassified. Abandoned.
- **MM-Fit** — fixed the placement (wrist smartwatch, 7 classes), scored well
  in-domain, but a **Wear-OS TicWatch**, leaving a device/orientation gap to the
  Apple Watch. Superseded by the Kaggle anchor.

**Target classes: three** — `bicep_curl`, `tricep_extension`, `row`, each grouping
biomechanically equivalent Kaggle exercises. They span three distinct movement axes
(elbow flexion / overhead extension / horizontal pull) for the cleanest decision
boundaries and are gym-reproducible by any tester. More classes were rejected
because each one lowers separability on a single-subject anchor (`push_up` is absent
here; `lateral_raise`/`shoulder_press` overlap the curl/triceps signals).
`rest`/`uncertain`/`unknown` are **not** trained classes — they come from the
inference safeguards (§5).

**Commit the trained model + metrics.** `best_model.joblib`,
`random_forest.joblib`, `novelty_detector.joblib`, `model_metrics.json` and
`feature_names.txt` are committed (via `.gitignore` negations) so the app runs after
a clone with no 1–2 GB download. Chosen over Git LFS / cloud download /
regenerate-on-first-run, which all break zero-friction handover. Reproducibility is
preserved — `scripts/train_model.py` regenerates the identical model
(`random_state=42`).

## 4. Signal Preprocessing & Features

**Sliding window: 2 s, 50 % overlap.** 2 s is the canonical wrist-HAR window — long
enough for one rep of the slowest exercise, short enough to rarely straddle an
exercise/rest transition; 50 % overlap roughly doubles the sample count at the HAR
standard. Recordings are resampled to a uniform 100 Hz grid before windowing, so a
window is always 2 s (200 samples) regardless of the source device's rate.

**Apple-Watch (Sensor Logger) loader + canonical units.** Auto-detects the Sensor
Logger `WristMotion.csv` columns, reconstructs **total acceleration**
(`userAcceleration + gravity`) to match the training canonicalisation, keeps the
gyroscope in **rad/s**, and resamples to the training grid. The *identical*
loader/feature code runs in training and the app — the core shared-pipeline rule
(no duplicated preprocessing logic).

**Device-invariant features (~39).** Built from quantities robust to how the watch
sits on the wrist: magnitude statistics on `|accel|`/`|gyro|`, per-window
z-normalised shape features (zero-crossing rate, dominant frequency), axis-pair
correlations, and the gyro/accel ratio. Raw per-axis statistics were dropped from
the final model — they depend on handedness/strap/orientation and drove
bicep↔triceps confusion on real recordings. Invariant features are the cheapest
reliable way to transfer a single-subject model to new users.

**Augmentation as a subject-diversity substitute.** Because the anchor is one
subject, each training window gets 5 augmented copies (→6×): random 3-D rotation
(orientation/handedness), time-warp (rep tempo), axis mirror (other wrist) and
per-axis jitter (sensor/body noise) — synthesising the variability a multi-subject
dataset would provide. (An earlier rotation-only experiment on *multi-subject
MM-Fit* was unhelpful there; the single-subject case is different, hence composed
augmentation now.) Only training windows are augmented; held-out sets stay pristine
and their augmented copies are excluded from training folds, so evaluation stays
leak-free.

## 5. Inference Safeguards — What the App Actually Outputs

The model is a closed-set 3-class classifier; these post-hoc layers stop it from
confidently emitting a wrong label on real, open-ended recordings:

- **Activity gate → `rest`.** Rest is detected by an **energy threshold** (accel
  magnitude std > 0.08 g *or* gyro magnitude mean > 0.30 rad/s), not a trained class
  — a learned `rest` over-predicted on real uploads and the Kaggle anchor has no
  rest data. Windows below both thresholds are `rest` and never reach the model.
  *Calibration:* the thresholds must stay **below** real exercise energy (upper
  bound, from Kaggle) **and above** genuine rest energy (lower bound, from committed
  `data/Testdaten/Rest/`). `scripts/calibrate_gate.py` (`make calibrate`) measures
  both distributions and recommends a threshold in the gap, so rest recordings
  *calibrate the gate* rather than becoming a class (~90 % of Kaggle exercise
  windows clear the gate; the rest are low-energy pauses).
- **Confidence threshold → `uncertain`.** A top class probability below `0.50` is
  reported `uncertain` instead of a forced class — explicit abstention.
- **Novelty detection → `unknown`.** An optional per-class Gaussian + Mahalanobis
  detector (Ledoit-Wolf covariance, 99th-percentile per-class threshold) rejects
  windows unlike any known class, so untrained exercises (squats, etc.) aren't
  confidently mislabelled. Absent artifact ⇒ pipeline unchanged.
- **Session / bout summary.** Per-window labels are folded into **bouts** (maximal
  runs of non-`rest` windows; `rest` separates sets) and shown as a "Detected Sets"
  table — users think in sets, and the vote smooths isolated single-window errors.
- **Plurality rule.** The whole-recording "Overall Result" and each bout's label use
  the most frequent label with `rest` excluded but `uncertain`/`unknown` **counted**
  — so a mostly-unsure set reads as "uncertain" rather than surfacing a minority
  confident exercise. We never over-claim an exercise the model didn't clearly
  predict.

## 6. Evaluation

**Leak-free metric: leave-one-set-out cross-validation.** Each Kaggle file is one set
(group); for every held-out set the model trains on all others (incl. their
augmentations) and is tested only on that set's original windows — the strongest
leak-free estimate for a single subject (macro-F1 ≈ 0.78). Random window/k-fold
splits leak same-set windows and inflate scores — rejected.

**Central limitation — single subject.** True leave-one-*subject*-out is impossible
(one person, one watch). Real-world accuracy on a **new person** is expected *below*
the reported number; this is documented prominently. Invariant features and
augmentation (§4) are the mitigations; a few of the user's own labelled recordings
are the robust fix (motivating §8).

*Earlier datasets used subject-based splits and majority-class (`rest`)
undersampling — superseded by leave-one-set-out on the single-subject Kaggle anchor,
but the macro-F1-first principle remains.*

## 7. Documentation & Reproducibility

**Code/doc standard.** Every `.py` has a module docstring; every function/class a
Google-style docstring; non-trivial blocks WHAT/WHY comments; notebook markdown
explains each cell. The project must be explainable to a new team and to non-ML
reviewers — `docs/Handoff.md` is the condensed handover summary.

**Notebooks track the live pipeline.** The CRISP-DM notebooks import from
`src/ml4b/` and orchestrate (never duplicate) the 3-class Apple-Watch pipeline, and
run top-to-bottom — the handover narrative, which must agree with the code.

**Reproducibility.** `random_state=42` everywhere; `uv.lock` committed; the model is
regenerated deterministically by `scripts/train_model.py`.

## 8. Continual Learning — Folder-Based Retraining

The single-subject limitation (§6) means the base model transfers imperfectly to a
new person; the robust fix is more of our **own** labelled Apple-Watch recordings.
Because the Kaggle anchor is itself Apple Watch SE data (§3), our Sensor Logger
recordings are the **same device domain** — they extend the anchor rather than
mixing in a foreign source.

**Commit data, not models.** Everyone commits *recordings*; the model is a build
artifact regenerated from them. This cures the "every laptop has a different model"
divergence: one deterministic build over committed data converges the whole team on
one model state. Only the build output (`models/saved/`) is committed back — by one
person or CI, never hand-edited.

- **Collect.** One exercise per file (one clean set), committed under
  `data/Testdaten/<Exercise>/`. The **folder name** sets the label (`Biceps_Curls*` →
  `bicep_curl`, `Rows*` → `row`, `Triceps_Extensions*` → `tricep_extension`), so
  filenames are free-form.
- **Rebuild.** `scripts/rebuild_from_testdaten.py` (`make update`) windows + gates
  every recording, concatenates it with the Kaggle anchor, and runs the *identical*
  windowing → augmentation → invariant-feature → Random Forest pipeline as initial
  training; it also refits the novelty detector on base + Testdaten and recomputes
  the leave-one-set-out metrics.

**Retraining, not incremental "further training".** The Random Forest has no
`partial_fit`; every update rebuilds from scratch. Deliberate — incremental updates
from a few samples risk catastrophic forgetting, whereas a full rebuild reuses the
tested pipeline and is reproducible (same commit → same model). See
`docs/project/continual_training.md`.

**Rest and Uncertain are not training classes.** `Rest/` recordings validate the
energy gate (§5); `Uncertain/` holds *other* exercises that validate open-set
rejection (the novelty detector should flag them `unknown`). We deliberately do not
train an "everything-else" class — it only memorises the few foreign exercises
recorded, whereas the novelty detector rejects unseen foreign motion too.

**Superseded.** The earlier in-app "Correct & Improve" editor and the
`feedback.jsonl` loop are removed — they produced per-laptop local state (the
divergence above). `add_labelled_recording.py` / `update_model.py` remain as
lower-level building blocks, but `rebuild_from_testdaten.py` is the canonical full
rebuild. Retraining stays an explicit **offline** step; roll back with
`git checkout HEAD~1 -- models/saved/`.

## 9. Two-Model Comparison — Showing the Effect of Our Own Data

**Decision.** Ship **two** models and run both on every upload:
- **Model 1 (baseline)** — Kaggle anchor only (`baseline_model.joblib` +
  `baseline_novelty_detector.joblib` + `baseline_metrics.json`).
- **Model 2 (current)** — Kaggle **+ our committed Testdaten** (`best_model.joblib` +
  `novelty_detector.joblib` + `model_metrics.json`).

The Predict page runs both on the same recording (timelines, distributions and a
per-window comparison side by side); the Model Performance page shows their CV F1
next to each other. This makes the value of our own data **visible** instead of
asserted.

**Same pipeline, only the data differs.** Testdaten is concatenated with the anchor
*before* the 6× augmentation (§4), and both models come from one shared module
(`src/ml4b/models/pipeline.py`) — identical windowing, augmentation, features, RF
hyper-parameters and novelty calibration. That is what makes the comparison fair.
`make update` rebuilds both in one deterministic run (the baseline never drifts);
`make train` writes the Kaggle model into *both* slots, so before any Testdaten the
two are identical ("no own-data effect yet").

**Caveat (surfaced in the app).** The two CV macro-F1 numbers are not a clean
apples-to-apples improvement — each is evaluated over its own held-out sets, and
Model 2's pool also contains our own (cross-person, harder) recordings, so the
aggregate can even dip slightly. The like-for-like comparison is the
**same-recording** view on the Predict page; both pages state this. Showing only one
model (simpler) would hide whether the extra data helped — rejected; a single
blended metric was rejected as misleading.
