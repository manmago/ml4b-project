# Project Decisions — ML4B Gym Exercise Recognition

**Status:** Living document · **Last updated:** 2026-06-05

This is the single, consolidated record of the project's important decisions —
*what* was decided and *why*. It replaces the former fine-grained per-decision
ADR files (`docs/decisions/ADR-001…025`), which are summarised here grouped by
theme. A traceability table at the end maps every old ADR number to its section.

When a new significant decision is made, **add or update an entry below** — do
not create separate decision files. Keep entries concise: decision + rationale +
the main alternative rejected. Minor implementation choices belong in code
comments, not here.

---

## 1. Environment & Tooling

**Package/environment manager: `uv` (Astral).** `pyproject.toml` is the single
source of truth; `uv.lock` pins every transitive dependency; `uv sync` reproduces
the environment and `uv run <cmd>` runs tools without manual venv activation.
Chosen over conda (heavyweight, licensing), pip+venv (no built-in lockfile) and
poetry (slower) for fast, reproducible, cross-platform (WSL/macOS/Windows) setup.

**One-command launch.** The app starts from a fresh clone with a single command:
`uv run streamlit run app/streamlit_app.py`. `uv run` provisions the environment
(and the right Python) on first run, so no separate `uv sync`, pip, conda or
Docker is needed. `make run` / `./run_app.sh` / `run_app.bat` wrap it for
non-CLI users. This, plus committing the trained model (§3), makes the repo
handover-ready: clone → run, no dataset download.

## 2. ML Framework & Model

**Framework: scikit-learn.** The pipeline turns each sensor window into a fixed
feature vector, making this classical tabular classification — scikit-learn's
`fit/predict` API, `Pipeline` objects (no leakage) and `joblib` serialisation are
the fastest path to a model the app loads directly. Deep learning (PyTorch/TF) was
rejected as overkill for engineered features and a mixed-experience team; the door
stays open for a future CNN/LSTM iteration (would be recorded here).

**Final model: Random Forest.** Three classifiers were compared (Random Forest,
XGBoost, SVM-RBF) by macro-F1. Random Forest won on the metric while also being
the fastest to train/serve, natively multi-class on string labels, and
interpretable via `feature_importances_`. XGBoost is kept as a documented backup;
SVM was rejected (lower F1, slow Platt scaling).

**Random Forest regularisation.** `n_estimators=300`, `max_depth=20`,
`min_samples_leaf=4`, `min_samples_split=5`, `class_weight='balanced'`,
`random_state=42`. Capping depth (vs unconstrained ~28-deep trees that memorised
the training set) costs almost no validation F1 but yields less over-confident,
better-calibrated probabilities — important because the app shows them as
"confidence" and the model must transfer across devices.

## 3. Dataset & Artifacts

**Current training anchor: Kaggle "Gym Workout IMU" dataset** — recorded on an
**Apple Watch SE at 100 Hz (left wrist)**, the same device family as deployment.
This is the decisive choice: matching the deployment *device* is what made real
uploads work. It is **single-subject**, which is the project's central
limitation (see §6).

*Why not the earlier datasets (evolution):*
- **RecoFit (Microsoft)** was the original anchor (6 classes). It is a **forearm
  armband**, not a wrist watch — a physical sensor-placement gap no preprocessing
  could close, so real Apple-Watch curls were misclassified. Abandoned.
- **MM-Fit** fixed the placement (wrist smartwatch, 7 classes incl. push-up) and
  scored well in-domain, but it is a **Wear-OS TicWatch**, leaving a device/
  orientation gap to the Apple Watch. Superseded by the Kaggle Apple-Watch anchor.

**Target classes: three.** `bicep_curl`, `tricep_extension`, `row` — each formed
by grouping biomechanically equivalent Kaggle exercise abbreviations. They span
three distinct movement axes (elbow flexion / overhead extension / horizontal
pull), giving the cleanest decision boundaries, and are gym-reproducible by
anyone testing the app. More classes were rejected because every added class
lowers separability on a single-subject anchor (`push_up` is absent from this
dataset; `lateral_raise`/`shoulder_press` overlap the curl/triceps signals).
`rest`, `uncertain` and `unknown` are **not** trained classes — they are produced
by the inference safeguards (§5).

**Commit the trained model + metrics to git.** `best_model.joblib`,
`random_forest.joblib`, `novelty_detector.joblib`, `model_metrics.json` and
`feature_names.txt` are committed (via `.gitignore` negations) so the app runs
after a clone with no 1–2 GB dataset download. Chosen over Git LFS / cloud
download / regenerate-on-first-run, all of which break the zero-friction
handover. Reproducibility is preserved: `scripts/train_model.py` regenerates the
identical model deterministically (`random_state=42`).

## 4. Signal Preprocessing & Features

**Sliding window: 2 seconds, 50 % overlap.** 2 s is the canonical wrist-HAR
window — long enough for one full repetition of the slowest gym exercise, short
enough to rarely straddle an exercise/rest transition. 50 % overlap roughly
doubles the sample count with moderate redundancy and is the HAR-standard,
keeping results comparable. At the 100 Hz Apple-Watch rate the window is 200
samples; recordings are resampled to a uniform 100 Hz grid before windowing so a
window is always 2 s regardless of the source device's exact rate.

**Apple-Watch (Sensor Logger) loader + canonical units.** The loader auto-detects
the Sensor Logger `WristMotion.csv` columns, reconstructs **total acceleration**
(`userAcceleration + gravity`) to match the training canonicalisation, keeps the
gyroscope in **rad/s**, and resamples to the training grid. The *identical*
loader/feature code runs in training and in the app — the project's core
shared-pipeline rule (no duplicated preprocessing logic).

**Device-invariant features (~39).** Built from quantities robust to how the watch
sits on the wrist: magnitude statistics on `|accel|`/`|gyro|`, per-window
z-normalised shape features (zero-crossing rate, dominant frequency), axis-pair
correlations, and the gyro/accel ratio. Raw per-axis statistics were dropped from
the final model because they depend on handedness/strap/orientation and drove
bicep↔triceps confusion on real recordings. Invariant features are the cheapest,
most reliable way to transfer a single-subject model to new users.

**Augmentation as a subject-diversity substitute.** Because the anchor is one
subject, each training window gets 5 augmented copies (→6×): random 3-D rotation
(watch orientation/handedness), time-warp (rep tempo), axis mirror (other wrist),
and per-axis jitter (sensor/body noise). This synthesises the variability a
multi-subject dataset would provide. (Note: an earlier rotation-only experiment
on the *multi-subject MM-Fit* data was rejected as unhelpful there; the
single-subject situation is different, which is why composed augmentation is used
now.) Only training windows are augmented; held-out sets stay pristine and their
augmented copies are excluded from training folds, so evaluation stays leak-free.

## 5. Inference Safeguards — What the App Actually Outputs

The model is a closed-set 3-class classifier; these post-hoc layers stop it from
confidently emitting a wrong label on real, open-ended recordings:

- **Activity gate → `rest`.** Rest is detected by an **energy threshold** (accel
  magnitude std > 0.08 g *or* gyro magnitude mean > 0.30 rad/s), not a trained
  class. A learned `rest` over-predicted on real uploads and the Kaggle anchor has
  no rest data anyway. Windows below both thresholds are `rest` and never reach
  the model. The same gate is used in training exploration and the app.
  *Calibration:* the thresholds are bounded from two sides — they must stay **below**
  real exercise energy (upper bound, from Kaggle) **and above** genuine rest energy
  (lower bound, from committed `Testdaten/Rest/`). `scripts/calibrate_gate.py`
  (`make calibrate`) measures both distributions and recommends a threshold in the
  gap; rest recordings thus *calibrate the gate* rather than becoming a class. With
  no `Testdaten/Rest/` data the lower bound is unverified (currently ~90% of Kaggle
  exercise windows clear the gate; the rest are low-energy pauses).
- **Confidence threshold → `uncertain`.** If the model's top class probability is
  below `0.50`, the window is reported `uncertain` instead of a forced class —
  honest abstention for a 3-class model on open-ended motion.
- **Novelty detection → `unknown`.** An optional per-class Gaussian + Mahalanobis
  detector (Ledoit-Wolf covariance, 99th-percentile per-class threshold) rejects
  windows unlike any known class as `unknown` so untrained exercises (squats, etc.)
  are not confidently mislabelled. Optional artifact: absent ⇒ pipeline unchanged.
- **Session / bout summary.** Per-window labels are folded into **bouts** (maximal
  runs of non-`rest` windows; `rest` separates sets) and shown as a "Detected
  Sets" table — users think in sets, not 2 s windows, and the vote smooths
  isolated single-window errors.
- **`uncertain`/`unknown` can be the overall result.** The whole-recording
  "Overall Result" and each bout's label use one shared **plurality rule**: the
  most frequent label wins with `rest` excluded but `uncertain`/`unknown`
  **counted**. So a set the model was mostly unsure about reads as "uncertain"
  rather than surfacing a minority confident exercise — we never over-claim an
  exercise the model did not clearly predict.

## 6. Evaluation

**Honest metric: leave-one-set-out cross-validation.** Each Kaggle file is one set
(group); for every held-out set the model trains on all other sets (incl. their
augmentations) and is tested only on that set's original windows. This is the
strongest leak-free estimate available for a single subject (leave-one-set-out
macro-F1 ≈ 0.78). Random window/k-fold splits leak same-set windows and inflate
scores — rejected.

**Central limitation — single subject.** True leave-one-*subject*-out is
impossible (one person, one watch). Real-world accuracy on a **new person** is
expected to be *below* the reported number; this is documented prominently. The
invariant features (§4) and augmentation (§4) are the mitigations; a small set of
the user's own labelled recordings is the robust fix (and motivates the
correction/continual-learning loop).

*Earlier datasets used subject-based train/val/test splits, majority-class
(`rest`) undersampling, and macro-F1 as primary — superseded by leave-one-set-out
on the single-subject Kaggle anchor, but the macro-F1-first principle remains.*

## 7. Documentation & Reproducibility

**Code/doc standard.** Every `.py` file has a module docstring; every function/
class has a Google-style docstring; non-trivial blocks have WHAT/WHY comments;
notebook markdown explains each cell. The project must be explainable to a new
team and to non-ML reviewers.

**Notebooks track the live pipeline.** The CRISP-DM notebooks import from
`src/ml4b/` and orchestrate (never duplicate) the current 3-class Apple-Watch
pipeline, and run top-to-bottom. They are the handover narrative and must agree
with the code they document.

**Reproducibility.** `random_state=42` everywhere; `uv.lock` committed; the model
is regenerated deterministically by `scripts/train_model.py`.

## 8. Continual Learning — Folder-Based Retraining

The single-subject limitation (§6) means the base model transfers imperfectly to a
new person; the robust fix is more of our **own** labelled Apple-Watch recordings.
Crucially, the Kaggle anchor is itself Apple Watch SE / CoreMotion data (§3), so our
Sensor Logger recordings are the **same device domain** — they extend the anchor
rather than mixing in a foreign source.

**Contribution model — commit data, not models.** Everyone commits *recordings*; the
model is a **build artifact** regenerated from them. This is the cure for the "every
laptop has a different model" divergence: with one deterministic build over committed
data, the whole team converges on one model state. Only the build output
(`models/saved/`) is committed back — by one person or CI, never hand-edited.

- **Collect.** Record one exercise per file (one clean set) and commit it under
  `Testdaten/<Exercise>/`. The **folder name** sets the label (`Biceps_Curls*` →
  `bicep_curl`, `Rows*` → `row`, `Triceps_Extensions*` → `tricep_extension`), so
  filenames are free-form.
- **Rebuild.** `scripts/rebuild_from_testdaten.py` (`make update`) windows + gates
  every committed recording, concatenates it with the Kaggle anchor, and runs the
  *identical* windowing → augmentation → invariant-feature → Random Forest pipeline
  as initial training. It also **refits the novelty detector** on base + Testdaten
  (so our own exercises read as `known`, not `unknown`) and recomputes the honest
  leave-one-set-out metrics in `model_metrics.json`.

**This is retraining, not incremental "further training".** The Random Forest has no
`partial_fit`; every update rebuilds the model from scratch on the current data. That
is deliberate — incremental updates from a few samples risk catastrophic forgetting,
whereas a full rebuild reuses the tested pipeline and is fully reproducible (same
commit → same model). See `docs/project/continual_training.md`.

**Rest and Uncertain are not training classes.**
- `Rest/` recordings validate the energy activity gate (§5): rest is detected by a
  device-agnostic threshold, not a learned class, which transfers far better across
  people (a learned rest class over-predicts on real uploads).
- `Uncertain/` holds recordings of *other* exercises. They validate open-set
  rejection — the novelty detector (§5) should flag them `unknown`. We deliberately
  do **not** train an "everything-else" class: it can only memorise the few foreign
  exercises recorded and generalises badly, whereas the novelty detector rejects
  unseen foreign motion too.

**Superseded.** The earlier in-app "✏️ Correct & Improve" editor and the
correction-driven `feedback.jsonl` loop are removed: they produced per-laptop local
state (git-ignored) — exactly the divergence above. `add_labelled_recording.py` and
`update_model.py` remain as lower-level building blocks, but
`rebuild_from_testdaten.py` is the canonical full rebuild. Retraining stays an
explicit **offline** step, never live during a demo; roll back any rebuild with
`git checkout HEAD~1 -- models/saved/`.

---

## Traceability — former ADRs → sections

| Former ADR | Topic | Section |
|------------|-------|---------|
| ADR-001 | Package manager (uv) | §1 |
| ADR-002 | ML framework (scikit-learn) | §2 |
| ADR-004 | Code/doc standard | §7 |
| ADR-005 | Exercise class selection (RecoFit 6) | §3 (superseded) |
| ADR-006 | Sliding window params | §4 |
| ADR-007 | Subject-based split | §6 (superseded) |
| ADR-008 | Undersampling strategy | §6 (superseded) |
| ADR-009 | Model selection (3 candidates) | §2 |
| ADR-010 | Random Forest as final model | §2 |
| ADR-011 | Commit model to git | §3 |
| ADR-012 | Apple-Watch mapping + resampling | §4 |
| ADR-013 | Switch dataset RecoFit → MM-Fit | §3 (superseded) |
| ADR-014 | Rotation augmentation rejected (MM-Fit) | §4 (superseded by ADR-019) |
| ADR-015 | RF regularisation + rest rebalancing | §2 |
| ADR-016 | Final 3 classes (Kaggle Apple-Watch) | §3 |
| ADR-017 | Activity-gate rest detection | §5 |
| ADR-018 | Device-invariant features | §4 |
| ADR-019 | Augmentation as subject-diversity substitute | §4 |
| ADR-020 | Confidence threshold → uncertain | §5 |
| ADR-021 | Leave-one-set-out evaluation | §6 |
| ADR-022 | One-command launch | §1 |
| ADR-023 | Notebooks aligned to 3-class pipeline | §7 |
| ADR-024 | Novelty detection → unknown | §5 |
| ADR-025 | Bout/session summary | §5 |
| ADR-026 | `uncertain`/`unknown` can be the overall/per-set result | §5 |
| ADR-027 | Human-in-the-loop correction & continual learning | §8 |
