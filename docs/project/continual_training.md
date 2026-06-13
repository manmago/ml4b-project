# Continual Training — How We Keep Improving the Model

> Short version: drop one clean set per file into `Testdaten/<Exercise>/`, commit it,
> run `make update`, commit the rebuilt `models/saved/`. Everyone then pulls the same
> model. See [DECISIONS.md §8](../DECISIONS.md) for the rationale.

This guide explains how the team adds data and regenerates the model up to (and
after) the demo, and answers a question that comes up a lot:

## Is this "further training" or "retraining"?

**It is retraining — a full rebuild from scratch — not incremental "further
training".**

Our model is a scikit-learn **Random Forest**, which has **no `partial_fit`**: you
cannot feed it a few new samples and have it nudge its existing trees. The only sound
way to teach it new data is to **rebuild the whole forest** on *all* the data we have
(the Kaggle anchor **plus** every committed recording).

We do this on purpose, not as a workaround:

- **Incremental updates from a handful of samples cause *catastrophic forgetting*** —
  the model would over-fit the few newest examples and lose the rest.
- **A full rebuild is reproducible.** Same commit (same data) → byte-for-byte the
  same model, because everything is seeded (`random_state=42`). That reproducibility
  is what keeps the whole team on **one** model state instead of N divergent local
  ones.

So when we say "continual learning" in this project, we mean: *the dataset grows
continually, and we retrain from scratch each time it does.* The **effect** is a
model that keeps getting better; the **mechanism** is retraining, not online learning.

| | "Further training" (online) | **What we do (retraining)** |
|---|---|---|
| Mechanism | `partial_fit` on new data only | rebuild the forest on **all** data |
| Old knowledge | can be forgotten | always preserved (all data present) |
| Reproducible | no (depends on update order) | **yes** (same commit → same model) |
| Fits Random Forest? | no (`partial_fit` unsupported) | **yes** |

## The workflow

### 1. Record (one clean set per file)
- Record **one** exercise per file (one set), following
  [`apple_watch_data_collection_guide.md`](apple_watch_data_collection_guide.md).
- Single sets keep labels unambiguous — that is why we train on them.

### 2. Commit the recording into the right folder
The **folder name** is the label (filenames can be anything):

| Put the recording in… | Trains the class |
|---|---|
| `Testdaten/Biceps_Curls/` | `bicep_curl` |
| `Testdaten/Rows/` | `row` |
| `Testdaten/Triceps_Extensions/` | `tricep_extension` |
| `Testdaten/Rest/` | *(not a class — validates the rest gate)* |
| `Testdaten/Uncertain/` | *(not a class — validates `unknown` rejection)* |

A Sensor Logger export is a folder containing `WristMotion.csv` (the rebuild reads
that file); a `.zip` export dropped directly in the folder also works.

```bash
git pull                      # always start from the team's latest data
# copy your export folder into Testdaten/<Exercise>/
git add Testdaten/ && git commit -m "data: add <exercise> set(s)" && git push
```

### 3. Rebuild the model (one person, or CI)
Whoever has the Kaggle anchor in `data/raw/kaggle_gym_imu/` runs:

```bash
make update          # = uv run python scripts/rebuild_from_testdaten.py
```

This regenerates **everything** from committed data — **both models** so the app can
show the effect of our own data side by side (see DECISIONS.md §9):
- **Model 2 (current — Kaggle + Testdaten):** `models/saved/best_model.joblib`
  (+ `random_forest.joblib`), `novelty_detector.joblib` (refit on base + Testdaten so
  **our own** exercises count as `known`), `model_metrics.json`
- **Model 1 (baseline — Kaggle only):** `models/saved/baseline_model.joblib`,
  `baseline_novelty_detector.joblib`, `baseline_metrics.json`
- `data/processed/feature_names.txt` (shared — the invariant feature set is identical)

Both models are built by the **same** shared pipeline (`src/ml4b/models/pipeline.py`)
with identical 6× augmentation; the only difference is that Model 1 never sees the
Testdaten. The Predict page runs both on each upload and highlights where our own data
changed the prediction.

It also prints a **validation block**: for each `Uncertain/` recording, what fraction
the novelty detector flags as `unknown` (want high); for each `Rest/` recording, what
fraction is gated out as rest (want high). Neither is used for training.

### 4. Commit the rebuilt model
```bash
git add models/saved/ data/processed/feature_names.txt   # both models + metrics
git commit -m "model: retrain on latest Testdaten" && git push
```
Now everyone `git pull`s and runs the **same** model. **Only the rebuild output is
committed by one person** — nobody hand-edits the model, which keeps merges clean
(binary `.joblib` files cannot be merged).

> **Rule of thumb:** contributors commit **data**; the model is a build artifact one
> source regenerates. That single rule is what prevents model divergence.

## Why Rest and Uncertain are validation, not classes
- **Rest** is detected by a device-agnostic **energy gate**, not a learned class.
  A learned rest class transfers badly between people and over-predicts rest on real
  uploads, so we keep the gate. `Rest/` recordings instead **calibrate** the gate:
  the threshold must sit *below* real exercise energy and *above* genuine rest energy,
  so the gate is bounded from both sides. `make calibrate`
  (`scripts/calibrate_gate.py`) measures both distributions, reports the margin to
  each side, and recommends a threshold in the gap — apply one by editing the two
  constants in `src/ml4b/data/activity_gate.py`. Record a few pauses (watch still,
  fidgeting, drinking) into `Testdaten/Rest/` to verify/tune the lower bound.
- **Uncertain** holds *other* exercises (squats, presses, …). Instead of an
  "everything-else" class (which only memorises the few foreign exercises we happened
  to record), the **novelty detector** rejects anything far from the known classes —
  including exercises we never recorded. `Uncertain/` recordings validate that.

## Practical notes for the demo
- **More data per class = better, more representative recognition.** Aim for several
  clean sets per exercise from different team members. With few sets, the invariant
  features still let the Kaggle anchor carry recognition, but your own style is only a
  small fraction of the training data until you record more.
- **Class balance matters.** If one class (e.g. `tricep_extension`) has far fewer
  recordings than the others, record more of it — the per-class F1 in
  `model_metrics.json` will show the weak class.
- **Rollback is one command:** `git checkout HEAD~1 -- models/saved/` restores the
  previous committed model if a rebuild ever makes things worse.
- **Retraining is offline only** — never triggered from the app, so a live demo never
  stalls.

## Optional: GitHub Actions (future)
Because the rebuild is a pure function of committed data, it can later be wrapped in a
GitHub Action that runs `make update` on every push to `Testdaten/**` and commits the
new `models/saved/`. The one prerequisite is making the Kaggle anchor reachable in CI
(it is git-ignored today). Until then, the "one person runs `make update`" workflow
above gives the same single-model guarantee.
