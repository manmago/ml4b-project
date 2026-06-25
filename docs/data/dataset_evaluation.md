# Dataset Evaluation — ML4B Gym Exercise Recognition

> CRISP-DM Phase 2 — Data Understanding · Last updated: 2026-06-17
>
> **Outcome:** the project trains on the **Kaggle "Gym Workout IMU" dataset**
> (recorded on an Apple Watch) **plus our own committed Apple-Watch recordings**
> (`data/Testdaten/`). This document records *how* we got there: which public
> datasets we tried first, why they were dropped, and why the final combination
> works. Column-level detail is in [`data_dictionary.md`](data_dictionary.md);
> the full decision rationale is in [`../DECISIONS.md`](../DECISIONS.md).

---

## 1. What the data has to match

The deployment target is fixed: a **wrist-worn Apple Watch**, streaming
accelerometer + gyroscope, uploaded via Sensor Logger. The single most important
lesson of this project is that a model only transfers if the **training data
comes from the same kind of sensor in the same place on the body**. Everything
below is judged against that: wrist placement, accelerometer **and** gyroscope,
a documented sampling rate, Python-parseable, and our three target movements
(bicep curl, triceps extension, row).

---

## 2. The two public datasets we tried (and dropped)

We started from published, peer-reviewed datasets. Two were serious candidates;
both were abandoned for the **same underlying reason — a device/placement gap to
the Apple Watch** — which is exactly what motivated recording our own data.

### 2.1 RecoFit (Microsoft Research) — *first anchor, abandoned*

| Field | Detail |
|-------|--------|
| Source | Morris et al. (2014), *RecoFit*, CHI / Microsoft Research |
| Placement | **Forearm armband** (not a wrist watch) |
| Sensors | Accelerometer + Gyroscope, 50 Hz |
| Participants | 200+ |

RecoFit was the original training anchor (a 6-class model). It is the best
*public* match on paper — large subject pool, gyroscope, peer-reviewed — but it
is worn on the **forearm**, not the wrist. That placement gap is physical and no
preprocessing can close it: real Apple-Watch curls were systematically
misclassified. **Abandoned.**

### 2.2 MM-Fit — *second anchor, abandoned*

| Field | Detail |
|-------|--------|
| Source | Strömbäck et al. (2020), *MM-Fit*, ACM IMWUT |
| Placement | **Wrist** — but a **Wear-OS TicWatch**, not an Apple Watch |
| Sensors | Accelerometer + Gyroscope (+ other modalities), 100 Hz |
| Participants | 10 |

MM-Fit fixed the *placement* problem (it is wrist-worn) and scored well
in-domain, so we switched to it. But it is recorded on a **non-Apple
smartwatch**: a residual **device-domain gap** (different sensor hardware,
axis orientation and signal conditioning) again hurt real Apple-Watch uploads.
**Superseded** by an Apple-Watch dataset.

> Both rejections point the same way: in-sample accuracy is meaningless if the
> sensor/placement differs from deployment. The fix is to train on data recorded
> *on an Apple Watch, on the wrist* — which is what the final datasets give us.

---

## 3. Final data — Kaggle Apple-Watch anchor + our own Testdaten

### 3.1 Anchor: Kaggle "Gym Workout IMU" dataset

| Field | Detail |
|-------|--------|
| Name | Kaggle *Gym Workout IMU Dataset* (shakthisairam123) |
| Device | **Apple Watch SE**, left wrist — same device family as deployment |
| Sensors | CoreMotion: user acceleration + gravity (g), rotation rate (rad/s) |
| Sampling rate | **100 Hz** |
| Size | 164 single-set CSV files; **75** map to our 3 classes |
| Subjects | **1** (single subject — the central limitation) |
| Source | https://www.kaggle.com/datasets/shakthisairam123/gym-workout-imu-dataset |

Choosing a dataset recorded **on an Apple Watch** removed the device-domain shift
that sank RecoFit and MM-Fit — this is what finally made real uploads work.

**Three target classes**, each grouping biomechanically equivalent Kaggle
exercises along a distinct movement axis:

| Class | Movement axis | Kaggle sets |
|-------|---------------|------------:|
| `bicep_curl` | elbow flexion | 24 |
| `tricep_extension` | overhead elbow extension | 30 |
| `row` | horizontal pull | 21 |

`lateral_raise` / `shoulder_press` were rejected (signal overlap with the curl /
triceps classes); `push_up` is absent from this dataset. `rest`, `uncertain` and
`unknown` are **not** trained classes — they come from inference safeguards
(see `../DECISIONS.md`).

### 3.2 Our own recordings: `data/Testdaten/`

The Kaggle anchor is single-subject, so the robust fix is **more Apple-Watch data
of our own**. We record one clean set per file with Sensor Logger and commit it
under `data/Testdaten/<Exercise>/`; the folder name sets the label. Because the
anchor is *also* Apple Watch / CoreMotion data, these recordings are the **same
device domain** — they extend the anchor rather than reintroducing a foreign
source.

Currently committed (per category):

| Folder | Role | Recordings |
|--------|------|-----------:|
| `Biceps_Curls/` | training — `bicep_curl` | 11 |
| `Rows/` | training — `row` | 17 |
| `Triceps_Extensions/` | training — `tricep_extension` | 7 |
| `Rest/` | calibrates the activity gate (not a class) | 16 |
| `Uncertain/` | validates open-set `unknown` rejection (not a class) | 6 |

`make update` rebuilds the model deterministically from Kaggle + Testdaten. See
[`../project/continual_training.md`](../project/continual_training.md).

---

## 4. Limitation

The anchor is **one subject**, so true leave-one-*subject*-out is impossible. We
evaluate with **leave-one-set-out** cross-validation (each recording held out
once), which measures generalisation to an unseen *set*, not a new *person*.
Real cross-person accuracy will be **below** the reported numbers; augmentation
and device-invariant features (see `../DECISIONS.md`) are the mitigations, and
our own Testdaten is the concrete fix.

| Model | Training data | Leave-one-set-out macro-F1 |
|-------|---------------|---------------------------:|
| Baseline | Kaggle only (75 sets) | **0.776** |
| Current (shipped) | Kaggle + Testdaten (110 sets) | **0.792** |

(The two numbers are not a clean apples-to-apples gain — each is scored over its
own held-out pool; Model 2's pool also contains our harder cross-person
recordings. The like-for-like view is the same-recording comparison in the app.)
