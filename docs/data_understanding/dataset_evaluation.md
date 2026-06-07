# Dataset Evaluation — ML4B Gym Exercise Recognition

> CRISP-DM Phase 2 — Data Understanding | Completed: 2026-05-15

> ✅ **FINAL decision: Kaggle Gym Workout IMU dataset (DECISIONS.md, 2026-05-31).**
> The training anchor changed twice. (1) Phase 2 chose **RecoFit** (forearm-worn,
> 50 Hz). (2) DECISIONS.md switched to **MM-Fit** (wrist, but a non-Apple smartwatch).
> Both scored well in-sample yet failed on real Apple-Watch uploads — a
> **device-domain mismatch**. (3) DECISIONS.md adopts the **Kaggle Gym Workout IMU
> dataset, recorded *on an Apple Watch*** (100 Hz, left wrist), removing that
> mismatch, with **3** classes: bicep_curl, tricep_extension, row. The RecoFit
> and MM-Fit evaluations below are retained for history; the final-state summary
> is in §6. See [`DECISIONS.md`](../DECISIONS.md).

---

## 1. Evaluation Criteria

All candidate datasets were assessed against the following criteria, in order of priority:

| Criterion | Why it matters |
|-----------|---------------|
| Sensor placement | Must be **wrist-worn** to match Apple Watch deployment target |
| Sensor modalities | Must have **Accelerometer + Gyroscope** minimum — gyroscope encodes rotation critical for upper-body exercise discrimination |
| Exercise classes | Must cover as many of our target classes as possible |
| Number of participants | More participants = better inter-subject generalisability of trained models |
| Sampling rate | Must be documentable and consistent for resampling alignment with Apple Watch data |
| Data format | Must be parseable with Python (no proprietary closed tools) |
| Accessibility | Must be freely downloadable without institutional paywall |
| Scientific backing | Peer-reviewed paper preferred — ensures methodology and labels are validated |

---

## 2. Datasets Evaluated

### 2.1 RecoFit (Microsoft Research)

| Field | Detail |
|-------|--------|
| Source | https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors |
| Paper | Morris et al. (2014). *RecoFit: using a wearable sensor to find, recognize, and count repetitive exercises.* CHI Conference, ACM. Microsoft Research. |
| Sensor placement | **Wrist-worn** |
| Sensors | **Accelerometer + Gyroscope** (3 axes each, X/Y/Z) |
| Units | Acceleration in g, angular velocity in dps |
| Sampling rate | **50 Hz** (confirmed from official README) |
| Participants | **200+** |
| Format | MATLAB `.mat` file (~2.5 GB), readable with `scipy.io.loadmat(path, simplify_cells=True)` |
| Exercises | Full list to be confirmed during Phase 2 notebook exploration. Confirmed classes include at minimum: **Bicep Curl** (Two-arm Dumbbell Curl). Paper reports Shoulder Press, Squat, and Bench Press variants — exact label strings will be extracted in `notebooks/02_data_understanding.ipynb`. |
| Original paper accuracy | >95% precision/recall reported by Morris et al. (2014) using a Hidden Markov Model |

**Verdict: ✅ PRIMARY DATASET**

Best overall match across all evaluation criteria: wrist placement confirmed, largest participant pool (200+) of all candidates, 50 Hz sampling rate confirmed from official source, scientifically backed by a peer-reviewed CHI paper from Microsoft Research. Full exercise class list will be extracted in Phase 2 notebook exploration — target classes in `docs/business_understanding/business_understanding.md` Section 3 will be updated accordingly.

---

### 2.2 MM-Fit Dataset

| Field | Detail |
|-------|--------|
| Source | https://mmfit.github.io/ |
| Paper | Strömbäck et al. (2020). *MM-Fit: Multimodal Deep Learning for Automatic Exercise Logging Across Sensing Devices.* ACM IMWUT. |
| Sensor placement | Wrist + Smartphone + Earbuds (multi-modal) |
| Sensors | Accelerometer + Gyroscope + Heart Rate |
| Participants | **10** |

**Verdict: ⚠️ REJECTED AS PRIMARY**

Only 10 participants — insufficient for robust inter-subject generalisation. The complex multi-modal format (three device types simultaneously) adds significant preprocessing overhead not justified for this project scope. Could be revisited as a supplementary validation set in a future extension.

---

### 2.3 IEEE Gym Gesture Classification Dataset

| Field | Detail |
|-------|--------|
| Source | https://ieee-dataport.org/documents/gym-gesture-classification-using-imu-sensor-dataset |
| Sensor placement | Wrist-worn (Arduino Nano 33 BLE) |
| Sensors | Accelerometer + Gyroscope at 100 Hz |
| Participants | **5** |
| Exercises | Chest Press, Chest Fly, Lat Pulldown, Tricep Extension, Seated Row |
| Format | CSV (ax, ay, az, gx, gy, gz) |

**Verdict: ⚠️ REJECTED AS PRIMARY**

Only 5 participants — far too few for reliable training. More critically, the exercise classes (Chest Fly, Lat Pulldown, Tricep Extension, Seated Row) have minimal overlap with our 7 target classes. Kept as a potential supplementary source if we want to extend the class set later.

---

### 2.4 WEAR Dataset

| Field | Detail |
|-------|--------|
| Source | https://mariusbock.github.io/wear/ |
| Paper | Bock et al. (2024). *WEAR: An Outdoor Sports Dataset for Wearable and Egocentric Activity Recognition.* ACM IMWUT. |
| Sensor placement | Wrist + Ankle (two devices) |
| Sensors | **Accelerometer ONLY** — no Gyroscope |
| Participants | 22 |
| Exercises | 18 outdoor activities (running, jumping, etc.) |

**Verdict: ❌ REJECTED**

Missing gyroscope data — a critical modality for distinguishing upper-body gym exercises. Outdoor activities (running, jumping jacks, burpees) do not match our gym exercise target classes. Multi-sensor placement (wrist + ankle simultaneously) does not match our single Apple Watch setup. Not suitable for any role in this project.

---

### 2.5 Kaggle Fitness Tracker Dataset

| Field | Detail |
|-------|--------|
| Source | https://www.kaggle.com/datasets/krishujeniya/fitness-tracker-accelerometer-and-gyroscope-data |
| Paper | None — community-contributed dataset |
| Participants | Unknown |
| Sensor placement | Unclear / undocumented |

**Verdict: ❌ REJECTED**

No peer-reviewed paper or validated methodology. Participant count and sensor placement are undocumented. Insufficient scientific rigour for an academic project. Not reproducible or citable.

---

## 3. Summary Comparison Table

| Dataset | Wrist placement | Acc + Gyro | Gym exercises | Participants | Sampling rate | Python-parseable | Free access | Peer-reviewed |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **RecoFit** (Microsoft) | ✅ | ✅ | ✅ | ✅ 200+ | ✅ 50 Hz | ✅ scipy | ✅ | ✅ CHI 2014 |
| MM-Fit | ✅ | ✅ | ✅ | ⚠️ 10 | ✅ | ✅ | ✅ | ✅ IMWUT 2020 |
| IEEE Gym Gesture | ✅ | ✅ | ⚠️ partial | ❌ 5 | ✅ 100 Hz | ✅ CSV | ✅ | ⚠️ preprint |
| WEAR | ⚠️ wrist+ankle | ❌ no gyro | ❌ outdoor | ⚠️ 22 | ✅ | ✅ | ✅ | ✅ IMWUT 2024 |
| Kaggle Fitness | ❌ unclear | ✅ | ⚠️ unknown | ❌ unknown | ❌ unknown | ✅ CSV | ✅ | ❌ none |

---

## 4. Final Decision (HISTORICAL — Iteration 1)

> Superseded twice; the current decision is in §6. Kept for history.

**Primary dataset: RecoFit (Microsoft Research)**

**Rationale:** RecoFit is the only dataset that satisfies all eight evaluation criteria simultaneously. It is wrist-worn, provides both accelerometer and gyroscope data at a documented 50 Hz, has the largest participant pool (200+), is freely available, and is backed by a peer-reviewed CHI publication — the most rigorous conference in human-computer interaction research.

**Open question:** The full list of exercise classes in RecoFit will be confirmed during Phase 2 notebook exploration (`notebooks/02_data_understanding.ipynb`). Target exercise classes in `docs/business_understanding/business_understanding.md` Section 3 will be updated once the complete class list is known.

**Next steps:**
1. Download RecoFit `.mat` files from https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors
2. Place files in `data/raw/recofit/` (see `data/raw/recofit/README.md` for filenames)
3. Run `notebooks/02_data_understanding.ipynb` to explore class list, signal properties, and data quality

---

## 5. Exercise Class Selection (HISTORICAL — Iteration 1, RecoFit 6-class)

> Superseded by the 3-class Kaggle selection in §6. Kept for history.

The RecoFit dataset contains **75 exercise classes** in total. The following criteria were used to select the final 6 target classes for model training:

**Selection criteria:**
- Wrist-worn exercises only (gym resistance exercises, not machine-based)
- Minimum 30 participants (above the 50% subject threshold visible in the class distribution plot from `notebooks/02_data_understanding.ipynb`)
- Clearly defined wrist movement pattern distinguishable from other classes

**Final mapping table:**

| # | Target Class | RecoFit Source | Participants |
|---|-------------|----------------|--------------|
| 1 | bicep_curl | Two-arm Dumbbell Curl (both arms, not alternating) | ~45 |
| 2 | shoulder_press | Shoulder Press (dumbbell) | ~43 |
| 3 | squat | Squat (arms in front of body) + Squat | ~43+25 combined |
| 4 | tricep_extension | Overhead Triceps Extension | ~42 |
| 5 | lateral_raise | Lateral Raise | ~30 |
| 6 | rest | Non-Exercise + Device on Table | ~90 |

**Classes considered but rejected:**
- **Bench Press** (Chest Press rack, ~20 participants) — insufficient data, below 50% threshold
- **Deadlift** (Dumbbell Deadlift Row, ~20 participants) — insufficient data, below 50% threshold

**Rationale:** This data-driven selection approach is methodologically stronger than an arbitrary upfront selection. By anchoring class selection on actual subject coverage in the dataset, the model is guaranteed sufficient training samples for each class. See `docs/DECISIONS.md` for the full decision rationale.

---

## 6. Final State (Iteration 3 — Kaggle Apple-Watch anchor, DECISIONS.md)

After RecoFit (Iteration 1) and MM-Fit (Iteration 2) both failed to transfer to
real Apple-Watch uploads, the decisive factor was identified as **device-domain
shift**: a model trained on a non-Apple sensor does not generalise to an Apple
Watch. The fix is to train on data recorded *on* an Apple Watch.

### Chosen dataset

| Field | Detail |
|-------|--------|
| Name | Kaggle "Gym Workout IMU Dataset" (shakthisairam123) |
| Device | **Apple Watch SE**, left wrist (same device family as deployment) |
| Sensors | CoreMotion: user acceleration (g) + gravity (g) + rotation rate (rad/s) |
| Sampling rate | **100 Hz** |
| Size | 164 single-set CSV files; **75** map to our 3 classes |
| Subjects | **1** (single subject — the key limitation, see DECISIONS.md) |
| Source | https://www.kaggle.com/datasets/shakthisairam123/gym-workout-imu-dataset |

### Final 3-class selection (DECISIONS.md)

Chosen to maximise per-class coverage **and** biomechanical distinctness across
different movement axes:

| Class | Kaggle abbreviations | Movement axis | Sets |
|-------|----------------------|---------------|------|
| bicep_curl | AIDBC, IDBC, PREC | elbow flexion | 24 |
| tricep_extension | CGOCTE, MTE, SAOCTE, SAODTE | overhead elbow extension | 30 |
| row | CGCR, NGCR, MGTBR | horizontal pull | 21 |

Rejected alternatives: `lateral_raise` (overlaps with the start of a curl) and
`shoulder_press` (confusable with overhead triceps, fewer sets). `push_up` is
**absent** from this dataset (its `APULL` is an assisted *pull*-up), so it was
dropped from the original plan. Full rationale: DECISIONS.md.

### Honest limitation

Single-subject data means we evaluate with **leave-one-set-out** CV (DECISIONS.md),
which measures generalisation to an unseen *set*, not a new *person*.
Augmentation (DECISIONS.md) synthesises the missing subject diversity. Real-world
cross-person accuracy will be below the reported macro F1 of 0.776.
