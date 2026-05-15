# Dataset Evaluation — ML4B Gym Exercise Recognition

> CRISP-DM Phase 2 — Data Understanding | Completed: 2026-05-15

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

## 4. Final Decision

**Primary dataset: RecoFit (Microsoft Research)**

**Rationale:** RecoFit is the only dataset that satisfies all eight evaluation criteria simultaneously. It is wrist-worn, provides both accelerometer and gyroscope data at a documented 50 Hz, has the largest participant pool (200+), is freely available, and is backed by a peer-reviewed CHI publication — the most rigorous conference in human-computer interaction research.

**Open question:** The full list of exercise classes in RecoFit will be confirmed during Phase 2 notebook exploration (`notebooks/02_data_understanding.ipynb`). Target exercise classes in `docs/business_understanding/business_understanding.md` Section 3 will be updated once the complete class list is known.

**Next steps:**
1. Download RecoFit `.mat` files from https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors
2. Place files in `data/raw/recofit/` (see `data/raw/recofit/README.md` for filenames)
3. Run `notebooks/02_data_understanding.ipynb` to explore class list, signal properties, and data quality
