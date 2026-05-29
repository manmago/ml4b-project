# Data Dictionary — ML4B Gym Exercise Recognition

> Documents every column produced by the data preparation pipeline (`src/ml4b/data/`). Use this as the single source of truth when writing or reviewing modelling code in Phase 4+.

---

## Dataset Overview

| Field | Value |
|-------|-------|
| Primary training source | **MM-Fit** (wrist-worn smartwatch, CC-BY-4.0), `data/raw/mm-fit/` — see ADR-013 |
| Original training source (superseded) | RecoFit (Microsoft Research), forearm-worn — used in Phases 1–5, replaced because the Apple Watch is wrist-worn (ADR-013) |
| Generalization test source | Self-recorded Apple Watch data (Sensor Logger app) |
| Sensor modalities | Accelerometer (ax, ay, az), Gyroscope (gx, gy, gz) |
| Sampling rate | 50 Hz (MM-Fit recorded at 100 Hz, decimated to 50 Hz — ADR-013) |
| Raw file format | NumPy `.npy` (MM-Fit) / CSV (Apple Watch) |
| Target classes | 7 — `bicep_curl`, `shoulder_press`, `squat`, `tricep_extension`, `lateral_raise`, `push_up`, `rest` (ADR-005 + ADR-013) |
| Label type | Multi-class: one exercise label per windowed sample |
| Units (canonical = MM-Fit) | Accel **m/s² including gravity**; gyro **rad/s** (the Apple Watch loader aligns to these — ADR-013) |

---

## Raw Long-Format DataFrame
Produced by `src/ml4b/data/mmfit_loader.py::load_mmfit_workout()` (current) — and
historically by `loader.py::load_recofit_raw()`. **Both emit the identical
schema**, so windowing + features are shared. One row per time sample.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `subject_id` | str/int | — | MM-Fit workout id (e.g. `"w01"`); RecoFit: subject index |
| `exercise_name` | str | — | Mapped target class (one of the 7 above) |
| `recording_id` | str/int | — | Per-wrist contiguous exercise segment — separates sessions so windows never straddle a label boundary |
| `timestamp` | float | s | Sample time relative to the start of the recording |
| `ax`, `ay`, `az` | float | m/s² | Accelerometer x / y / z (MM-Fit units, gravity included) |
| `gx`, `gy`, `gz` | float | rad/s | Gyroscope x / y / z (MM-Fit units) |

---

## Windowed DataFrame
Produced by `src/ml4b/data/windowing.py::apply_sliding_window()`. One row per 100-sample (2 s) window — see ADR-006.

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Carried from raw DataFrame |
| `exercise_name` | str | Carried from raw DataFrame (label) |
| `window_id` | int | Globally unique id across all windows |
| `raw_ax`, `raw_ay`, `raw_az` | list[float] (length 100) | Accelerometer samples in this window |
| `raw_gx`, `raw_gy`, `raw_gz` | list[float] (length 100) | Gyroscope samples in this window |

---

## Engineered Features (47 numeric columns)
Produced by `src/ml4b/data/features.py::extract_features()`. One row per window; identifier columns `subject_id`, `exercise_name`, `window_id` are carried through.

### Per-axis statistical features (7 stats × 6 axes = 42 columns)
For each `axis ∈ {ax, ay, az, gx, gy, gz}`:

| Column | Description | Why it's informative |
|--------|-------------|----------------------|
| `<axis>_mean` | Window mean | Static gravity direction (accel) / steady rotation rate (gyro) |
| `<axis>_std` | Window standard deviation | Variability — separates motion from rest |
| `<axis>_min` | Minimum value in window | Extreme position reached on this axis |
| `<axis>_max` | Maximum value in window | Extreme position reached on this axis |
| `<axis>_range` | `max − min` | Peak-to-peak amplitude of the movement |
| `<axis>_rms` | Root-mean-square | Energy-style summary; does not cancel positive and negative excursions |
| `<axis>_zero_crossing_rate` | Fraction of sign changes between adjacent samples | Proxy for movement frequency without a full FFT |

### Magnitude features (3 columns)
| Column | Formula | Interpretation |
|--------|---------|----------------|
| `accel_magnitude_mean` | mean of `√(ax² + ay² + az²)` | Overall linear motion intensity, orientation-invariant |
| `accel_magnitude_std`  | std  of `√(ax² + ay² + az²)` | Variability of linear motion across the window |
| `gyro_magnitude_mean`  | mean of `√(gx² + gy² + gz²)` | Overall rotational intensity, orientation-invariant |

### Frequency-domain features (2 columns)
Computed via `np.fft.rfft` on the mean-centred accelerometer magnitude (sampled at 50 Hz).

| Column | Description |
|--------|-------------|
| `dominant_frequency` | Frequency (Hz) of the strongest spectral component — typically the repetition rate of the exercise |
| `spectral_energy` | Sum of squared FFT magnitudes — total oscillatory energy of the window |

The ordered list of all 47 column names is also written to `data/processed/feature_names.txt` by the Phase 3 notebook and should be the source consumed by Phase 4 modelling code.

---

## Label Definition

Current mapping is from **MM-Fit** activity strings (single dict `MMFIT_TO_ML4B`
in `src/ml4b/data/mmfit_loader.py`):

| Label | Source MM-Fit activity |
|-------|------------------------|
| `bicep_curl` | `bicep_curls` |
| `shoulder_press` | `dumbbell_shoulder_press` |
| `squat` | `squats` |
| `tricep_extension` | `tricep_extensions` |
| `lateral_raise` | `lateral_shoulder_raises` |
| `push_up` | `pushups` |
| `rest` | `non_activity` (everything outside a labelled set) |

MM-Fit activities with no ML4B equivalent (`lunges`, `situps`, `dumbbell_rows`,
`jumping_jacks`) are dropped. Original RecoFit mapping (`EXERCISE_MAPPING` in
`loader.py`) is retained for the historical pipeline. Selection rationale:
ADR-005 + ADR-013.

---

## Splits (MM-Fit official workout-id partition — ADR-013)
| Split | MM-Fit workout ids | File |
|-------|--------------------|------|
| train | w01,02,03,04,06,07,08,16,17,18 | `data/processed/train_features.csv` |
| val   | w14,15,19 | `data/processed/val_features.csv` |
| test  | w09,10,11 | `data/processed/test_features.csv` |

No workout (session) appears in more than one split — a session/subject-level
split with no leakage, the same principle as ADR-007. The historical RecoFit
pipeline used `subject_based_split()` (~70/10/20 by subject).

---

## Data Collection Protocol (Apple Watch — generalization test)

> Used only for the Phase 5 cross-device generalization test.

- **Participants:** 1 (project author, plus optional volunteers)
- **Exercises:** the same 7 target classes
- **Repetitions per exercise per participant:** ≥ 30 s of motion per class
- **Watch placement:** dominant wrist
- **File naming convention:** `<participant_id>_<exercise>_<session>.csv`
- **Quality checks:** confirm 50 Hz sampling and no NaN values before feature extraction
