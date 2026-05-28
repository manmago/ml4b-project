# Data Dictionary — ML4B Gym Exercise Recognition

> Documents every column produced by the data preparation pipeline (`src/ml4b/data/`). Use this as the single source of truth when writing or reviewing modelling code in Phase 4+.

---

## Dataset Overview

| Field | Value |
|-------|-------|
| Primary training source | RecoFit (Microsoft Research), `data/raw/recofit/exercise_data.50.0000_singleonly.mat` |
| Generalization test source | Self-recorded Apple Watch data (Sensor Logger app), `data/raw/personal/` |
| Sensor modalities | Accelerometer (ax, ay, az), Gyroscope (gx, gy, gz) |
| Sampling rate | 50 Hz (confirmed empirically in Phase 2 notebook) |
| Raw file format | MATLAB `.mat` (RecoFit) / CSV (Apple Watch) |
| Target classes | 6 — `bicep_curl`, `shoulder_press`, `squat`, `tricep_extension`, `lateral_raise`, `rest` (see ADR-005) |
| Label type | Multi-class: one exercise label per windowed sample |

---

## Raw Long-Format DataFrame
Produced by `src/ml4b/data/loader.py::load_recofit_raw()`. One row per time sample.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `subject_id` | int | — | RecoFit subject index (0-indexed, 0–93) |
| `exercise_name` | str | — | Mapped target class (one of the 6 above) |
| `recording_id` | int | — | Per-subject/exercise recording index — separates multiple sessions of the same exercise |
| `timestamp` | float | s | Sample time relative to the start of the recording |
| `ax`, `ay`, `az` | float | g | Accelerometer x / y / z (raw RecoFit units) |
| `gx`, `gy`, `gz` | float | dps | Gyroscope x / y / z (degrees per second) |

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

| Label | Source RecoFit classes (merged) |
|-------|---------------------------------|
| `bicep_curl` | "Bicep Curl", "Two-arm Dumbbell Curl (both arms, not alternating)" |
| `shoulder_press` | "Shoulder Press (dumbbell)", "Squat Rack Shoulder Press" |
| `squat` | "Squat", "Squat (arms in front of body, parallel to ground)", "Dumbbell Squat (hands at side)" |
| `tricep_extension` | "Overhead Triceps Extension", "Triceps extension (lying down)" |
| `lateral_raise` | "Lateral Raise" |
| `rest` | "Non-Exercise", "Device on Table", "Rest" |

Selection rationale: ADR-005 (data-driven, >30 participants per class). Mapping is the single dict `EXERCISE_MAPPING` in `src/ml4b/data/loader.py`.

---

## Splits (produced by `subject_based_split()`)
| Split | Default share of subjects | File |
|-------|--------------------------|------|
| train | ~70% | `data/processed/train_features.csv` |
| val   | ~10% | `data/processed/val_features.csv` |
| test  | ~20% | `data/processed/test_features.csv` |

No subject appears in more than one split. See ADR-007.

---

## Data Collection Protocol (Apple Watch — generalization test)

> Used only for the Phase 5 cross-device generalization test.

- **Participants:** 1 (project author, plus optional volunteers)
- **Exercises:** the same 6 target classes
- **Repetitions per exercise per participant:** ≥ 30 s of motion per class
- **Watch placement:** dominant wrist
- **File naming convention:** `<participant_id>_<exercise>_<session>.csv`
- **Quality checks:** confirm 50 Hz sampling and no NaN values before feature extraction
