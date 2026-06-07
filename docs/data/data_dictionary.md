# Data Dictionary — ML4B Gym Exercise Recognition

> Documents every column produced by the **current** data pipeline
> (`src/ml4b/data/`). Single source of truth for the 3-class Apple-Watch model
> (DECISIONS.md). The legacy MM-Fit/RecoFit per-axis pipeline is noted at the end.

---

## Dataset Overview

| Field | Value |
|-------|-------|
| Training source | **Kaggle Gym Workout IMU dataset** — Apple Watch SE, left wrist, `data/raw/kaggle_gym_imu/` (DECISIONS.md) |
| Abandoned sources | MM-Fit (non-Apple smartwatch) and RecoFit (forearm) — device-domain mismatch (DECISIONS.md) |
| Inference source | Apple Watch via **Sensor Logger** (`WristMotion.csv` / ZIP) |
| Sensor modalities | Accelerometer (ax, ay, az), Gyroscope (gx, gy, gz) |
| Sampling rate | **100 Hz** (native; the app resamples any rate to 100 Hz) |
| Raw file format | CSV (both Kaggle and Sensor Logger — Apple CoreMotion) |
| Target classes | **3** — `bicep_curl`, `tricep_extension`, `row` (DECISIONS.md) |
| Non-model outputs | `rest` (energy gate, DECISIONS.md), `unknown` (novelty detector, DECISIONS.md), `uncertain` (confidence threshold, DECISIONS.md) |
| Units (canonical) | Accel **total acceleration in g** (userAccel + gravity); gyro **rad/s** |

Canonicalization is defined once in `src/ml4b/data/canonical.py` and shared by
the training loader (`kaggle_loader.py`) and the inference loader
(`apple_watch_loader.py`).

---

## Raw Long-Format DataFrame
Produced by `src/ml4b/data/kaggle_loader.py::load_kaggle_3class()` (training) and
`apple_watch_loader.py` (inference). One row per time sample.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `subject_id` | int | — | Always 0 (single-subject Kaggle anchor) |
| `exercise_name` | str | — | Target class: `bicep_curl`, `tricep_extension`, or `row` |
| `recording_id` | str | — | Per-set id (Kaggle filename stem) — windows never cross a set; used for leave-one-set-out CV (DECISIONS.md) |
| `timestamp` | float | s | Sample time relative to recording start |
| `ax`, `ay`, `az` | float | g | Total acceleration (userAccel + gravity) x / y / z |
| `gx`, `gy`, `gz` | float | rad/s | Gyroscope (rotation rate) x / y / z |

---

## Windowed DataFrame
Produced by `src/ml4b/data/windowing.py::apply_sliding_window()`. One row per
**200-sample (2 s @ 100 Hz)** window, 50% overlap (DECISIONS.md).

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Carried from raw DataFrame |
| `exercise_name` | str | Carried from raw DataFrame (label) |
| `recording_id` | str | Carried through — the leave-one-set-out group key |
| `window_id` | int | Globally unique id across all windows |
| `raw_ax`, `raw_ay`, `raw_az` | list[float] (length 200) | Accelerometer samples (g) |
| `raw_gx`, `raw_gy`, `raw_gz` | list[float] (length 200) | Gyroscope samples (rad/s) |

---

## Engineered Features (39 device-invariant columns)
Produced by `src/ml4b/data/features_invariant.py::extract_invariant_features()`
(DECISIONS.md). One row per window; identifier columns are carried through. The
ordered list is written to `data/processed/feature_names.txt` (committed).

### Magnitude features (rotation-invariant) — 20 columns
For each magnitude signal `m ∈ {accel_mag = √(ax²+ay²+az²), gyro_mag = √(gx²+gy²+gz²)}`:

| Column | Description |
|--------|-------------|
| `<m>_mean`, `<m>_std`, `<m>_min`, `<m>_max`, `<m>_range`, `<m>_rms`, `<m>_mad` | Amplitude statistics of the magnitude signal |
| `<m>_zcr` | Zero-crossing rate of the mean-centred magnitude (cadence proxy) |
| `<m>_dom_freq` | Dominant FFT frequency (Hz) — repetition cadence |
| `<m>_spec_energy` | Total spectral energy of the magnitude |

### Per-window z-normalized shape features — 12 columns
Each axis is standardized within the window (offset/scale removed), then:

| Column | Description |
|--------|-------------|
| `<axis>_zcr` | Zero-crossing rate of the z-normalized axis (`axis ∈ {ax,ay,az,gx,gy,gz}`) |
| `<axis>_dom_freq` | Dominant FFT frequency of the z-normalized axis |

### Axis-pair correlations — 6 columns
Scale/offset-invariant coordination structure:
`corr_ax_ay`, `corr_ax_az`, `corr_ay_az`, `corr_gx_gy`, `corr_gx_gz`, `corr_gy_gz`.

### Cross-sensor ratio — 1 column
| Column | Description |
|--------|-------------|
| `gyro_accel_ratio` | mean `gyro_mag` / mean `accel_mag` — how rotational vs translational the movement is |

**Why invariant?** Magnitudes are unchanged by device rotation; per-window
z-normalization removes per-device offset/gain; correlations are offset/scale
invariant. This is what lets a single-subject model transfer across watches and
users (DECISIONS.md), complemented by augmentation (DECISIONS.md).

---

## Label Definition (DECISIONS.md)
Mapping from Kaggle exercise abbreviations (`ABBREV_TO_CLASS` in
`kaggle_loader.py`):

| Label | Kaggle abbreviations |
|-------|----------------------|
| `bicep_curl` | AIDBC, IDBC, PREC |
| `tricep_extension` | CGOCTE, MTE, SAOCTE, SAODTE |
| `row` | CGCR, NGCR, MGTBR |

All other Kaggle exercises are ignored. `rest`, `unknown` and `uncertain` are
produced at inference, not trained.

---

## Evaluation Split (DECISIONS.md)
**Leave-one-set-out** cross-validation grouped by `recording_id` (one Kaggle file
= one set). Each set is held out once; its augmented copies are excluded from
training. True leave-one-*subject*-out is impossible (single subject). The
shipped model is trained on all sets + augmentation; the honest metric is the
cross-validation aggregate (macro F1 0.776), stored in
`models/saved/model_metrics.json`.

---

## Legacy pipeline (abandoned, kept for history)
`src/ml4b/data/features.py` produced **47 per-axis features** for the MM-Fit/
RecoFit pipeline (accel m/s² including gravity, gyro rad/s, 100-sample windows at
50 Hz). It is superseded by the invariant features above (DECISIONS.md) and is no
longer used by the app or the final model.
