# Data Dictionary — ML4B Gym Exercise Recognition

> **Status:** Placeholder — to be completed once sensor data is available.

---

## Dataset Overview

| Field | Value |
|-------|-------|
| Source | Apple Watch (Series 6 or later) |
| Sensor modalities | Accelerometer (ax, ay, az), Gyroscope (gx, gy, gz) |
| Sampling rate | TBD (typically 50 Hz or 100 Hz) |
| File format | CSV |
| Location | `data/raw/` (not in git) |
| Label type | Multi-class: one exercise label per recording window |

---

## Raw Features

> To be filled in after data collection. Document each sensor column below.

| Column | Unit | Description |
|--------|------|-------------|
| `timestamp` | ms | Unix timestamp in milliseconds |
| `ax` | m/s² | Accelerometer — x axis |
| `ay` | m/s² | Accelerometer — y axis |
| `az` | m/s² | Accelerometer — z axis |
| `gx` | °/s | Gyroscope — x axis (roll rate) |
| `gy` | °/s | Gyroscope — y axis (pitch rate) |
| `gz` | °/s | Gyroscope — z axis (yaw rate) |
| `label` | string | Exercise label (e.g. `bicep_curl`, `squat`) |

---

## Engineered Features

> To be completed during CRISP-DM Phase 3 (Data Preparation).  
> Features will be computed over a sliding window of W samples.

Planned feature categories:

| Category | Examples |
|----------|---------|
| Statistical | mean, std, min, max, median per axis |
| Frequency domain | dominant frequency, spectral energy (via FFT) |
| Magnitude | `sqrt(ax² + ay² + az²)` — sensor-magnitude vector |
| Correlation | cross-axis correlations |
| Peak features | peak count, peak height within window |

---

## Label Definition

| Label | Description |
|-------|-------------|
| TBD | Exercise labels to be defined during data collection |

**Labelling protocol:** Labels are assigned per recording session. Each session contains one exercise type performed for multiple repetitions.

---

## Data Collection Protocol

> To be completed once the collection protocol is finalised.

- **Participants:** TBD
- **Exercises:** TBD
- **Repetitions per exercise per participant:** TBD
- **Watch placement:** TBD (e.g. dominant wrist)
- **File naming convention:** `<participant_id>_<exercise>_<session>.csv`
- **Quality checks:** TBD (missing values, sampling rate validation)
