# Apple Watch Data Collection Guide

## Goal
Record personal gym session data to test model generalization on real Apple Watch data.  
Target: ≥ 65% macro F1 on self-recorded data (see `docs/business_understanding/business_understanding.md` Section 5).

This guide tells you exactly what to record and how to export it so Cell 8 in
`notebooks/05_evaluation.ipynb` can process the files automatically.

---

## App Setup
- **App:** **Sensor Logger** — free on the iOS App Store:
  https://apps.apple.com/app/sensor-logger/id1531582925
  (install it on both the iPhone **and** the Apple Watch)
- **Device:** Apple Watch (Series 6 or later recommended)
- **Channels to enable:** Wrist Motion (Accelerometer + Gyroscope) — all others optional
- **Sampling rate:** Set to **50 Hz** to match the RecoFit training data

---

## Recording Protocol
For each of the 6 exercises:
1. Start a new recording in Sensor Logger
2. Perform **3 sets × 10 repetitions**
3. Rest **30 seconds** between sets (keep recording — rest windows are important for the model)
4. Stop recording after the 3rd set
5. Name the file: `<exercise_name>_session<N>.csv` (e.g. `bicep_curl_session1.csv`)

---

## Exercises to Record
| Exercise | Sets | Reps | Notes |
|----------|------|------|-------|
| bicep_curl | 3 | 10 | Dumbbell, both arms alternating or simultaneous |
| shoulder_press | 3 | 10 | Dumbbell, standing |
| squat | 3 | 10 | Bodyweight or barbell |
| tricep_extension | 3 | 10 | Overhead dumbbell, one or both arms |
| lateral_raise | 3 | 10 | Dumbbell, standing |
| rest | — | — | 2 minutes of standing/walking between exercises (record separately) |

---

## How to Export from Sensor Logger
1. Open Sensor Logger → tap the recording you want.
2. Tap **Share / Export** → **Save to Files** (or AirDrop / email to yourself).
3. Choose **CSV** (a folder/ZIP of CSVs) or the combined **ZIP** export.
4. Transfer the export to the computer running the app.

A Sensor Logger export folder/ZIP contains several files. The one that matters is:

| File | Use |
|------|-----|
| **`WristMotion.csv`** | **PRIMARY** — accelerometer + gyroscope from the wrist |
| `HeartRate.csv` | optional |
| `Tags.csv` | optional — user-defined tags with timestamps, useful for labeling |
| `Metadata.csv`, `Manifest.csv` | metadata |
| `Watch*.csv` (Barometer, Compass, Location, Magnetometer, …) | ignored |

---

## What File to Upload to the App
On the app's **🔮 Predict Exercise** page you can upload **either**:

- the single **`WristMotion.csv`** file, **or**
- the **full ZIP** of the Sensor Logger export — the app automatically finds
  `WristMotion.csv` inside it (even in a subfolder).

### File placement for notebook-based evaluation
For the generalization test in `notebooks/05_evaluation.ipynb`, place files in
`data/raw/apple_watch/` using this naming convention:

```
data/raw/apple_watch/
├── bicep_curl_session1.csv
├── shoulder_press_session1.csv
├── squat_session1.csv
├── tricep_extension_session1.csv
├── lateral_raise_session1.csv
└── rest_session1.csv
```

---

## Confirmed WristMotion.csv Columns (from a real Apple Watch export)

```
time, seconds_elapsed, rotationRateX, rotationRateY, rotationRateZ,
gravityX, gravityY, gravityZ, accelerationX, accelerationY, accelerationZ,
quaternionW, quaternionX, quaternionY, quaternionZ, pitch, roll, yaw
```

- **Used by the model:** `seconds_elapsed` (timestamp), `accelerationX/Y/Z`
  (linear acceleration → `ax/ay/az`), `rotationRateX/Y/Z` (gyroscope → `gx/gy/gz`).
- **Discarded:** `gravityX/Y/Z`, `quaternionW/X/Y/Z`, `pitch`, `roll`, `yaw`, `time`.
- **Sampling rate:** Apple Watch records at **~100 Hz**; the pipeline auto-detects
  this and decimates to **50 Hz** to match training — see ADR-012.

> ⚠️ **Known calibration caveat (ADR-012):** RecoFit's accelerometer includes
> gravity (rest ≈ 1 g), whereas Sensor Logger's `accelerationX/Y/Z` has gravity
> removed (rest ≈ 0). This distribution gap currently degrades real-watch
> predictions; reconstructing `acceleration + gravity` and/or fine-tuning on
> Apple Watch recordings are recommended follow-ups.

---

## Column Format Expected
The loader (`src/ml4b/data/apple_watch_loader.py`,
`detect_and_normalize_columns()`) auto-detects the format and normalizes every
input to the internal schema `[timestamp, ax, ay, az, gx, gy, gz]`. One of the
following column sets must be present (matched case-insensitively):

| Format | Columns (source) |
|--------|------------------|
| A — Sensor Logger default WristMotion.csv | `time, seconds_elapsed, x, y, z, roll, pitch, yaw` |
| B — pre-normalized | `timestamp, ax, ay, az, gx, gy, gz` |
| C — seconds_elapsed variant | `seconds_elapsed, x, y, z, roll, pitch, yaw` |
| D — DeviceMotion export | `time, accelerationX/Y/Z, rotationRateX/Y/Z` |

If detection fails, the loader raises a `ValueError` listing the columns it
found, so you can identify the mismatch and add a new mapping to
`WRIST_MOTION_COLUMN_MAPPINGS`.

**Unit note:** Apple's Core Motion (Wrist Motion) reports user acceleration
already in **g** and rotation rate in **rad/s**, matching the RecoFit training
units — so the loader does **not** apply any unit conversion.

---

## Troubleshooting
| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ValueError: Could not detect Sensor Logger column format` | Column name mismatch | Check the CSV headers and add a mapping to `WRIST_MOTION_COLUMN_MAPPINGS` in `apple_watch_loader.py` |
| `FileNotFoundError: WristMotion.csv not found in ZIP` | Wrong ZIP uploaded | Upload the Sensor Logger export ZIP that contains `WristMotion.csv` |
| `ValueError: Recording too short` | Fewer than 100 samples (2 s) | Record at least ~15–30 seconds |
| All predictions are `rest` | Recording too short / mostly idle | Perform clear, repeated reps; ensure 50 Hz sampling |
| Low macro F1 (< 50%) | Wrong sensor rate | Confirm Sensor Logger is set to 50 Hz |
