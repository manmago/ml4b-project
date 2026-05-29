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

> ℹ️ **Unit alignment (implemented, ADR-013):** the model is trained on **MM-Fit**
> (wrist-worn smartwatch). The loader matches MM-Fit's units automatically — it
> reconstructs total acceleration and converts g → m/s²
> (`ax = (accelerationX + gravityX) × 9.80665`, ~9.81 m/s² at rest) and leaves
> the gyroscope in rad/s (which already matches MM-Fit). The earlier RecoFit
> deg/s conversion (ADR-012) is reverted.
>
> ⚠️ **Known limitation (ADR-013/014):** with MM-Fit, **push-ups are recognized
> correctly** on real Apple Watch recordings. However, **bicep curls are still
> confused with tricep extensions**. Two reasons: (1) a residual orientation gap
> between the Apple Watch and MM-Fit's TicWatch, and (2) bicep curl and tricep
> extension are near-identical at the wrist (both are elbow rotations) — even the
> in-domain test confuses them most. Rotation augmentation was tried and rejected
> (ADR-014). The robust fix for this specific pair is a few labeled Apple Watch
> recordings of *your own* bicep/tricep sessions to fine-tune the model — use the
> Recording Protocol above.

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

**Unit note (ADR-013):** Apple's Core Motion (Wrist Motion) reports *user*
acceleration in **g** (gravity removed) plus a separate gravity vector, and
rotation rate in **rad/s**. The training data (MM-Fit) is in **m/s² including
gravity** and **rad/s**, so the loader reconstructs total acceleration and
converts g → m/s² (`× 9.80665`); the gyroscope is left in rad/s.

---

## Troubleshooting
| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ValueError: Could not detect Sensor Logger column format` | Column name mismatch | Check the CSV headers and add a mapping to `WRIST_MOTION_COLUMN_MAPPINGS` in `apple_watch_loader.py` |
| `FileNotFoundError: WristMotion.csv not found in ZIP` | Wrong ZIP uploaded | Upload the Sensor Logger export ZIP that contains `WristMotion.csv` |
| `ValueError: Recording too short` | Fewer than 100 samples (2 s) | Record at least ~15–30 seconds |
| All predictions are `rest` | Recording too short / mostly idle | Perform clear, repeated reps; ensure 50 Hz sampling |
| Low macro F1 (< 50%) | Wrong sensor rate | Confirm Sensor Logger is set to 50 Hz |
