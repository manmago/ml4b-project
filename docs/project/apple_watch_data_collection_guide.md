# Apple Watch Data Collection Guide

## Goal
Record an Apple Watch gym session with **Sensor Logger** and upload it to the
app to recognize **bicep curl, tricep extension, and row** per 2-second window.
Pauses are detected automatically as **rest** (energy gate, ADR-017), so you do
**not** need to record rest separately.

> The model is trained on the Kaggle Gym Workout IMU dataset (Apple Watch, 100 Hz
> — ADR-016). Only the single **`WristMotion.csv`** file is needed.

---

## App Setup
- **App:** **Sensor Logger** — free on the iOS App Store:
  https://apps.apple.com/app/sensor-logger/id1531582925
  (install it on both the iPhone **and** the Apple Watch)
- **Device:** Apple Watch (Series 6 or later recommended)
- **Channel to enable:** **Wrist Motion** (Device Motion) — provides accelerometer
  + gyroscope. All other channels are optional/ignored.
- **Sampling rate:** the Apple Watch records Wrist Motion at **~100 Hz**, which
  matches training. The app also resamples any rate to 100 Hz automatically, so
  you don't need to set this precisely.

---

## Recording Protocol
For each of the three exercises:
1. Start a new recording in Sensor Logger.
2. Perform **3 sets × 10 repetitions** with normal rest between sets — keep the
   recording running through the pauses (they are detected as `rest`).
3. Stop the recording after the last set.
4. Export it (see below) and upload the `WristMotion.csv` to the app.

| Exercise | Sets | Reps | Notes |
|----------|------|------|-------|
| bicep_curl | 3 | 10 | Dumbbell curl — elbow flexion |
| tricep_extension | 3 | 10 | Overhead dumbbell/cable extension — one or both arms |
| row | 3 | 10 | Cable / dumbbell row — horizontal pull |

(You can also record a single exercise per file to check one class at a time.)

---

## How to Export from Sensor Logger
1. Open Sensor Logger → tap the recording you want.
2. Tap **Share / Export** → **Save to Files** (or AirDrop / email to yourself).
3. Choose **CSV** (a folder/ZIP of CSVs) or the combined **ZIP** export.
4. Transfer the export to the computer running the app.

A Sensor Logger export contains several files. Only one matters:

| File | Use |
|------|-----|
| **`WristMotion.csv`** | **THE ONLY FILE NEEDED** — accelerometer + gyroscope from the wrist |
| `HeartRate.csv`, `Tags.csv`, `Metadata.csv`, `Watch*.csv` … | ignored by the app |

---

## What to Upload to the App
On the **🔮 Predict Exercise** page upload **either**:

- the single **`WristMotion.csv`** file, **or**
- the **full ZIP** of the export — the app finds `WristMotion.csv` inside it
  automatically (even in a subfolder).

---

## Confirmed WristMotion.csv Columns (from a real Apple Watch export)

```
time, seconds_elapsed, rotationRateX, rotationRateY, rotationRateZ,
gravityX, gravityY, gravityZ, accelerationX, accelerationY, accelerationZ,
quaternionW, quaternionX, quaternionY, quaternionZ, pitch, roll, yaw
```

- **Used by the model:** `seconds_elapsed` (timestamp), `accelerationX/Y/Z`
  (user acceleration in g), `gravityX/Y/Z` (g), `rotationRateX/Y/Z` (gyro, rad/s).
- **Canonicalization (matches training exactly — ADR-016):** acceleration is
  reconstructed as **total acceleration in g** (`ax = accelerationX + gravityX`,
  ≈ 1 g at rest) and the gyroscope is kept in **rad/s**. Both the Kaggle training
  data and Sensor Logger are Apple CoreMotion, so **no unit conversion** is needed.
- **Sampling rate:** ~100 Hz; the pipeline resamples to exactly 100 Hz
  (`canonical.resample_uniform`).
- **Discarded:** `quaternion*`, `pitch`, `roll`, `yaw`, `time`.

---

## How the App Interprets a Recording
1. **Resample** to 100 Hz, **window** into 2 s segments (200 samples, 50% overlap).
2. **Activity gate** (ADR-017): low-motion windows → `rest` (this is how pauses
   between sets are handled — you do not record rest separately).
3. **Invariant features** (ADR-018) → **Random Forest** predicts one of the three
   exercises.
4. **Confidence threshold** (ADR-020): if the top probability < 0.50 the window is
   reported as `uncertain`.

---

## Supported Column Formats
`src/ml4b/data/apple_watch_loader.py` (`detect_and_normalize_columns()`)
auto-detects the format and normalizes to `[timestamp, ax, ay, az, gx, gy, gz]`
(total accel in g, gyro rad/s). One of these column sets must be present
(case-insensitive):

| Format | Columns (source) |
|--------|------------------|
| PRIMARY — real WristMotion.csv | `seconds_elapsed, accelerationX/Y/Z, rotationRateX/Y/Z` (+ `gravityX/Y/Z`) |
| A — default WristMotion.csv | `time, seconds_elapsed, x, y, z, roll, pitch, yaw` |
| B — pre-normalized | `timestamp, ax, ay, az, gx, gy, gz` |
| C — seconds_elapsed variant | `seconds_elapsed, x, y, z, roll, pitch, yaw` |
| D — DeviceMotion export | `time, accelerationX/Y/Z, rotationRateX/Y/Z` |

If detection fails, the loader raises a `ValueError` listing the columns it
found, so a new mapping can be added to `WRIST_MOTION_COLUMN_MAPPINGS`.

---

## Troubleshooting
| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ValueError: Could not detect Sensor Logger column format` | Column-name mismatch | Check the CSV headers; add a mapping to `WRIST_MOTION_COLUMN_MAPPINGS` in `apple_watch_loader.py` |
| `FileNotFoundError: WristMotion.csv not found in ZIP` | Wrong ZIP uploaded | Upload the Sensor Logger export ZIP that contains `WristMotion.csv` |
| `ValueError: Recording too short` | Fewer than 200 samples (~2 s at 100 Hz) | Record at least ~15–30 seconds |
| Many windows are `rest` | Long pauses / watch held still | Expected — the energy gate marks low-motion windows as rest |
| Many windows are `uncertain` | Out-of-scope motion or a new user's style | Expected for an exercise outside the 3 classes; see the single-subject limitation (ADR-021) |
| Curls predicted as tricep extensions | Both are elbow movements that look alike at the wrist | Known confusable pair — the most-confused cell in the confusion matrix |
