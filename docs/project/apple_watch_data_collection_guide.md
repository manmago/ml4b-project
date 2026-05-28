# Apple Watch Data Collection Guide

## Goal
Record personal gym session data to test model generalization on real Apple Watch data.  
Target: ≥ 65% macro F1 on self-recorded data (see `docs/business_understanding/business_understanding.md` Section 5).

This guide tells you exactly what to record and how to export it so Cell 8 in
`notebooks/05_evaluation.ipynb` can process the files automatically.

---

## App Setup
- **App:** Sensor Logger (free, iOS App Store)
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

## Export & File Placement
1. Export CSV from Sensor Logger after each recording session
2. Place files in `data/raw/apple_watch/`
3. Run `notebooks/05_evaluation.ipynb` Cell 8 to evaluate generalization

File naming convention:
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

## Expected Output Columns from Sensor Logger
The loader (`src/ml4b/data/apple_watch_loader.py`) handles the most common export
formats automatically. One of the following column sets must be present:

| Format | Columns |
|--------|---------|
| Combined motion file | `time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z` |
| Per-sensor file | `time, x, y, z` (accelerometer only — gyroscope in a separate file) |
| Standard names | `timestamp, ax, ay, az, gx, gy, gz` |

If the column detection fails, the loader raises a `ValueError` with the available
column names so you can identify the mismatch.

**Unit note:** Sensor Logger exports accelerometer in m/s². The loader converts
to g (÷ 9.80665) automatically to match the training data units.

---

## Troubleshooting
| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ValueError: Could not find columns` | Column name mismatch | Check actual CSV headers, update the loader's `_detect_column_map()` |
| Low macro F1 (< 50%) | Recording at wrong sensor rate | Confirm Sensor Logger is set to 50 Hz |
| All predictions are `rest` | Recording is too short | Ensure each CSV has at least 100 rows (2 s) per window |
| High `rest` predictions during exercise | Exercise segments too short | Each set should have at least 10 reps (~5–10 s) |
