# ADR-012: Apple Watch Column Mapping and 100 Hz → 50 Hz Resampling

**Status:** Accepted
**Date:** 2026-05-29
**Deciders:** Anshul Agrawal

> **Numbering note:** the original task referred to this as "ADR-011", but
> `ADR-011-commit-trained-model-to-git.md` already existed, so this decision is
> recorded as **ADR-012** to avoid overwriting it.

---

## Context

Real Apple Watch recordings exported from the Sensor Logger app were predicted
incorrectly. Two concrete bugs in `src/ml4b/data/apple_watch_loader.py` were
identified from three confirmed `WristMotion.csv` samples:

1. **Wrong column mapping.** The confirmed Sensor Logger `WristMotion.csv`
   header is:
   `time, seconds_elapsed, rotationRateX, rotationRateY, rotationRateZ,
   gravityX, gravityY, gravityZ, accelerationX, accelerationY, accelerationZ,
   quaternionW, quaternionX, quaternionY, quaternionZ, pitch, roll, yaw`.
   The previous "Format D" mapped the nanosecond epoch column `time` to the
   timestamp, which also broke sampling-rate detection. The accelerometer and
   gyroscope live in `accelerationX/Y/Z` and `rotationRateX/Y/Z`.

2. **Wrong sampling rate.** Apple Watch via Sensor Logger records at **~100 Hz**;
   the model was trained on **50 Hz** RecoFit data. `window_size=100` at 100 Hz
   is a **1-second** window instead of 2 seconds, so the extracted features no
   longer match the training distribution.

---

## Decision

1. Add a **PRIMARY** column mapping (first in `WRIST_MOTION_COLUMN_MAPPINGS`)
   that uses `seconds_elapsed → timestamp`, `accelerationX/Y/Z → ax/ay/az`, and
   `rotationRateX/Y/Z → gx/gy/gz`. All previous mappings are kept as fallbacks.
2. Add `detect_sampling_rate()` (median timestamp diff) and
   `resample_to_target_hz()` (decimation — keep every Nth sample).
3. `predict_from_sensor_logger()` now detects the rate after loading and
   decimates to 50 Hz when it differs by ≥ 5 Hz, and records `detected_hz` /
   `n_samples_after_resample` in `DataFrame.attrs`.

---

## Alternatives Considered

| Option | Why not chosen |
|--------|----------------|
| **Retrain the model on 100 Hz data** | Requires re-running the whole pipeline and re-validating; the dataset is 50 Hz, so this is a larger, riskier change. |
| **Use `window_size=200` for Apple Watch** | Would give a 2 s window at 100 Hz, but then two code paths (50 vs 100) and feature definitions (e.g. zero-crossing rate, FFT) would diverge from training. |
| **Interpolate/upsample instead of decimate** | Introduces synthetic data points; decimation keeps only real samples and is simpler. |

The column-mapping fix is required regardless of which rate strategy is chosen.

## Rationale

Decimation to 50 Hz is the simplest fix that keeps inference features on the
same time-scale as training, with no retraining. The PRIMARY column mapping is
mandatory because the prior mapping read the wrong columns and a non-seconds
timestamp (which also defeated rate detection).

---

## Consequences

**Positive:**
- Sampling rate is now detected correctly (100 Hz) and decimated to 50 Hz, so a
  100-sample window is ~2 s as in training.
- The correct sensor channels are read; the timestamp is in seconds, so timing
  and rate detection work.

**Negative / Trade-offs:**
- Decimation discards half the samples (acceptable quality loss).

**Known limitation discovered during validation (follow-up required):**
- Even after both fixes, the three samples are **still misclassified**. The
  diagnostic (`scripts/test_apple_watch_prediction.py`) shows the dominant
  cause is a **units/calibration mismatch**: RecoFit's accelerometer includes
  gravity (`accel_magnitude_mean ≈ 1.0 g`), while Sensor Logger's
  `accelerationX/Y/Z` is **userAcceleration with gravity removed**
  (`accel_magnitude_mean ≈ 0.09 g`, ~13 σ from training). The `*_zero_crossing_rate`
  features are also 6–9 σ off for the same reason.
- Experiment: reconstructing total acceleration as
  `acceleration{X,Y,Z} + gravity{X,Y,Z}` restores `accel_magnitude_mean` to
  ~1.0 g and removes the spurious `squat` predictions, but the curl windows are
  then classified mostly as `rest` — indicating a **residual domain shift**
  (sensor placement/orientation, rep cadence) between RecoFit subjects and this
  Apple Watch user.
- **Recommended next steps (separate decision):** (a) restore gravity in the
  loader to match training units, and (b) collect a small set of labeled Apple
  Watch recordings to fine-tune/recalibrate the model. These are deferred here
  because they change the model contract and were out of scope for this fix.
