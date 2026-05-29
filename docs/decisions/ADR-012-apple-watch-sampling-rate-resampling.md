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
4. **Unit alignment to the training distribution (added after validation):**
   - **Gravity reconstruction.** When `gravityX/Y/Z` is present alongside
     `accelerationX/Y/Z`, set `ax = accelerationX + gravityX` (etc.). Sensor
     Logger's accelerationX/Y/Z is *user* acceleration (gravity removed,
     ~0 g at rest); RecoFit's accelerometer includes gravity (~1 g at rest).
   - **Gyroscope rad/s → deg/s.** When the mapping reads `rotationRateX/Y/Z`,
     multiply by `180/π`. Sensor Logger reports rad/s; RecoFit is deg/s
     (training `gyro_magnitude_mean ≈ 53` vs ≈ 1 in rad/s).

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

**Units are now aligned; a residual domain shift remains (follow-up required):**
The diagnostic in `scripts/test_apple_watch_prediction.py` was used to peel back
three successive distribution mismatches, each now fixed in the loader:
1. Sampling rate 100 → 50 Hz (decimation).
2. Accelerometer gravity restored (`accel_magnitude_mean` 0.09 g → ~1.0 g,
   matching training's 1.013 g).
3. Gyroscope rad/s → deg/s (`gyro_magnitude_mean` ~1 → ~50, matching ~53).

After **all three** corrections the two bicep-curl samples are still **not**
recognized (sample 1 → `rest`, sample 2 → `lateral_raise`). The per-magnitude
features now match training, so the remaining error is a genuine **domain
shift**: device, on-wrist orientation / axis conventions, and per-user execution
differ from the RecoFit subjects, so the per-axis features the model relies on
do not line up. (`push_up` additionally is not one of the 6 trained classes and
can never be correct.)

**Recommended next step (separate decision, requires new data):** collect a
small set of labeled Apple Watch recordings per exercise with Sensor Logger and
**fine-tune / retrain** the model on them (optionally combined with RecoFit).
This is the only robust fix for the orientation/execution domain gap; it changes
the model contract and is therefore out of scope for this loader fix.
