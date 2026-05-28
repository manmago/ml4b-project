"""Feature extraction for the ML4B exercise recognition pipeline.

This module is the third stage of the data preparation pipeline. It consumes
the windowed DataFrame produced by
:func:`ml4b.data.windowing.apply_sliding_window` and computes a fixed-size
feature vector per window, combining:

* Per-axis statistics (mean, std, min, max, range, RMS, zero-crossing rate)
  for all six channels (ax, ay, az, gx, gy, gz).
* Combined accelerometer and gyroscope magnitudes — orientation-invariant
  summaries of overall motion intensity.
* Two frequency-domain features computed via FFT on the accelerometer
  magnitude — dominant frequency and total spectral energy.

The resulting feature matrix is the direct input to the classical ML models
trained in Phase 4 (Random Forest, SVM, etc.).
"""

import numpy as np
import pandas as pd

# The six raw signal channels carried through from windowing.py. Listed in a
# constant so the feature loop is data-driven and adding a new axis later
# (e.g. magnetometer) needs only one edit.
_AXES: tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")

# Default sampling rate of the RecoFit dataset, used for FFT frequency
# binning. Kept as a module constant rather than a function argument because
# the rest of the pipeline assumes 50 Hz end-to-end.
_SAMPLING_RATE_HZ: int = 50


def _zero_crossing_rate(signal: np.ndarray) -> float:
    """Compute the per-sample zero-crossing rate of a signal.

    A zero crossing happens whenever consecutive samples have opposite signs.
    Higher rates indicate higher-frequency oscillation, which is a strong
    discriminator between fast exercises (lateral raise) and slow ones
    (squat).

    Args:
        signal: 1-D numpy array of sensor values.

    Returns:
        Fraction of adjacent sample pairs that change sign.
    """
    if signal.size < 2:
        return 0.0
    # np.sign returns -1, 0, or 1. A change between any two consecutive
    # signs (excluding the 0→±1 case) marks a zero crossing.
    signs = np.sign(signal)
    return float(np.mean(signs[:-1] != signs[1:]))


def _axis_features(signal: np.ndarray, axis_name: str) -> dict[str, float]:
    """Compute the seven per-axis statistical features for one channel.

    Args:
        signal: 1-D numpy array containing one window of one axis.
        axis_name: Prefix used in the returned dict keys (e.g. "ax").

    Returns:
        Dict mapping ``f"{axis_name}_{stat}"`` to its scalar value.
    """
    # Pre-compute aggregations once so we touch the array a minimum number of
    # times — feature extraction runs on tens of thousands of windows.
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))

    return {
        # mean — average signal level; captures static gravity direction (accel)
        # or steady-state rotation rate (gyro) characteristic of the pose.
        f"{axis_name}_mean": mean_val,
        # std — variability; higher std = larger-amplitude movement.
        f"{axis_name}_std": std_val,
        # min / max — extreme positions reached during the window;
        # exercises with large range of motion (squat) differ from
        # small-range ones (rest).
        f"{axis_name}_min": min_val,
        f"{axis_name}_max": max_val,
        # range — peak-to-peak amplitude; captures the swing of one rep.
        f"{axis_name}_range": max_val - min_val,
        # RMS — root mean square; an energy-style summary that, unlike mean,
        # does not cancel out positive and negative excursions.
        f"{axis_name}_rms": float(np.sqrt(np.mean(signal**2))),
        # zero-crossing rate — proxy for movement frequency without needing
        # a full FFT per axis; differentiates jerky vs smooth motion.
        f"{axis_name}_zero_crossing_rate": _zero_crossing_rate(signal),
    }


def _magnitude_features(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
) -> dict[str, float]:
    """Compute orientation-invariant magnitude features.

    Magnitude features collapse the three spatial axes into one scalar per
    sample, so they remain meaningful even if the device is worn at a
    slightly different orientation between participants — an important
    robustness property for cross-subject generalization.

    Args:
        ax, ay, az: Accelerometer x, y, z channels (1-D numpy arrays).
        gx, gy, gz: Gyroscope x, y, z channels (1-D numpy arrays).

    Returns:
        Dict with three magnitude-based summary features.
    """
    # Euclidean magnitude of the acceleration vector — overall "how much
    # linear motion is happening" regardless of wrist orientation.
    accel_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    # Same idea for gyroscope — overall rotation intensity, useful for
    # discriminating rotational exercises (bicep curl) from translational
    # ones (squat) and from rest.
    gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)

    return {
        # Mean intensity of linear motion across the window.
        "accel_magnitude_mean": float(np.mean(accel_magnitude)),
        # Variability of linear motion — distinguishes steady poses from
        # repetitive movement of similar average intensity.
        "accel_magnitude_std": float(np.std(accel_magnitude)),
        # Mean intensity of rotational motion across the window.
        "gyro_magnitude_mean": float(np.mean(gyro_magnitude)),
    }


def _frequency_features(accel_magnitude: np.ndarray) -> dict[str, float]:
    """Compute frequency-domain features from the accelerometer magnitude.

    Operating on the magnitude (instead of one axis) yields a single,
    orientation-invariant spectrum that captures the repetition cadence
    of an exercise — bicep curls oscillate at ~0.5–1 Hz, jumping jacks
    much faster. This is the single most informative signal for
    distinguishing periodic exercises from non-periodic rest.

    Args:
        accel_magnitude: 1-D numpy array, sqrt(ax² + ay² + az²) for one window.

    Returns:
        Dict with the dominant frequency (Hz) and total spectral energy.
    """
    n = accel_magnitude.size
    if n < 2:
        return {"dominant_frequency": 0.0, "spectral_energy": 0.0}

    # Subtract the mean so the DC component (gravity offset) does not
    # dominate the spectrum — we care about the AC oscillation pattern.
    centered = accel_magnitude - np.mean(accel_magnitude)

    # rfft returns only the non-negative frequencies, which is all we need
    # for a real-valued signal; halves the compute compared with fft.
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(n, d=1.0 / _SAMPLING_RATE_HZ)

    # The bin with maximum magnitude in the spectrum is the dominant motion
    # frequency — typically the repetition rate of the exercise.
    dominant_idx = int(np.argmax(spectrum))

    return {
        # Frequency (Hz) of the strongest oscillation component.
        "dominant_frequency": float(freqs[dominant_idx]),
        # Sum of squared magnitudes ≈ Parseval energy; a rest window has
        # almost no spectral energy, a vigorous exercise has lots.
        "spectral_energy": float(np.sum(spectrum**2)),
    }


def _row_features(row: pd.Series) -> dict[str, float | str | int]:
    """Compute the full feature vector for one window row.

    Args:
        row: One row from the windowed DataFrame containing raw_* lists,
            subject_id, exercise_name, and window_id.

    Returns:
        Dict mapping every feature name (and the carried-over identifiers)
        to its value. Keys are deterministic and consistent across rows.
    """
    # Convert each raw list to a numpy array exactly once so all downstream
    # math is vectorized.
    arrays = {axis: np.asarray(row[f"raw_{axis}"], dtype=float) for axis in _AXES}

    feats: dict[str, float | str | int] = {}

    # Per-axis statistical features for all six channels.
    for axis in _AXES:
        feats.update(_axis_features(arrays[axis], axis))

    # Orientation-invariant magnitude summaries.
    feats.update(
        _magnitude_features(
            arrays["ax"],
            arrays["ay"],
            arrays["az"],
            arrays["gx"],
            arrays["gy"],
            arrays["gz"],
        )
    )

    # Frequency-domain features computed on the accel magnitude — recomputing
    # the magnitude here keeps _frequency_features decoupled from the dict
    # returned by _magnitude_features.
    accel_magnitude = np.sqrt(arrays["ax"] ** 2 + arrays["ay"] ** 2 + arrays["az"] ** 2)
    feats.update(_frequency_features(accel_magnitude))

    # Carry through the identifying columns so the resulting frame is
    # immediately usable by the splitter and by the model trainer.
    feats["subject_id"] = row["subject_id"]
    feats["exercise_name"] = row["exercise_name"]
    feats["window_id"] = row["window_id"]
    return feats


def extract_features(window_df: pd.DataFrame) -> pd.DataFrame:
    """Extract statistical and frequency-domain features from sensor windows.

    For each window, computes a fixed-size feature vector from the raw
    accelerometer and gyroscope signals. These features summarize the
    motion pattern of each window and serve as input to the ML model.

    Features extracted per axis (ax, ay, az, gx, gy, gz):
    - mean: average signal level
    - std: signal variability
    - min, max: signal range
    - range: max - min
    - rms: root mean square (signal energy)
    - zero_crossing_rate: how often signal crosses zero (movement frequency)

    Additional combined features:
    - accel_magnitude_mean: mean of sqrt(ax²+ay²+az²) — overall movement intensity
    - gyro_magnitude_mean: mean of sqrt(gx²+gy²+gz²) — overall rotation intensity
    - accel_magnitude_std: variability of overall movement

    Frequency domain (via FFT on accel_magnitude):
    - dominant_frequency: frequency with highest energy
    - spectral_energy: total energy in frequency domain

    Args:
        window_df: DataFrame from apply_sliding_window() where each row is one window.

    Returns:
        DataFrame where each row is one window with all extracted features as columns,
        plus subject_id and exercise_name (label) columns.
    """
    # Early-return on empty input preserves the schema-aware behaviour of the
    # upstream windowing step.
    if window_df.empty:
        return pd.DataFrame()

    # Build feature dicts in a list and construct the DataFrame once at the
    # end — same O(n²) avoidance pattern as in the loader and windowing
    # modules.
    rows: list[dict[str, float | str | int]] = [
        _row_features(row) for _, row in window_df.iterrows()
    ]

    feature_df = pd.DataFrame(rows)

    # Move the identifier columns to the front so the file is easy to read
    # by a human; numeric feature columns follow in their insertion order
    # (deterministic because dicts preserve order in Python 3.7+).
    identifier_cols = ["subject_id", "exercise_name", "window_id"]
    feature_cols = [c for c in feature_df.columns if c not in identifier_cols]
    return feature_df[identifier_cols + feature_cols]
