"""Device-invariant feature extraction for the 3-class Apple-Watch model.

This is the feature stage used by BOTH the training pipeline and the Streamlit
inference pipeline (the project's reuse rule). It replaces the earlier
per-axis ``ml4b.data.features`` for the final model because raw per-axis
statistics are sensitive to exactly how the watch sits on the wrist, which
differs between people and between the training watch and a new user's watch.

The features here are deliberately robust to that nuisance variation:

* **Magnitude features** — computed on the rotation-invariant accelerometer and
  gyroscope magnitudes ``|a| = sqrt(ax²+ay²+az²)`` and ``|g| = sqrt(gx²+gy²+gz²)``.
  These carry the movement *intensity* and *cadence* without depending on watch
  orientation (a rotation of the device leaves a magnitude unchanged).
* **Per-window z-normalized shape features** — each axis is standardized within
  the window (mean removed, divided by std) before computing shape-only
  descriptors (zero-crossing rate, dominant frequency). Standardizing per window
  removes the per-device DC offset and gain, so only the *pattern* of motion
  survives, not its absolute scale.
* **Axis-pair correlations** — Pearson correlations between axes capture how the
  movement is coordinated across axes; correlation is invariant to per-axis
  offset and scale, so it transfers across devices.
* **Spectral features** — dominant frequency and spectral energy of ``|a|``
  describe the repetition cadence, the single most informative cue separating a
  real exercise from a still pause.

See DECISIONS.md for the rationale and DECISIONS.md for the augmentation that complements
these features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4b.data.canonical import TARGET_HZ

# Raw list-columns produced by apply_sliding_window, grouped by sensor.
_ACCEL_AXES: tuple[str, ...] = ("ax", "ay", "az")
_GYRO_AXES: tuple[str, ...] = ("gx", "gy", "gz")
_ALL_AXES: tuple[str, ...] = _ACCEL_AXES + _GYRO_AXES

# Identifier columns carried through from windowing; never fed to the model.
_ID_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "exercise_name",
    "recording_id",
    "window_id",
)


def _zero_crossing_rate(signal: np.ndarray) -> float:
    """Fraction of adjacent sample pairs that change sign.

    Args:
        signal: 1-D array (expected mean-centered for a meaningful rate).

    Returns:
        Zero-crossing rate in ``[0, 1]``; ``0.0`` for signals shorter than 2.
    """
    if signal.size < 2:
        return 0.0
    signs = np.sign(signal)
    return float(np.mean(signs[:-1] != signs[1:]))


def _dominant_frequency(signal: np.ndarray, sampling_rate: int) -> float:
    """Frequency (Hz) of the strongest oscillation in a mean-centered signal.

    Args:
        signal: 1-D array for one window.
        sampling_rate: Samples per second.

    Returns:
        Dominant frequency in Hz, or ``0.0`` for trivially short signals.
    """
    n = signal.size
    if n < 2:
        return 0.0
    # Remove the DC component so a constant gravity offset cannot dominate.
    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return float(freqs[int(np.argmax(spectrum))])


def _magnitude_stats(
    mag: np.ndarray, prefix: str, sampling_rate: int
) -> dict[str, float]:
    """Amplitude + spectral statistics of a (rotation-invariant) magnitude signal.

    Args:
        mag: 1-D magnitude signal (``|a|`` or ``|g|``) for one window.
        prefix: Feature-name prefix, e.g. ``"accel_mag"`` or ``"gyro_mag"``.
        sampling_rate: Samples per second, for the spectral features.

    Returns:
        Dict of named scalar features summarising this magnitude signal.
    """
    mean_val = float(np.mean(mag))
    std_val = float(np.std(mag))
    centered = mag - mean_val
    spectrum = np.abs(np.fft.rfft(centered)) if mag.size >= 2 else np.zeros(1)
    return {
        # Overall motion intensity and its variability (rotation-invariant).
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_min": float(np.min(mag)),
        f"{prefix}_max": float(np.max(mag)),
        f"{prefix}_range": float(np.max(mag) - np.min(mag)),
        # RMS — energy-style summary that does not cancel signs.
        f"{prefix}_rms": float(np.sqrt(np.mean(mag**2))),
        # Mean absolute deviation — robust spread measure.
        f"{prefix}_mad": float(np.mean(np.abs(centered))),
        # Cadence cues: how fast and how energetic the oscillation is.
        f"{prefix}_zcr": _zero_crossing_rate(centered),
        f"{prefix}_dom_freq": _dominant_frequency(mag, sampling_rate),
        f"{prefix}_spec_energy": float(np.sum(spectrum**2)),
    }


def _znorm(signal: np.ndarray) -> np.ndarray:
    """Standardize a signal to zero mean and unit variance (per window).

    Args:
        signal: 1-D array for one window.

    Returns:
        The standardized signal; returned mean-centered only if std is ~0.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    # Avoid divide-by-zero on a flat signal; centering alone is fine then.
    if std < 1e-8:
        return signal - mean
    return (signal - mean) / std


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two signals, robust to flat inputs.

    Args:
        a: First 1-D signal.
        b: Second 1-D signal.

    Returns:
        Correlation in ``[-1, 1]``; ``0.0`` if either signal is constant.
    """
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_invariant_features(window_df: pd.DataFrame) -> pd.DataFrame:
    """Compute device-invariant features for every window.

    Args:
        window_df: Windowed DataFrame from
            :func:`ml4b.data.windowing.apply_sliding_window` (rows hold the
            ``raw_a*``/``raw_g*`` lists plus the identifier columns).

    Returns:
        DataFrame with one row per window: the identifier columns that were
        present in the input followed by the numeric feature columns. Returns an
        empty DataFrame for empty input.
    """
    if window_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | str | int]] = []
    for _, row in window_df.iterrows():
        # Materialize each axis once as a float array.
        arr = {ax: np.asarray(row[f"raw_{ax}"], dtype=float) for ax in _ALL_AXES}

        # Rotation-invariant magnitudes.
        accel_mag = np.sqrt(arr["ax"] ** 2 + arr["ay"] ** 2 + arr["az"] ** 2)
        gyro_mag = np.sqrt(arr["gx"] ** 2 + arr["gy"] ** 2 + arr["gz"] ** 2)

        feats: dict[str, float | str | int] = {}
        # Amplitude + spectral stats on both magnitudes.
        feats.update(_magnitude_stats(accel_mag, "accel_mag", TARGET_HZ))
        feats.update(_magnitude_stats(gyro_mag, "gyro_mag", TARGET_HZ))

        # Per-window z-normalized shape features per axis (offset/scale removed).
        for ax in _ALL_AXES:
            z = _znorm(arr[ax])
            feats[f"{ax}_zcr"] = _zero_crossing_rate(z)
            feats[f"{ax}_dom_freq"] = _dominant_frequency(z, TARGET_HZ)

        # Axis-pair correlations — coordination structure, scale/offset-invariant.
        feats["corr_ax_ay"] = _safe_corr(arr["ax"], arr["ay"])
        feats["corr_ax_az"] = _safe_corr(arr["ax"], arr["az"])
        feats["corr_ay_az"] = _safe_corr(arr["ay"], arr["az"])
        feats["corr_gx_gy"] = _safe_corr(arr["gx"], arr["gy"])
        feats["corr_gx_gz"] = _safe_corr(arr["gx"], arr["gz"])
        feats["corr_gy_gz"] = _safe_corr(arr["gy"], arr["gz"])

        # Ratio of rotational to linear motion — distinguishes the exercises'
        # rotation content (curl/triceps rotate the forearm; rows translate it).
        feats["gyro_accel_ratio"] = float(
            np.mean(gyro_mag) / (np.mean(accel_mag) + 1e-8)
        )

        # Carry through whichever identifier columns the input had.
        for id_col in _ID_COLUMNS:
            if id_col in row:
                feats[id_col] = row[id_col]
        rows.append(feats)

    feature_df = pd.DataFrame(rows)
    # Put identifier columns first for readability; features follow.
    id_cols = [c for c in _ID_COLUMNS if c in feature_df.columns]
    feature_cols = [c for c in feature_df.columns if c not in id_cols]
    return feature_df[id_cols + feature_cols]


def feature_columns(feature_df: pd.DataFrame) -> list[str]:
    """Return the model-input feature columns (everything but identifiers).

    Args:
        feature_df: Output of :func:`extract_invariant_features`.

    Returns:
        Ordered list of numeric feature column names.
    """
    return [c for c in feature_df.columns if c not in _ID_COLUMNS]
