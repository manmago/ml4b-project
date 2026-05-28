"""Sliding window segmentation for the ML4B exercise recognition pipeline.

This module is the second stage of the data preparation pipeline. It takes
the long-format DataFrame produced by :func:`ml4b.data.loader.load_recofit_raw`
and segments each continuous sensor recording into fixed-length, optionally
overlapping windows. Each resulting window is a single training sample.

Windows are produced strictly within one (subject_id, exercise_name,
recording_id) group — we never let a window straddle two recordings, two
exercises, or two subjects, because that would mix unrelated motion patterns
and corrupt the labels.

Window size of 2 s (100 samples at 50 Hz) and 50% overlap chosen — see ADR-006.
"""

import pandas as pd


def apply_sliding_window(
    df: pd.DataFrame,
    window_size: int = 100,
    overlap: float = 0.5,
    sampling_rate: int = 50,
) -> pd.DataFrame:
    """Apply sliding window segmentation to raw sensor time series.

    Segments continuous sensor recordings into fixed-size windows.
    Each window becomes one training sample for the ML model.
    Windows are created per subject and per recording to avoid
    mixing data across different exercise sessions.

    Window size of 2 s and 50% overlap chosen — see ADR-006.

    Args:
        df: Raw DataFrame from load_recofit_raw() with columns
            [subject_id, exercise_name, recording_id, timestamp, ax, ay, az, gx, gy, gz]
        window_size: Number of samples per window. Default 100 = 2 seconds at 50Hz.
        overlap: Fraction of overlap between consecutive windows. Default 0.5 = 50%.
        sampling_rate: Sensor sampling rate in Hz. Default 50.

    Returns:
        DataFrame where each row is one window, with columns:
        [subject_id, exercise_name, window_id, raw_ax, raw_ay, raw_az,
         raw_gx, raw_gy, raw_gz]
        where raw_* columns contain lists of values for that window.
    """
    # Guard against nonsensical configuration that would create an infinite or
    # zero-length stride loop.
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    # Step size is how many samples we advance per window. With 50% overlap on
    # a 100-sample window we step by 50 samples, doubling the sample count
    # versus non-overlapping windows.
    step = max(1, int(round(window_size * (1 - overlap))))

    # Group by (subject, exercise, recording) so windows never straddle a
    # boundary between two unrelated time series — a window crossing two
    # recordings would have an artificial discontinuity in the middle.
    group_cols = ["subject_id", "exercise_name", "recording_id"]

    windows: list[dict] = []
    # Global counter so every window has a unique id across the whole dataset;
    # downstream code uses this as a stable join key when needed.
    window_id = 0

    # sort=False preserves insertion order, which keeps results reproducible
    # for a given input ordering without paying the cost of an extra sort.
    for (subject_id, exercise_name, _recording_id), group in df.groupby(
        group_cols, sort=False
    ):
        # Convert each axis to a numpy array once per group — list-slicing a
        # Series inside the inner loop is ~10x slower.
        ax = group["ax"].to_numpy()
        ay = group["ay"].to_numpy()
        az = group["az"].to_numpy()
        gx = group["gx"].to_numpy()
        gy = group["gy"].to_numpy()
        gz = group["gz"].to_numpy()
        n_samples = len(ax)

        # Skip recordings that are shorter than one window — they cannot
        # produce a full-length sample and we never want partial windows.
        if n_samples < window_size:
            continue

        # Walk a window of `window_size` samples across the recording,
        # advancing by `step` samples each iteration. The final start index is
        # n_samples - window_size, which guarantees the last window is full
        # and we drop any trailing remainder shorter than `window_size`.
        for start in range(0, n_samples - window_size + 1, step):
            end = start + window_size

            # Store raw signals as Python lists. This is convenient because
            # pandas can write a DataFrame of lists to pickle/parquet, and
            # downstream feature extraction does not need numpy semantics yet.
            windows.append(
                {
                    "subject_id": subject_id,
                    "exercise_name": exercise_name,
                    "window_id": window_id,
                    "raw_ax": ax[start:end].tolist(),
                    "raw_ay": ay[start:end].tolist(),
                    "raw_az": az[start:end].tolist(),
                    "raw_gx": gx[start:end].tolist(),
                    "raw_gy": gy[start:end].tolist(),
                    "raw_gz": gz[start:end].tolist(),
                }
            )
            window_id += 1

    # Build the final DataFrame once at the end — same O(n²)-avoidance reason
    # as in loader.py.
    columns = [
        "subject_id",
        "exercise_name",
        "window_id",
        "raw_ax",
        "raw_ay",
        "raw_az",
        "raw_gx",
        "raw_gy",
        "raw_gz",
    ]
    # sampling_rate is part of the public signature for clarity at call
    # sites (it documents the timing assumption) even though the windowing
    # logic itself works in sample counts only.
    _ = sampling_rate

    if not windows:
        # Return an empty frame with the correct schema rather than raising —
        # an empty result is meaningful (e.g. all recordings too short) and
        # callers can detect it with `len(df) == 0`.
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(windows, columns=columns)
