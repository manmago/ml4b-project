"""Canonical Apple-Watch sensor representation shared by training and inference.

This module is the single source of truth for two things that MUST be identical
in the training pipeline (Kaggle Gym Workout IMU dataset) and in the Streamlit
app's inference pipeline (Sensor Logger uploads):

1. **Pipeline constants** — sampling rate, window size, overlap, lag trim, and
   the confidence threshold. Both ``scripts/train_model.py`` and
   ``ml4b.data.apple_watch_loader`` import these so the two pipelines can never
   drift apart (the project's core architectural rule).

2. **Channel canonicalization** — both data sources are Apple CoreMotion
   ``DeviceMotion`` streams. CoreMotion reports *user* acceleration (gravity
   removed) in units of g, a separate gravity vector in g, and rotation rate in
   rad/s. We canonicalize every recording to six channels
   ``[ax, ay, az, gx, gy, gz]`` where:

   * ``ax/ay/az`` = **total** acceleration in g  (userAcceleration + gravity).
     Total acceleration is always reconstructable and keeps the gravity
     direction, which encodes arm posture; rotation augmentation
     (:mod:`ml4b.data.augmentation`) then teaches the model to be invariant to
     the exact watch orientation.
   * ``gx/gy/gz`` = rotation rate in rad/s, passed through unchanged.

Because the Kaggle dataset (Apple Watch SE) and Sensor Logger (any Apple Watch)
use the *same* CoreMotion conventions, no cross-device unit conversion is needed
— this device-domain match is the whole reason the Kaggle dataset was chosen as
the training anchor (see DECISIONS.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Shared pipeline constants ------------------------------------------------
# Native sampling rate of both the Kaggle dataset and Apple Watch DeviceMotion.
TARGET_HZ: int = 100
# Window length in samples. 200 samples @ 100 Hz = 2.0 s, long enough to contain
# at least one full repetition of the target exercises — see DECISIONS.md.
WINDOW_SIZE: int = 200
# Fraction of overlap between consecutive windows.
OVERLAP: float = 0.5
# Apple Watch CoreMotion needs ~0.24 s to start emitting valid samples; the
# leading rows of every Kaggle file are NaN. We trim this lag — see Phase 1.
LAG_SECONDS: float = 0.24
# Below this maximum class probability the prediction is reported as
# "uncertain" rather than forcing one of the three classes — see DECISIONS.md.
CONFIDENCE_THRESHOLD: float = 0.50

# The canonical six-channel schema every loader must output (plus a timestamp).
CANONICAL_CHANNELS: list[str] = ["ax", "ay", "az", "gx", "gy", "gz"]


def reconstruct_total_acceleration(
    df: pd.DataFrame,
    user_accel_cols: tuple[str, str, str],
    gravity_cols: tuple[str, str, str] | None,
) -> pd.DataFrame:
    """Reconstruct total acceleration in g from CoreMotion components.

    CoreMotion splits the measured force into *user* acceleration (gravity
    removed) and a gravity vector. Adding them back gives total acceleration —
    what a raw accelerometer would read — which is robust and always available.

    Args:
        df: Source DataFrame containing the CoreMotion columns.
        user_accel_cols: The (x, y, z) user-acceleration column names (units g).
        gravity_cols: The (x, y, z) gravity column names (units g), or ``None``
            if the export has no gravity channels. When ``None``, user
            acceleration is used unchanged (a minor approximation, logged by the
            caller).

    Returns:
        A DataFrame with three columns ``ax, ay, az`` holding total acceleration
        in g, aligned to ``df``'s index.
    """
    ux, uy, uz = user_accel_cols
    # Without a gravity vector we cannot reconstruct total acceleration, so fall
    # back to user acceleration. This keeps odd exports working; the invariant
    # features and rotation augmentation tolerate the small resulting offset.
    if gravity_cols is None:
        return pd.DataFrame(
            {
                "ax": df[ux].to_numpy(dtype=float),
                "ay": df[uy].to_numpy(dtype=float),
                "az": df[uz].to_numpy(dtype=float),
            }
        )
    gxc, gyc, gzc = gravity_cols
    # Total = user acceleration + gravity, component-wise, kept in g.
    return pd.DataFrame(
        {
            "ax": df[ux].to_numpy(dtype=float) + df[gxc].to_numpy(dtype=float),
            "ay": df[uy].to_numpy(dtype=float) + df[gyc].to_numpy(dtype=float),
            "az": df[uz].to_numpy(dtype=float) + df[gzc].to_numpy(dtype=float),
        }
    )


def resample_uniform(
    df: pd.DataFrame,
    target_hz: int = TARGET_HZ,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Resample irregular/odd-rate sensor data onto a uniform target-Hz grid.

    Apple Watch recordings are nominally 100 Hz but timestamps jitter, and some
    Sensor Logger exports use a different rate. Linear interpolation onto an
    evenly spaced time grid makes the windowing step rate-correct regardless of
    the source rate (it both up- and down-samples, unlike plain decimation).

    Args:
        df: Canonical DataFrame with a ``timestamp`` column (seconds) and the
            six channel columns.
        target_hz: Target sampling rate in Hz. Defaults to :data:`TARGET_HZ`.
        timestamp_col: Name of the seconds-valued timestamp column.

    Returns:
        A new DataFrame sampled uniformly at ``target_hz`` over the original
        time span, with the same channel columns. Returned unchanged if there
        are fewer than two samples.
    """
    # Need at least two points to define a time span to interpolate over.
    if len(df) < 2:
        return df.reset_index(drop=True)

    t = df[timestamp_col].to_numpy(dtype=float)
    t0, t1 = t[0], t[-1]
    # Guard against a degenerate (zero-duration) recording.
    if t1 <= t0:
        return df.reset_index(drop=True)

    # Build the evenly spaced grid: one sample every 1/target_hz seconds.
    n_samples = int(round((t1 - t0) * target_hz)) + 1
    grid = np.linspace(t0, t1, n_samples)

    # Linearly interpolate each channel onto the grid.
    out = {timestamp_col: grid}
    for col in CANONICAL_CHANNELS:
        out[col] = np.interp(grid, t, df[col].to_numpy(dtype=float))
    return pd.DataFrame(out)
