"""Energy-threshold activity gate — detects rest WITHOUT a trained class.

Real gym recordings contain long pauses between sets. Earlier models learned a
``rest`` class from the training data, but a learned rest boundary transfers
badly across devices and people: a new user's idle behaviour (fidgeting,
adjusting the watch, drinking) looks nothing like the training set's rest, so
rest gets massively over-predicted on real Apple-Watch uploads.

This module takes a more robust approach (DECISIONS.md): rest is a *low-energy*
physical state, so we detect it with a simple, device-agnostic energy threshold
instead of a classifier. A window is "active" (a real exercise) when either:

* the standard deviation of the accelerometer magnitude exceeds
  :data:`ACCEL_MAG_STD_THRESHOLD` g  (the watch is being moved, not held still),
  **or**
* the mean gyroscope magnitude exceeds :data:`GYRO_MAG_MEAN_THRESHOLD` rad/s
  (the forearm is rotating).

Windows that fail both tests are labelled ``rest`` and never reach the model, so
the three trained classes only ever compete on genuine movement.

The thresholds are calibrated to sit well below the energy of the Kaggle
exercise windows (whose 1st-percentile energies are far above these values),
leaving a wide margin so true exercise is never gated out while genuinely still
pauses are. The same gate runs in training-data exploration and in the app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Label emitted for windows that fall below the activity thresholds.
REST_LABEL: str = "rest"

# Accelerometer-magnitude standard deviation (in g) below which the wrist is
# considered still. At rest |a| ≈ 1 g and barely varies (std ~0.01-0.03 g);
# during any of the target exercises it varies by several tenths of a g.
ACCEL_MAG_STD_THRESHOLD: float = 0.08

# Mean gyroscope-magnitude (rad/s) above which the forearm is clearly rotating.
# At rest the gyroscope reads near zero; exercise rotation is well above this.
GYRO_MAG_MEAN_THRESHOLD: float = 0.30


def window_energy(
    raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz
) -> tuple[float, float]:
    """Compute the two energy statistics used by the gate for one window.

    Args:
        raw_ax, raw_ay, raw_az: Accelerometer axis samples (lists or arrays, g).
        raw_gx, raw_gy, raw_gz: Gyroscope axis samples (lists or arrays, rad/s).

    Returns:
        Tuple ``(accel_mag_std, gyro_mag_mean)``: the standard deviation of the
        accelerometer magnitude and the mean of the gyroscope magnitude.
    """
    ax = np.asarray(raw_ax, dtype=float)
    ay = np.asarray(raw_ay, dtype=float)
    az = np.asarray(raw_az, dtype=float)
    gx = np.asarray(raw_gx, dtype=float)
    gy = np.asarray(raw_gy, dtype=float)
    gz = np.asarray(raw_gz, dtype=float)

    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    return float(np.std(accel_mag)), float(np.mean(gyro_mag))


def is_active(
    accel_mag_std: float,
    gyro_mag_mean: float,
    accel_std_threshold: float = ACCEL_MAG_STD_THRESHOLD,
    gyro_mean_threshold: float = GYRO_MAG_MEAN_THRESHOLD,
) -> bool:
    """Decide whether a window contains real movement.

    Args:
        accel_mag_std: Std of the accelerometer magnitude for the window (g).
        gyro_mag_mean: Mean of the gyroscope magnitude for the window (rad/s).
        accel_std_threshold: Accel-std cutoff. Defaults to the module constant.
        gyro_mean_threshold: Gyro-mean cutoff. Defaults to the module constant.

    Returns:
        ``True`` if the window is active (exercise), ``False`` if it is rest.
    """
    # Either enough linear shake OR enough rotation marks the window active.
    return accel_mag_std > accel_std_threshold or gyro_mag_mean > gyro_mean_threshold


def gate_window_df(
    window_df: pd.DataFrame,
    accel_std_threshold: float = ACCEL_MAG_STD_THRESHOLD,
    gyro_mean_threshold: float = GYRO_MAG_MEAN_THRESHOLD,
) -> pd.Series:
    """Compute a boolean active/rest mask for every window in a window frame.

    Args:
        window_df: Windowed DataFrame with ``raw_a*``/``raw_g*`` list columns.
        accel_std_threshold: Accel-std cutoff. Defaults to the module constant.
        gyro_mean_threshold: Gyro-mean cutoff. Defaults to the module constant.

    Returns:
        Boolean Series aligned to ``window_df.index``; ``True`` = active window.
    """
    active_flags: list[bool] = []
    for _, row in window_df.iterrows():
        accel_std, gyro_mean = window_energy(
            row["raw_ax"],
            row["raw_ay"],
            row["raw_az"],
            row["raw_gx"],
            row["raw_gy"],
            row["raw_gz"],
        )
        active_flags.append(
            is_active(accel_std, gyro_mean, accel_std_threshold, gyro_mean_threshold)
        )
    return pd.Series(active_flags, index=window_df.index, name="is_active")
